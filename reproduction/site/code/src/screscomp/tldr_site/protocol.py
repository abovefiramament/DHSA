from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROTOCOL_ID = "tldr_site_reft_audit_fixed_protocol_20260721_v2"
PROTOCOL_VERSION = 2
SELECTORS = (
    "rcm_zero_signed",
    "rcm_patch_signed",
    "iti",
    "random",
)
CANDIDATE_IDS = ("candidate_00", "candidate_01", "candidate_02")


class ProtocolError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_float_list(values: Any) -> list[float]:
    return [float(value) for value in values]


@dataclass(frozen=True, slots=True)
class TldrSitePaths:
    root: Path

    @property
    def protocol_dir(self) -> Path:
        return self.root / "protocol"

    @property
    def input_dir(self) -> Path:
        return self.root / "input"

    @property
    def selector_pairs(self) -> Path:
        return self.input_dir / "selector_pairs_300.csv"

    @property
    def selector_admission(self) -> Path:
        return self.input_dir / "selector_pair_admission.jsonl"

    @property
    def selector_manifest(self) -> Path:
        return self.input_dir / "selector_pair_manifest.json"

    @property
    def shared_native_margins(self) -> Path:
        return self.root / "selectors" / "shared" / "native_margins.jsonl"

    @property
    def shared_native_manifest(self) -> Path:
        return self.root / "selectors" / "shared" / "native_margins_manifest.json"

    def selector_dir(self, selector: str) -> Path:
        return self.root / "selectors" / selector

    def candidates_json(self, selector: str) -> Path:
        return self.selector_dir(selector) / "candidate_configurations.json"

    def candidates_csv(self, selector: str) -> Path:
        return self.selector_dir(selector) / "candidate_configurations.csv"

    def actuator_dir(self, selector: str, candidate_id: str, bank: str) -> Path:
        return self.root / "actuators" / selector / candidate_id / bank

    def calibration_sweep(self, selector: str, candidate_id: str) -> Path:
        return self.root / "calibration" / "generations" / selector / candidate_id / "sweep.jsonl"

    def calibration_alpha_dir(self) -> Path:
        return self.root / "calibration" / "alpha_points"

    @property
    def calibration_judge_dir(self) -> Path:
        return self.root / "calibration" / "judge"

    @property
    def calibration_health_csv(self) -> Path:
        return self.root / "calibration" / "health.csv"

    @property
    def calibration_selection(self) -> Path:
        return self.root / "calibration" / "frozen_selection.json"

    @property
    def calibration_selection_seal(self) -> Path:
        return self.root / "calibration" / "frozen_selection_seal.json"

    @property
    def final_dir(self) -> Path:
        return self.root / "final"

    @property
    def final_judge_dir(self) -> Path:
        return self.final_dir / "judge"

    @property
    def final_health_csv(self) -> Path:
        return self.final_dir / "health.csv"


class TldrSiteProtocol:
    def __init__(self, config_path: Path, data: dict[str, Any]) -> None:
        self.config_path = config_path.resolve()
        self.data = data
        self.config_sha256 = sha256_file(self.config_path)
        self.repo_root = self.config_path.parent.parent.resolve()
        self.paths = TldrSitePaths(Path(data["execution"]["run_root"]))

    @classmethod
    def load(cls, config_path: Path, *, validate_files: bool = False) -> "TldrSiteProtocol":
        path = config_path.resolve()
        data = json.loads(path.read_text(encoding="utf-8"))
        protocol = cls(path, data)
        protocol.validate()
        if validate_files:
            protocol.validate_input_files()
        return protocol

    @property
    def selectors(self) -> dict[str, Any]:
        return self.data["selection"]["selectors"]

    @property
    def alpha_grid(self) -> list[float]:
        return _as_float_list(self.data["actuator"]["alpha_grid"])

    @property
    def model_path(self) -> Path:
        return Path(self.data["model"]["path"])

    @property
    def pairs_path(self) -> Path:
        return Path(self.data["inputs"]["pairs_csv"])

    def validate(self) -> None:
        data = self.data
        require(data.get("protocol_id") == PROTOCOL_ID, "unexpected TLDR Site protocol id")
        require(data.get("version") == PROTOCOL_VERSION, "unexpected TLDR Site protocol version")
        require(data.get("status") == "locked_before_gpu_execution", "protocol is not locked")
        require(data.get("remote_source_of_truth") == "LOCAL_HOME/RPEC/projects/screscomp", "remote source drift")
        require(
            data.get("documentation") == "docs/TLDR_SITE_REFT_AUDIT_FIXED_PROTOCOL_20260721_V2.md",
            "documentation path drift",
        )
        require(
            data.get("preflight") == "scripts/paper/30_prepare_tldr_site_reft_audit_protocol.py",
            "preflight path drift",
        )
        require(
            data.get("runner") == "scripts/paper/31_run_tldr_site_reft_audit_serial.py",
            "runner path drift",
        )

        model = data["model"]
        architecture = model["expected_architecture"]
        require(model["revision"] == "5284f23cef13674869bdb95eea91cafd0fa6d124", "model revision drift")
        require(model["use_chat_template"] is False, "GPT-J may not use a chat template")
        require(model["torch_dtype"] == "bfloat16", "model dtype drift")
        require(architecture["attention_layers"] == 28, "layer count drift")
        require(architecture["attention_heads_per_layer"] == 16, "head count drift")
        require(architecture["head_dim"] == 256, "head dimension drift")
        require(architecture["candidate_heads"] == 448, "candidate universe drift")

        inputs = data["inputs"]
        groups = inputs["selector_groups"]
        require(inputs["event"] == "tldr_summary_preference", "event drift")
        require(groups["source_split"] == "train", "selector split drift")
        require(groups["source_order"] == "pairs_csv_file_order", "selector order drift")
        require(groups["shuffle"] is False, "selector groups may not be shuffled")
        require(groups["unique_group_key"] == "prompt_id", "selector group key drift")
        require(groups["target_unique_groups"] == 300, "selector sample count drift")
        require(inputs["calibration"]["rows"] == 320, "calibration size drift")
        require(inputs["final"]["rows"] == 320, "final size drift")
        require(inputs["calibration"]["sha256"] != inputs["final"]["sha256"], "clean splits overlap by hash")

        selection = data["selection"]
        require(selection["candidate_space"] == "pre_o_attention_head", "candidate space drift")
        require(selection["selected_count"] == 8, "K drift")
        require(
            selection["signed_partition"] == {"target_support_count": 4, "competitor_support_count": 4},
            "signed partition drift",
        )
        require(selection["state_timing"] == "generation_decision_states", "selector timing drift")
        require(tuple(selection["selectors"]) == SELECTORS, "selector set or order drift")
        require(selection["bootstrap"] == {
            "method": "paired_percentile_bootstrap",
            "samples": 500,
            "seed": 42,
            "percentiles": [2.5, 97.5],
        }, "bootstrap drift")
        ranking = selection["rcm_ranking"]
        require(ranking["target_support_score"] == "ci95_low", "positive ranking drift")
        require(ranking["competitor_support_score"] == "negative_ci95_high", "negative ranking drift")
        require(ranking["ci_admission_gate"] is False, "CI may not be an admission gate")
        require(ranking["ci_crossing_policy"] == "retain_and_report", "CI crossing policy drift")
        require(selection["coarse_to_fine"]["coarse_beam_per_role"] == 4, "coarse beam drift")
        require(selection["coarse_to_fine"]["coarse_groups"] == 300, "coarse sample drift")
        require(selection["coarse_to_fine"]["head_groups"] == 300, "head sample drift")
        require(self.selectors["random"]["seeds"] == [20260702, 20260703, 20260704], "Random seed drift")

        audit = data["common_position_audit"]
        require(audit["complete_configurations_per_selector"] == 3, "audit budget drift")
        require(audit["heads_per_configuration"] == 8, "audit K drift")
        require(audit["raw_configuration"] == "candidate_00", "raw configuration drift")
        require(audit["alternative_seeds"] == [20260703, 20260704], "audit alternative seed drift")
        require(audit["ranked_reservoir_multiplier"] == 2, "audit reservoir drift")
        require(audit["ranked_reservoir_sizes"] == {"rcm_per_role": 8, "iti_unsigned": 16}, "reservoir size drift")
        construction = audit["candidate_construction"]
        require("20260703" in construction["candidate_01"], "candidate_01 seed drift")
        require("20260704" in construction["candidate_02"], "candidate_02 seed drift")
        metric = audit["selection_metric"]
        require(metric["name"] == "balanced_human_pairwise_win_rate", "calibration metric drift")
        require(metric["health_used_for_selection"] is False, "health may not tune positions")

        actuator = data["actuator"]
        require(actuator["family"] == "cast_low_rank_reft", "actuator family drift")
        require(actuator["base_model_frozen"] is True, "base model must remain frozen")
        require(actuator["native_selected_writes"] == "kept_active", "native write policy drift")
        require(
            actuator["locator_operation_present_during_training_or_generation"] is False,
            "selector operations may not leak into ReFT training or generation",
        )
        require(actuator["reft_mode"] == "low_rank", "ReFT mode drift")
        require(actuator["rank"] == 4, "ReFT rank drift")
        require(actuator["initialization"] == {
            "init_std": 0.01,
            "warm_start": False,
            "seed": 42,
        }, "ReFT initialization drift")
        require(actuator["expected_total_trainable_parameters_per_K8_configuration"] == 16384, "parameter budget drift")
        partition = actuator["partition"]
        require(partition["rcm_zero_signed"] == "two_independently_trained_four_head_role_banks", "zero bank drift")
        require(partition["rcm_patch_signed"] == "two_independently_trained_four_head_role_banks", "patch bank drift")
        require(partition["iti"] == "one_jointly_trained_unsigned_eight_head_bank", "ITI bank drift")
        require(partition["random"] == "one_jointly_trained_unsigned_eight_head_bank", "Random bank drift")
        require(partition["shared_alpha_across_rcm_role_banks"] is True, "RCM alpha drift")
        require(partition["independent_role_alpha_search"] is False, "per-role alpha search is forbidden")
        training = actuator["training"]
        expected_training = {
            "pairs_csv": inputs["pairs_csv"],
            "event": inputs["event"],
            "endpoint_objective": "pair_margin",
            "train_split": "train",
            "val_split": "val",
            "shuffle_train": True,
            "max_train_rows": 8192,
            "max_val_rows": 512,
            "epochs": 1,
            "train_batch_size": 2,
            "eval_batch_size": 2,
            "optimizer": "torch.optim.AdamW",
            "lr": 0.001,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_eps": 1e-8,
            "weight_decay": 0.01,
            "amsgrad": False,
            "scheduler": "none",
            "gradient_clipping": "none",
            "lambda_norm": 0.0001,
            "alpha_train": 1.0,
            "preference_loss_mode": "dpo",
            "dpo_beta": 0.5,
            "state_margin_weight": 0.0,
            "gain_weight": 1.0,
            "target_margin": 0.0,
            "target_gain": 0.0,
            "apply_mode": "all",
            "score_mode": "avglogp",
            "option_selection_mode": "model_max",
            "max_aliases_per_side": 1,
            "empty_cache_every": 25,
            "seed": 42,
            "skip_alpha_summary": True,
        }
        for key, expected in expected_training.items():
            require(training[key] == expected, f"ReFT training drift: {key}")
        require(self.alpha_grid == [0.2, 0.3, 0.4, 0.5, 0.6, 0.7], "alpha grid drift")

        generation = data["generation"]
        require(generation["split"] == "test" and generation["start"] == 0, "generation row selection drift")
        require(generation["apply_mode"] == "all", "generation apply mode drift")
        require(generation["samples_per_prompt"] == 1, "generation sample count drift")
        require(generation["max_new_tokens"] == 100, "generation window drift")
        require(generation["stop_strings"] == [], "generation stop-string drift")
        require(generation["do_sample"] is False, "formal generation must be deterministic")
        require(float(generation["temperature"]) == 0.0, "temperature drift")
        require(float(generation["sign"]) == 1.0, "generation sign drift")
        require(generation["batch_size"] == 1, "generation batch drift")
        require(generation["same_seed_across_alpha"] is True, "alpha seed schedule drift")
        judge = data["evaluation"]["judge"]
        require(judge["expected_rows"] == 320, "judge row count drift")
        require(float(judge["review_fraction"]) == 1.0, "full order swap is required")
        require(judge["thinking_mode"] == "disabled", "judge thinking mode drift")

        execution = data["execution"]
        require(execution["serial_only"] is True, "execution must be serial")
        require(
            execution["run_root"]
            == "LOCAL_HOME/RPEC/projects/screscomp/runs/tldr_site_reft_audit_20260721_v2",
            "persistent run root drift",
        )
        require(execution["single_gpu_per_job"] is True, "only one GPU may be exposed")
        require(execution["preferred_physical_gpu_indices"] == [2, 3], "GPU set drift")
        require(execution["failure_policy"] == "stop_without_fallback_or_protocol_mutation", "failure policy drift")

    def validate_input_files(self) -> dict[str, Any]:
        inputs = self.data["inputs"]
        required = [
            self.model_path,
            Path(self.data["model"]["tokenizer_path"]),
            self.pairs_path,
            Path(inputs["pair_build_manifest"]),
            Path(inputs["calibration"]["path"]),
            Path(inputs["final"]["path"]),
            Path(self.selectors["iti"]["official_repository"]),
        ]
        missing = [str(path) for path in required if not path.exists()]
        require(not missing, f"missing fixed input paths: {missing}")
        require(self.pairs_path.stat().st_size == int(inputs["pairs_expected_bytes"]), "pairs byte size drift")
        pair_manifest = Path(inputs["pair_build_manifest"])
        pair_manifest_sha256 = sha256_file(pair_manifest)
        require(pair_manifest_sha256 == inputs["pair_build_manifest_sha256"], "pair manifest hash drift")
        require(sha256_file(Path(inputs["calibration"]["path"])) == inputs["calibration"]["sha256"], "calibration hash drift")
        require(sha256_file(Path(inputs["final"]["path"])) == inputs["final"]["sha256"], "final hash drift")
        return {
            "pairs_bytes": self.pairs_path.stat().st_size,
            "pair_manifest_sha256": pair_manifest_sha256,
            "calibration_sha256": inputs["calibration"]["sha256"],
            "final_sha256": inputs["final"]["sha256"],
        }

    def validate_component_id(self, component_id: str) -> None:
        pattern = self.data["selection"]["component_id_pattern"]
        require(re.fullmatch(pattern, component_id) is not None, f"invalid component id: {component_id}")
        layer_text, head_text = component_id.removeprefix("L").split(".attn.h", 1)
        architecture = self.data["model"]["expected_architecture"]
        require(0 <= int(layer_text) < int(architecture["attention_layers"]), f"invalid layer: {component_id}")
        require(0 <= int(head_text) < int(architecture["attention_heads_per_layer"]), f"invalid head: {component_id}")

    def snapshot(self) -> dict[str, Any]:
        return {
            "protocol_id": PROTOCOL_ID,
            "protocol_version": PROTOCOL_VERSION,
            "config": str(self.config_path),
            "config_sha256": self.config_sha256,
            "model": self.data["model"],
            "selection": self.data["selection"],
            "common_position_audit": self.data["common_position_audit"],
            "actuator": self.data["actuator"],
            "generation": self.data["generation"],
            "evaluation": self.data["evaluation"],
        }
