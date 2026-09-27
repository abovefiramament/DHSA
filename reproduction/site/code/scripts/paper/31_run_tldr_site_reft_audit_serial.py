#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from screscomp.data import dump_json  # noqa: E402
from screscomp.tldr_site.audit import (  # noqa: E402
    all_calibration_points,
    build_selector_overlap,
    finalize_tables,
    freeze_calibration_selection,
    load_frozen_selection,
    point_name,
    split_generation_sweep,
    unique_frozen_points,
)
from screscomp.tldr_site.protocol import CANDIDATE_IDS, SELECTORS, TldrSiteProtocol, require, sha256_file  # noqa: E402
from screscomp.tldr_site.selection import prepare_selector_pairs  # noqa: E402


STAGES = (
    "prepare",
    "selectors",
    "train",
    "calibration_generate",
    "calibration_evaluate",
    "freeze",
    "final_generate",
    "final_evaluate",
    "finalize",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the locked TLDR Site + ReFT audit protocol serially.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    return parser.parse_args()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SerialRunner:
    def __init__(self, protocol: TldrSiteProtocol) -> None:
        self.protocol = protocol
        self.paths = protocol.paths
        self.root = self.paths.root
        self.root.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.root / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.status_path = self.root / "status.tsv"
        if not self.status_path.exists():
            self.status_path.write_text("time\tstage\tstatus\tdetail\n", encoding="utf-8")

    def status(self, stage: str, status: str, detail: str = "") -> None:
        with self.status_path.open("a", encoding="utf-8") as stream:
            stream.write(f"{_now()}\t{stage}\t{status}\t{detail.replace(chr(9), ' ')}\n")
        print(f"[tldr-site-runner] stage={stage} status={status} {detail}", flush=True)

    def command(self, name: str, command: list[str], *, env: dict[str, str] | None = None) -> None:
        log_path = self.log_dir / f"{name}.log"
        command_manifest = self.log_dir / f"{name}.command.json"
        self.status(name, "running", f"log={log_path}")
        started = time.monotonic()
        started_at = _now()
        try:
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(f"\n[{_now()}] command={json.dumps(command)}\n")
                stream.flush()
                subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                    text=True,
                )
        except Exception as exc:
            elapsed = time.monotonic() - started
            dump_json(command_manifest, {
                "protocol_id": self.protocol.data["protocol_id"],
                "config_sha256": self.protocol.config_sha256,
                "name": name,
                "command": command,
                "started_at": started_at,
                "finished_at": _now(),
                "wall_seconds": elapsed,
                "status": "failed",
                "error_type": type(exc).__name__,
                "log": str(log_path),
            })
            self.status(name, "failed", f"wall_seconds={elapsed:.3f} error={type(exc).__name__} log={log_path}")
            raise
        elapsed = time.monotonic() - started
        dump_json(command_manifest, {
            "protocol_id": self.protocol.data["protocol_id"],
            "config_sha256": self.protocol.config_sha256,
            "name": name,
            "command": command,
            "started_at": started_at,
            "finished_at": _now(),
            "wall_seconds": elapsed,
            "status": "complete",
            "log": str(log_path),
        })
        self.status(name, "done", f"wall_seconds={elapsed:.3f} log={log_path}")

    def require_prepared(self) -> None:
        snapshot = self.paths.protocol_dir / "scientific_config.json"
        manifest_path = self.paths.protocol_dir / "run_manifest.json"
        require(snapshot.is_file() and manifest_path.is_file(), "prepare stage has not completed")
        require(sha256_file(snapshot) == self.protocol.config_sha256, "run-local config snapshot drift")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        require(manifest.get("config_sha256") == self.protocol.config_sha256, "run manifest config hash drift")
        require(
            manifest.get("selector_pair_manifest", {}).get("selector_pairs_sha256")
            == sha256_file(self.paths.selector_pairs),
            "run manifest selector-pair hash drift",
        )

    def _gpu_snapshot(self, stage: str) -> tuple[dict[str, Any], dict[str, str]]:
        execution = self.protocol.data["execution"]
        allowed = {int(value) for value in execution["preferred_physical_gpu_indices"]}
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).splitlines()
        rows = []
        for line in output:
            if not line.strip():
                continue
            index, uuid, name, used, total, utilization = (part.strip() for part in line.split(",", 5))
            if int(index) in allowed:
                rows.append({
                    "physical_index": int(index),
                    "uuid": uuid,
                    "name": name,
                    "memory_used_mib": int(used),
                    "memory_total_mib": int(total),
                    "utilization_percent": int(utilization),
                })
        require(len(rows) == len(allowed), f"could not inspect physical GPUs {sorted(allowed)}")
        selected = min(rows, key=lambda row: (row["memory_used_mib"], row["utilization_percent"], row["physical_index"]))
        process_output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).splitlines()
        processes = [line.strip() for line in process_output if line.strip().split(",", 1)[0].strip() == selected["uuid"]]
        snapshot = {
            "time": _now(),
            "stage": stage,
            "policy": execution["gpu_selection_policy"],
            "observed_candidates": rows,
            "selected": {**selected, "visible_compute_processes": processes, "co_tenancy_observed": bool(processes)},
            "co_tenancy_allowed": execution["gpu_co_tenancy_allowed"],
        }
        gpu_dir = self.root / "protocol" / "gpu"
        gpu_dir.mkdir(parents=True, exist_ok=True)
        dump_json(gpu_dir / f"{stage}_{int(time.time())}.json", snapshot)
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(selected["physical_index"])
        env["PYTHONPATH"] = f"{SRC}:{env.get('PYTHONPATH', '')}"
        return snapshot, env

    def prepare(self) -> None:
        stage = "prepare"
        self.status(stage, "running")
        self.paths.protocol_dir.mkdir(parents=True, exist_ok=True)
        snapshot = self.paths.protocol_dir / "scientific_config.json"
        if snapshot.exists():
            require(sha256_file(snapshot) == self.protocol.config_sha256, "run-local config snapshot drift")
        else:
            shutil.copy2(self.protocol.config_path, snapshot)
        require(sha256_file(snapshot) == self.protocol.config_sha256, "run-local config snapshot hash drift")
        preflight_dir = REPO_ROOT / self.protocol.data["execution"]["curated_results"] / "preflight"
        self.command(
            "prepare_preflight",
            [
                sys.executable,
                str(REPO_ROOT / self.protocol.data["preflight"]),
                "--config",
                str(self.protocol.config_path),
                "--out-dir",
                str(preflight_dir),
            ],
        )
        preflight_audit_path = preflight_dir / "protocol_audit.json"
        require(preflight_audit_path.is_file(), "preflight did not emit protocol_audit.json")
        preflight_audit = json.loads(preflight_audit_path.read_text(encoding="utf-8"))
        require(preflight_audit.get("config_sha256") == self.protocol.config_sha256, "preflight config hash drift")
        pair_manifest = prepare_selector_pairs(self.protocol)
        git_commit = subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(REPO_ROOT), "status", "--short"], text=True).splitlines()
        dump_json(self.paths.protocol_dir / "run_manifest.json", {
            **self.protocol.snapshot(),
            "run_root": str(self.root),
            "run_local_config": str(snapshot),
            "run_local_config_sha256": sha256_file(snapshot),
            "repository_commit": git_commit,
            "dirty_tree_manifest": dirty,
            "implementation_files": preflight_audit["implementation_files"],
            "preflight_audit": str(preflight_audit_path),
            "preflight_audit_sha256": sha256_file(preflight_audit_path),
            "selector_pair_manifest": pair_manifest,
            "created_at": _now(),
        })
        self.status(stage, "done", f"selector_groups={pair_manifest['selected_unique_groups']}")

    def selectors(self) -> None:
        require(self.paths.selector_pairs.is_file(), "prepare stage has not materialized selector pairs")
        _snapshot, env = self._gpu_snapshot("selectors")
        for selector in SELECTORS:
            self.command(
                f"selector_{selector}",
                [
                    sys.executable,
                    "-m",
                    "screscomp.cli.tldr_site_select_heads",
                    "--config",
                    str(self.protocol.config_path),
                    "--selector",
                    selector,
                    "--device",
                    "cuda:0",
                ],
                env=env,
            )
        for selector in SELECTORS:
            require(self.paths.candidates_json(selector).is_file(), f"selector incomplete: {selector}")
        build_selector_overlap(self.protocol)

    def _candidate(self, selector: str, candidate_id: str) -> dict[str, Any]:
        path = self.paths.candidates_json(selector)
        require(path.is_file(), f"missing candidate manifest: {path}")
        overlap_manifest_path = self.root / "selectors" / "selector_overlap_manifest.json"
        require(overlap_manifest_path.is_file(), "selector candidates are not sealed by the overlap manifest")
        overlap_manifest = json.loads(overlap_manifest_path.read_text(encoding="utf-8"))
        require(overlap_manifest.get("config_sha256") == self.protocol.config_sha256, "selector overlap config drift")
        sealed = {
            str(row["selector"]): (str(row["path"]), str(row["sha256"]))
            for row in overlap_manifest["source_candidate_manifests"]
        }
        require(selector in sealed, f"selector is absent from overlap seal: {selector}")
        sealed_path, sealed_hash = sealed[selector]
        require(sealed_path == str(path) and sealed_hash == sha256_file(path), f"candidate manifest hash drift: {selector}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        require(payload.get("config_sha256") == self.protocol.config_sha256, "candidate config hash drift")
        candidate_rows = Path(payload["candidate_rows"])
        require(candidate_rows.is_file(), f"missing candidate row table: {selector}")
        require(payload["candidate_rows_sha256"] == sha256_file(candidate_rows), f"candidate row hash drift: {selector}")
        matches = [row for row in payload["configurations"] if row["candidate_id"] == candidate_id]
        require(len(matches) == 1, f"missing or duplicate {selector}/{candidate_id}")
        return matches[0]

    def _banks(self, selector: str, candidate: dict[str, Any]) -> dict[str, list[str]]:
        heads = [str(value) for value in candidate["heads"]]
        roles = [str(value) for value in candidate["roles"]]
        if selector.startswith("rcm_"):
            banks = {
                "target_support": [head for head, role in zip(heads, roles, strict=True) if role == "target_support"],
                "competitor_support": [head for head, role in zip(heads, roles, strict=True) if role == "competitor_support"],
            }
            partition = self.protocol.data["selection"]["signed_partition"]
            require(
                len(banks["target_support"]) == int(partition["target_support_count"])
                and len(banks["competitor_support"]) == int(partition["competitor_support_count"]),
                f"signed RCM bank shape drift: {selector}",
            )
            return banks
        selected_count = int(self.protocol.data["selection"]["selected_count"])
        require(roles == ["unsigned"] * selected_count, f"unsigned bank role drift: {selector}")
        return {"unsigned": heads}

    def _validate_reft(self, output: Path, heads: list[str]) -> int:
        run_config = output / "run_config.json"
        payload_path = output / "cast_reft.pt"
        require(run_config.is_file() and payload_path.is_file(), f"incomplete ReFT training output: {output}")
        config = json.loads(run_config.read_text(encoding="utf-8"))
        expected = self.protocol.data["actuator"]["training"]
        require(config["model"] == str(self.protocol.model_path), f"ReFT model drift: {output}")
        require(config["pairs_csv"] == str(self.protocol.pairs_path), f"ReFT pair input drift: {output}")
        require(config["heads"].split(",") == heads, f"ReFT head order drift: {output}")
        require(config["reft_mode"] == "low_rank" and int(config["rank"]) == 4, "ReFT architecture drift")
        for key in (
            "event", "endpoint_objective", "train_split", "val_split",
            "max_train_rows", "max_val_rows", "epochs", "train_batch_size", "eval_batch_size",
            "lr", "adam_beta1", "adam_beta2", "adam_eps", "weight_decay", "amsgrad",
            "lambda_norm", "alpha_train", "preference_loss_mode", "dpo_beta",
            "state_margin_weight", "gain_weight", "target_margin", "target_gain", "apply_mode",
            "score_mode", "option_selection_mode", "max_aliases_per_side", "empty_cache_every", "seed",
            "skip_alpha_summary",
        ):
            require(config[key] == expected[key], f"ReFT run_config drift for {key}: {output}")
        require(config["shuffle"] == expected["shuffle_train"], f"ReFT shuffle drift: {output}")
        require(config["alpha_sweep"] == self.protocol.alpha_grid, f"ReFT alpha-grid record drift: {output}")
        initialization = self.protocol.data["actuator"]["initialization"]
        require(config["init_std"] == initialization["init_std"], f"ReFT init std drift: {output}")
        require(config["init_head_actuator"] is None, f"unexpected ReFT head warm start: {output}")
        require(config["init_fixed_actuator"] is None, f"unexpected ReFT fixed warm start: {output}")
        require(config["site_count"] == len(heads), f"ReFT site count drift: {output}")
        require(config["torch_dtype"] == self.protocol.data["model"]["torch_dtype"], f"ReFT dtype drift: {output}")
        import torch
        try:
            payload = torch.load(payload_path, map_location="cpu", weights_only=False)
        except TypeError:  # pragma: no cover
            payload = torch.load(payload_path, map_location="cpu")
        return sum(int(value.numel()) for value in payload["parameters"].values())

    def train(self) -> None:
        _snapshot, env = self._gpu_snapshot("train")
        spec = self.protocol.data["actuator"]["training"]
        alpha_text = ",".join(str(value) for value in self.protocol.alpha_grid)
        for selector in SELECTORS:
            for candidate_id in CANDIDATE_IDS:
                candidate = self._candidate(selector, candidate_id)
                total_parameters = 0
                bank_manifest = []
                for bank, heads in self._banks(selector, candidate).items():
                    output = self.paths.actuator_dir(selector, candidate_id, bank)
                    if not (output / "cast_reft.pt").is_file():
                        command = [
                            sys.executable,
                            "-m",
                            "screscomp.cli.cecm_train_cast_reft",
                            "--model", str(self.protocol.model_path),
                            "--pairs-csv", str(self.protocol.pairs_path),
                            "--event", spec["event"],
                            "--endpoint-objective", spec["endpoint_objective"],
                            "--heads", ",".join(heads),
                            "--reft-mode", self.protocol.data["actuator"]["reft_mode"],
                            "--rank", str(self.protocol.data["actuator"]["rank"]),
                            "--init-std", str(self.protocol.data["actuator"]["initialization"]["init_std"]),
                            "--train-split", spec["train_split"],
                            "--val-split", spec["val_split"],
                            "--max-train-rows", str(spec["max_train_rows"]),
                            "--max-val-rows", str(spec["max_val_rows"]),
                            "--epochs", str(spec["epochs"]),
                            "--train-batch-size", str(spec["train_batch_size"]),
                            "--eval-batch-size", str(spec["eval_batch_size"]),
                            "--lr", str(spec["lr"]),
                            "--adam-beta1", str(spec["adam_beta1"]),
                            "--adam-beta2", str(spec["adam_beta2"]),
                            "--adam-eps", str(spec["adam_eps"]),
                            "--weight-decay", str(spec["weight_decay"]),
                            "--lambda-norm", str(spec["lambda_norm"]),
                            "--alpha-train", str(spec["alpha_train"]),
                            "--preference-loss-mode", spec["preference_loss_mode"],
                            "--dpo-beta", str(spec["dpo_beta"]),
                            "--state-margin-weight", str(spec["state_margin_weight"]),
                            "--gain-weight", str(spec["gain_weight"]),
                            "--target-margin", str(spec["target_margin"]),
                            "--target-gain", str(spec["target_gain"]),
                            "--apply-mode", spec["apply_mode"],
                            "--score-mode", spec["score_mode"],
                            "--option-selection-mode", spec["option_selection_mode"],
                            "--max-aliases-per-side", str(spec["max_aliases_per_side"]),
                            "--alpha-sweep", alpha_text,
                            "--empty-cache-every", str(spec["empty_cache_every"]),
                            "--device", "cuda:0",
                            "--torch-dtype", self.protocol.data["model"]["torch_dtype"],
                            "--seed", str(spec["seed"]),
                            "--out-dir", str(output),
                        ]
                        if spec["shuffle_train"]:
                            command.append("--shuffle")
                        command.append("--amsgrad" if spec["amsgrad"] else "--no-amsgrad")
                        if spec["skip_alpha_summary"]:
                            command.append("--skip-alpha-summary")
                        self.command(f"train_{selector}_{candidate_id}_{bank}", command, env=env)
                    parameters = self._validate_reft(output, heads)
                    total_parameters += parameters
                    bank_manifest.append({
                        "bank": bank,
                        "heads": heads,
                        "payload": str(output / "cast_reft.pt"),
                        "payload_sha256": sha256_file(output / "cast_reft.pt"),
                        "trainable_parameters": parameters,
                    })
                expected_parameters = int(
                    self.protocol.data["actuator"]["expected_total_trainable_parameters_per_K8_configuration"]
                )
                require(
                    total_parameters == expected_parameters,
                    f"parameter count drift for {selector}/{candidate_id}: {total_parameters}",
                )
                dump_json(self.root / "actuators" / selector / candidate_id / "actuator_manifest.json", {
                    "protocol_id": self.protocol.data["protocol_id"],
                    "config_sha256": self.protocol.config_sha256,
                    "selector": selector,
                    "candidate_id": candidate_id,
                    "partition": self.protocol.data["actuator"]["partition"][selector],
                    "banks": bank_manifest,
                    "total_trainable_parameters": total_parameters,
                    "optimizer_steps_per_bank": (
                        (int(spec["max_train_rows"]) + int(spec["train_batch_size"]) - 1)
                        // int(spec["train_batch_size"])
                    ) * int(spec["epochs"]),
                    "total_optimizer_steps": (
                        (int(spec["max_train_rows"]) + int(spec["train_batch_size"]) - 1)
                        // int(spec["train_batch_size"])
                    ) * int(spec["epochs"]) * len(bank_manifest),
                    "compute_matching_note": (
                        "RCM has two full-budget role-bank runs; unsigned controls have one. "
                        "Parameter count is matched, optimizer steps are reported and are not compute matched."
                    ),
                    "status": "complete",
                })

    def _actuator_paths(self, selector: str, candidate_id: str) -> list[Path]:
        candidate = self._candidate(selector, candidate_id)
        paths = [self.paths.actuator_dir(selector, candidate_id, bank) / "cast_reft.pt" for bank in self._banks(selector, candidate)]
        require(all(path.is_file() for path in paths), f"missing actuator payload for {selector}/{candidate_id}")
        manifest_path = self.root / "actuators" / selector / candidate_id / "actuator_manifest.json"
        require(manifest_path.is_file(), f"missing actuator manifest for {selector}/{candidate_id}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        require(manifest.get("config_sha256") == self.protocol.config_sha256, "actuator manifest config drift")
        require(manifest.get("selector") == selector and manifest.get("candidate_id") == candidate_id, "actuator identity drift")
        sealed_payloads = [(str(row["payload"]), str(row["payload_sha256"])) for row in manifest["banks"]]
        require(sealed_payloads == [(str(path), sha256_file(path)) for path in paths], "actuator payload hash drift")
        require(
            int(manifest["total_trainable_parameters"])
            == int(self.protocol.data["actuator"]["expected_total_trainable_parameters_per_K8_configuration"]),
            "actuator parameter-count seal drift",
        )
        return paths

    def _generation_command(
        self,
        prompts: Path,
        output: Path,
        control_name: str,
        alphas: list[float],
        actuators: list[Path],
    ) -> list[str]:
        spec = self.protocol.data["generation"]
        command = [
            sys.executable,
            "-m",
            "screscomp.cli.run_cast_reft_generation",
            "--model", str(self.protocol.model_path),
            "--tokenizer", str(self.protocol.data["model"]["tokenizer_path"]),
            "--prompts-jsonl", str(prompts),
            "--out-jsonl", str(output),
            "--manifest-json", str(output.with_suffix(".manifest.json")),
            "--control-name", control_name,
            "--alpha-sweep", ",".join(str(value) for value in alphas),
            "--generation-apply-mode", spec["apply_mode"],
            "--split", spec["split"],
            "--start", str(spec["start"]),
            "--max-rows", str(self.protocol.data["inputs"]["calibration"]["rows"]),
            "--samples-per-prompt", str(spec["samples_per_prompt"]),
            "--generation-batch-size", str(spec["batch_size"]),
            "--max-new-tokens", str(spec["max_new_tokens"]),
            "--stop-strings", ",".join(spec["stop_strings"]),
            "--temperature", str(spec["temperature"]),
            "--top-p", str(spec["top_p"]),
            "--top-k", str(spec["top_k"]),
            "--seed", str(spec["seed"]),
            "--sign", str(spec["sign"]),
            "--device", "cuda:0",
            "--torch-dtype", self.protocol.data["model"]["torch_dtype"],
        ]
        command.append("--do-sample" if spec["do_sample"] else "--no-do-sample")
        if spec["same_seed_across_alpha"]:
            command.append("--same-seed-across-alpha")
        if actuators:
            command.extend(["--reft-actuator", str(actuators[0])])
            for path in actuators[1:]:
                command.extend(["--additional-reft-actuator", str(path)])
        return command

    def _generation_complete(
        self,
        prompts: Path,
        output: Path,
        control_name: str,
        alphas: list[float],
        actuators: list[Path],
        *,
        expected_rows: int,
    ) -> bool:
        manifest_path = output.with_suffix(".manifest.json")
        audit_path = output.with_suffix(".audit.json")
        if not output.is_file():
            require(not manifest_path.exists() and not audit_path.exists(), f"orphan generation metadata: {output}")
            return False

        prompt_rows = [row for row in load_json_rows(prompts) if str(row.get("split", "")) == "test"][:expected_rows]
        require(len(prompt_rows) == expected_rows, f"prompt count drift: {prompts}")
        prompt_ids = [str(row.get("sample_id") or row.get("prompt_id") or "") for row in prompt_rows]
        require(len(set(prompt_ids)) == expected_rows and all(prompt_ids), f"prompt identity drift: {prompts}")
        prompt_index = {
            sample_id: str(row.get("source_row_index", row.get("raw_index", "")))
            for sample_id, row in zip(prompt_ids, prompt_rows, strict=True)
        }

        rows = load_json_rows(output)
        expected_alpha = {float(value) for value in alphas}
        expected_actuators = [str(path) for path in actuators]
        keys: set[tuple[str, float, int]] = set()
        counts = {alpha: 0 for alpha in expected_alpha}
        seeds_by_sample: dict[str, set[str]] = {sample_id: set() for sample_id in prompt_ids}
        generation = self.protocol.data["generation"]
        for row in rows:
            sample_id = str(row.get("sample_id", ""))
            alpha = float(row.get("alpha", -999.0))
            sample_index = int(row.get("sample_index", -1))
            require(sample_id in prompt_index, f"unexpected generation sample: {output} {sample_id}")
            require(alpha in expected_alpha, f"unexpected generation alpha: {output} {alpha}")
            require(sample_index == 0, f"unexpected generation sample index: {output}")
            key = (sample_id, alpha, sample_index)
            require(key not in keys, f"duplicate generation key: {output} {key}")
            keys.add(key)
            counts[alpha] += 1
            seeds_by_sample[sample_id].add(str(row.get("batch_seed", "")))
            require(str(row.get("control_name", "")) == control_name, f"generation control drift: {output}")
            require(row.get("reft_actuator_paths", []) == expected_actuators, f"generation actuator drift: {output}")
            require(int(row.get("reft_bank_count", -1)) == len(actuators), f"generation bank-count drift: {output}")
            require(row.get("generation_apply_mode") == generation["apply_mode"], f"generation apply-mode drift: {output}")
            require(int(row.get("generation_do_sample", -1)) == int(generation["do_sample"]), f"generation sampling drift: {output}")
            require(float(row.get("generation_temperature", -1.0)) == float(generation["temperature"]), f"generation temperature drift: {output}")
            require(float(row.get("generation_top_p", -1.0)) == float(generation["top_p"]), f"generation top-p drift: {output}")
            require(int(row.get("generation_top_k", -1)) == int(generation["top_k"]), f"generation top-k drift: {output}")
            require(row.get("generation_stop_strings", []) == generation["stop_strings"], f"generation stop-string drift: {output}")
            require(int(row.get("max_new_tokens", -1)) == int(generation["max_new_tokens"]), f"generation window drift: {output}")
            require(int(row.get("same_seed_across_alpha", -1)) == 1, f"generation seed schedule drift: {output}")
            require(int(row.get("generation_batch_size", -1)) == int(generation["batch_size"]), f"generation batch drift: {output}")
            require(str(row.get("source_row_index", "")) == prompt_index[sample_id], f"generation source-row drift: {output}")

        require(all(len(values) <= 1 for values in seeds_by_sample.values()), f"alpha seed mismatch: {output}")
        expected_total = expected_rows * len(alphas)
        if len(rows) != expected_total:
            require(len(rows) < expected_total, f"too many generation rows: {output}")
            require(not manifest_path.exists() and not audit_path.exists(), f"incomplete sealed generation: {output}")
            return False
        require(all(value == expected_rows for value in counts.values()), f"generation alpha coverage drift: {output}")
        require(manifest_path.is_file(), f"complete generation lacks manifest: {output}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        require(manifest["prompts_jsonl"] == str(prompts), f"generation manifest prompt drift: {output}")
        require(manifest["out_jsonl"] == str(output), f"generation manifest output drift: {output}")
        require(manifest["reft_actuators"] == expected_actuators, f"generation manifest actuator drift: {output}")
        require([float(value) for value in manifest["alpha_sweep"]] == [float(value) for value in alphas], f"generation manifest alpha drift: {output}")
        require(int(manifest["prompt_rows"]) == expected_rows, f"generation manifest row drift: {output}")
        require(int(manifest["completed_rows"]) == expected_total, f"generation manifest completion drift: {output}")
        require(manifest["control_name"] == control_name, f"generation manifest control drift: {output}")
        require(manifest["generation_apply_mode"] == generation["apply_mode"], f"generation manifest apply drift: {output}")
        require(int(manifest["max_new_tokens"]) == int(generation["max_new_tokens"]), f"generation manifest window drift: {output}")
        require(manifest.get("stop_strings", []) == generation["stop_strings"], f"generation manifest stop-string drift: {output}")
        require(bool(manifest["do_sample"]) == bool(generation["do_sample"]), f"generation manifest sampling drift: {output}")
        require(int(manifest["seed"]) == int(generation["seed"]), f"generation manifest seed drift: {output}")
        require(bool(manifest["same_seed_across_alpha"]) is True, f"generation manifest alpha seed drift: {output}")

        seal = {
            "protocol_id": self.protocol.data["protocol_id"],
            "config_sha256": self.protocol.config_sha256,
            "prompts": str(prompts),
            "prompts_sha256": sha256_file(prompts),
            "output": str(output),
            "output_sha256": sha256_file(output),
            "generation_manifest": str(manifest_path),
            "generation_manifest_sha256": sha256_file(manifest_path),
            "actuators": [
                {"path": str(path), "sha256": sha256_file(path)} for path in actuators
            ],
            "control_name": control_name,
            "alphas": alphas,
            "actual_rows": len(rows),
            "status": "complete",
        }
        if audit_path.is_file():
            require(json.loads(audit_path.read_text(encoding="utf-8")) == seal, f"generation audit seal drift: {output}")
        else:
            dump_json(audit_path, seal)
        return True

    def calibration_generate(self) -> None:
        _snapshot, env = self._gpu_snapshot("calibration_generate")
        prompts = Path(self.protocol.data["inputs"]["calibration"]["path"])
        expected_rows = int(self.protocol.data["inputs"]["calibration"]["rows"])
        base = self.root / "calibration" / "base" / "sft.jsonl"
        if not self._generation_complete(prompts, base, "sft", [0.0], [], expected_rows=expected_rows):
            self.command(
                "calibration_generate_sft",
                self._generation_command(prompts, base, "sft", [0.0], []),
                env=env,
            )
        require(
            self._generation_complete(prompts, base, "sft", [0.0], [], expected_rows=expected_rows),
            "calibration SFT incomplete",
        )
        for selector in SELECTORS:
            for candidate_id in CANDIDATE_IDS:
                sweep = self.paths.calibration_sweep(selector, candidate_id)
                control_name = f"{selector}--{candidate_id}"
                actuators = self._actuator_paths(selector, candidate_id)
                if not self._generation_complete(
                    prompts, sweep, control_name, self.protocol.alpha_grid, actuators, expected_rows=expected_rows,
                ):
                    self.command(
                        f"calibration_generate_{selector}_{candidate_id}",
                        self._generation_command(
                            prompts,
                            sweep,
                            control_name,
                            self.protocol.alpha_grid,
                            actuators,
                        ),
                        env=env,
                    )
                require(
                    self._generation_complete(
                        prompts, sweep, control_name, self.protocol.alpha_grid, actuators, expected_rows=expected_rows,
                    ),
                    f"calibration generation incomplete: {selector}/{candidate_id}",
                )
                split_generation_sweep(self.protocol, selector, candidate_id, sweep)
        all_calibration_points(self.protocol)

    def _judge_command(
        self,
        prompts: Path,
        candidates: dict[str, Path],
        out_dir: Path,
        *,
        human_only: bool,
        all_pairs: bool,
    ) -> list[str]:
        judge = self.protocol.data["evaluation"]["judge"]
        command = [
            sys.executable,
            "-m",
            "screscomp.cli.evaluate_tldr_dpo_ds4",
            "--prompts-jsonl", str(prompts),
            "--out-dir", str(out_dir),
            "--expected-rows", str(judge["expected_rows"]),
            "--random-seed", str(judge["random_seed"]),
            "--model", judge["model"],
            "--thinking-mode", judge["thinking_mode"],
            "--reasoning-effort", judge["reasoning_effort"],
            "--review-fraction", str(judge["review_fraction"]),
        ]
        if human_only:
            command.append("--human-only")
        if all_pairs:
            command.append("--all-pairs")
        for name, path in candidates.items():
            require("_vs_" not in name and name != "human", f"invalid judge candidate name: {name}")
            command.extend(["--candidate", f"{name}={path}"])
        return command

    def _health_command(self, prompts: Path, candidates: dict[str, Path], out_csv: Path) -> list[str]:
        command = [
            sys.executable,
            "scripts/audit_tldr_text_health.py",
            "--prompts-jsonl", str(prompts),
            "--expected-rows", str(self.protocol.data["evaluation"]["judge"]["expected_rows"]),
            "--out-csv", str(out_csv),
            "--templates-json", str(out_csv.with_suffix(".templates.json")),
        ]
        for name, path in candidates.items():
            command.extend(["--candidate", f"{name}={path}"])
        return command

    def calibration_evaluate(self) -> None:
        points = all_calibration_points(self.protocol)
        prompts = Path(self.protocol.data["inputs"]["calibration"]["path"])
        self.command(
            "calibration_judge",
            self._judge_command(prompts, points, self.paths.calibration_judge_dir, human_only=True, all_pairs=False),
        )
        self.command(
            "calibration_health",
            self._health_command(prompts, points, self.paths.calibration_health_csv),
        )

    def freeze(self) -> None:
        payload = freeze_calibration_selection(self.protocol)
        self.status("freeze", "done", f"selection_sha256={sha256_file(self.paths.calibration_selection)} selectors={len(payload['selections'])}")

    def final_generate(self) -> None:
        load_frozen_selection(self.protocol)
        _snapshot, env = self._gpu_snapshot("final_generate")
        prompts = Path(self.protocol.data["inputs"]["final"]["path"])
        expected_rows = int(self.protocol.data["inputs"]["final"]["rows"])
        out_dir = self.paths.final_dir / "generations"
        base = out_dir / "sft.jsonl"
        if not self._generation_complete(prompts, base, "sft", [0.0], [], expected_rows=expected_rows):
            command = self._generation_command(prompts, base, "sft", [0.0], [])
            command[command.index("--max-rows") + 1] = str(self.protocol.data["inputs"]["final"]["rows"])
            self.command("final_generate_sft", command, env=env)
        require(
            self._generation_complete(prompts, base, "sft", [0.0], [], expected_rows=expected_rows),
            "final SFT incomplete",
        )
        for name, row in unique_frozen_points(self.protocol).items():
            output = out_dir / f"{name}.jsonl"
            actuators = self._actuator_paths(str(row["selector"]), str(row["candidate_id"]))
            command = self._generation_command(
                prompts,
                output,
                name,
                [float(row["alpha"])],
                actuators,
            )
            command[command.index("--max-rows") + 1] = str(self.protocol.data["inputs"]["final"]["rows"])
            if not self._generation_complete(
                prompts, output, name, [float(row["alpha"])], actuators, expected_rows=expected_rows,
            ):
                self.command(f"final_generate_{name}", command, env=env)
            require(
                self._generation_complete(
                    prompts, output, name, [float(row["alpha"])], actuators, expected_rows=expected_rows,
                ),
                f"final generation incomplete: {name}",
            )

    def final_evaluate(self) -> None:
        points = unique_frozen_points(self.protocol)
        generation_dir = self.paths.final_dir / "generations"
        candidates = {"sft": generation_dir / "sft.jsonl"}
        candidates.update({name: generation_dir / f"{name}.jsonl" for name in points})
        require(all(path.is_file() for path in candidates.values()), "final generation set is incomplete")
        prompts = Path(self.protocol.data["inputs"]["final"]["path"])
        self.command(
            "final_judge",
            self._judge_command(prompts, candidates, self.paths.final_judge_dir, human_only=False, all_pairs=True),
        )
        self.command(
            "final_health",
            self._health_command(prompts, candidates, self.paths.final_health_csv),
        )

    def finalize(self) -> None:
        manifest = finalize_tables(self.protocol)
        curated = REPO_ROOT / self.protocol.data["execution"]["curated_results"]
        curated.mkdir(parents=True, exist_ok=True)
        for source in (
            self.paths.calibration_selection,
            self.paths.calibration_selection_seal,
            self.paths.final_dir / "raw_selector_table.csv",
            self.paths.final_dir / "common_audit_table.csv",
            self.paths.final_dir / "pairwise_matrix_long.csv",
            self.paths.final_dir / "final_manifest.json",
            self.paths.final_health_csv,
            self.root / "selectors" / "selector_overlap.csv",
            self.root / "selectors" / "selector_overlap_manifest.json",
        ):
            shutil.copy2(source, curated / source.name)
        dump_json(curated / "source_run.json", {
            "protocol_id": self.protocol.data["protocol_id"],
            "config_sha256": self.protocol.config_sha256,
            "source_run_root": str(self.root),
            "final_manifest": manifest,
        })
        self.status("finalize", "done", f"curated={curated}")


def load_json_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    args = parse_args()
    protocol = TldrSiteProtocol.load(args.config, validate_files=False)
    require(protocol.repo_root == REPO_ROOT, "runner must execute from the remote source repository")
    runner = SerialRunner(protocol)
    selected = STAGES if args.stage == "all" else (args.stage,)
    for stage in selected:
        if stage != "prepare":
            runner.require_prepared()
        getattr(runner, stage)()


if __name__ == "__main__":
    main()
