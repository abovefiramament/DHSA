from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


PROTOCOL_ID = "site_position_full_fixed_protocol_20260719_v12"
PROTOCOL_VERSION = 12
SELECTORS = ("rcm_zero_signed", "rcm_patch_signed", "iti", "random")


class ProtocolError(ValueError):
    """Raised before work begins when an artifact disagrees with the lock."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SiteJob:
    job_id: str
    dataset: str
    subset: str
    model_key: str
    model: str
    model_revision: str
    use_chat_template: bool
    selector: str
    replicate: str
    seed: int
    output_dir: Path
    selector_spec: dict[str, Any]
    task: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SitePaths:
    root: Path
    job: SiteJob

    @property
    def shared_dataset_dir(self) -> Path:
        if self.job.dataset == "confiqa":
            return self.root / "confiqa" / self.job.subset
        return self.root / "imdb" / self.job.model_key

    @property
    def source_dir(self) -> Path:
        return self.shared_dataset_dir / "source"

    @property
    def open_rows(self) -> Path:
        return self.shared_dataset_dir / "open_rows.jsonl"

    @property
    def selector_source_audit_dir(self) -> Path:
        return self.shared_dataset_dir / "selector_source_audit"

    @property
    def selector_source_audit_manifest(self) -> Path:
        return self.selector_source_audit_dir / "sample_admission_manifest.json"

    @property
    def selector_prompts(self) -> Path:
        return self.shared_dataset_dir / "selector_prompts.jsonl"

    @property
    def alpha_dev_rows(self) -> Path:
        return self.shared_dataset_dir / "alpha_dev_rows.jsonl"

    @property
    def heldout_test_rows(self) -> Path:
        return self.shared_dataset_dir / "heldout_test_rows.jsonl"

    @property
    def imdb_eval_prompts(self) -> Path:
        return self.shared_dataset_dir / "eval_prompts.jsonl"

    @property
    def selector_good_generations(self) -> Path:
        return self.shared_dataset_dir / "selector_good_generations.jsonl"

    @property
    def selector_good_generation_manifest(self) -> Path:
        return self.shared_dataset_dir / "selector_good_generations.manifest.json"

    @property
    def selector_base_generations(self) -> Path:
        return self.shared_dataset_dir / "selector_base_generations.jsonl"

    @property
    def selector_base_generation_manifest(self) -> Path:
        return self.shared_dataset_dir / "selector_base_generations.manifest.json"

    @property
    def selector_state_pair_manifest(self) -> Path:
        return self.shared_dataset_dir / "selector_good_base_state_pair.manifest.json"

    @property
    def pair_dir(self) -> Path:
        return self.shared_dataset_dir / "pairs"

    @property
    def pairs_csv(self) -> Path:
        return self.pair_dir / "pairs.csv"

    @property
    def job_dir(self) -> Path:
        return self.root / self.job.job_id

    @property
    def selector_dir(self) -> Path:
        return self.job_dir / "selector"

    @property
    def selected_heads(self) -> Path:
        return self.selector_dir / "selected_heads.csv"

    @property
    def shared_selector_dir(self) -> Path:
        return (
            self.root
            / "shared_selector"
            / self.job.dataset
            / self.job.subset
            / self.job.model_key
        )

    @property
    def shared_zero_scan_dir(self) -> Path:
        return self.shared_selector_dir / "rcm_zero"

    @property
    def actuator_dir(self) -> Path:
        return self.job_dir / "actuator"

    @property
    def evaluation_dir(self) -> Path:
        return self.job_dir / "evaluation"


class SiteProtocol:
    def __init__(self, config_path: Path, data: dict[str, Any]) -> None:
        self.config_path = config_path.resolve()
        self.data = data
        self.site = data["site"]
        self.config_sha256 = sha256_file(self.config_path)

    @classmethod
    def load(cls, config_path: Path, *, run_canonical_validator: bool = True) -> "SiteProtocol":
        path = config_path.resolve()
        data = json.loads(path.read_text(encoding="utf-8"))
        _require(data.get("protocol_id") == PROTOCOL_ID, "unexpected Site protocol id")
        _require(data.get("version") == PROTOCOL_VERSION, "unexpected Site protocol version")
        _require(
            data.get("status") == "locked_before_gpu_execution",
            "Site configuration is not in the registered locked state",
        )
        protocol = cls(path, data)
        protocol._validate_execution_contract()
        if run_canonical_validator:
            protocol.run_canonical_validator()
        return protocol

    @property
    def canonical_validator(self) -> Path:
        repo_root = self.config_path.parent.parent.resolve()
        configured = Path(str(self.data["preflight"]))
        _require(not configured.is_absolute(), "canonical validator path must be repository-relative")
        validator = (repo_root / configured).resolve()
        _require(validator.is_relative_to(repo_root), "canonical validator escapes the repository root")
        return validator

    def run_canonical_validator(self) -> None:
        validator = self.canonical_validator
        _require(validator.is_file(), f"missing canonical validator: {validator}")
        subprocess.run(
            [sys.executable, str(validator), "--config", str(self.config_path)],
            cwd=self.config_path.parent.parent,
            check=True,
        )

    def _validate_execution_contract(self) -> None:
        contract = self.site["selection_contract"]
        _require(contract["candidate_space"] == "pre_o_attention_head", "candidate-space drift")
        _require(contract["selected_count"] == 8, "K drift")
        _require(
            contract["signed_partition"]
            == {"target_support_count": 4, "competitor_support_count": 4},
            "signed partition drift",
        )
        _require(
            tuple(contract["direction_by_selector"]) == SELECTORS,
            "selector direction map drift",
        )
        _require(
            contract["ci_policy"] == "role_oriented_ci95_lower_bound_ranking",
            "CI policy drift",
        )
        _require(contract["nonpositive_fallback"] is False, "non-positive fallback is forbidden")
        _require(tuple(self.site["selectors"]) == SELECTORS, "selector set/order drift")
        common = self.site["downstream_common"]
        rcm_common = self.site["rcm_common"]
        _require(
            rcm_common["ranking_statistic"] == "role_oriented_ci95_lower_bound",
            "RCM ranking statistic drift",
        )
        _require(
            rcm_common["head_refinement_candidate_pool"]
            == "union_of_positive_and_negative_layer_beams",
            "head refinement pool drift",
        )
        _require(
            rcm_common["head_effect_role_source"]
            == "candidate_head_own_mean_paired_effect",
            "head role source drift",
        )
        _require(
            rcm_common["parent_layer_sign_inheritance"] is False,
            "parent layer sign inheritance is forbidden",
        )
        signed = self.site["downstream_contract"]["signed_rcm_bank_contract"]
        _require(signed["heads_per_bank"] == 4, "signed RCM bank size drift")
        _require(signed["inference_composition"] == "sum_both_role_banks", "signed RCM composition drift")
        _require(
            signed["alpha_policy"] == "one_shared_alpha_for_both_role_banks",
            "signed RCM alpha policy drift",
        )
        _require(signed["independent_role_alpha_search"] is False, "independent role alpha search is forbidden")
        expected_adamw = {
            "optimizer": "torch.optim.AdamW",
            "lr": 0.05,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "amsgrad": False,
        }
        for key, value in expected_adamw.items():
            _require(common[key] == value, f"downstream optimizer drift: {key}")
        imdb = self.site["tasks"]["imdb"]
        state_generation = imdb["selector_state_generation"]
        _require(state_generation["prompt_rows"] == 300, "IMDb selector state budget drift")
        _require(
            state_generation["conditions"] == ["good", "base"],
            "IMDb selector condition order drift",
        )
        _require(
            state_generation["common_random_numbers_across_conditions"] is True,
            "IMDb good/base state random-number alignment drift",
        )
        _require(
            imdb["selector_preprocessing"]["iti"]["pair_source"]
            == "same_shared_good_base_generation_artifact_as_rcm",
            "IMDb ITI state source drift",
        )
        confiqa = self.site["tasks"]["confiqa"]
        prompt_contract = confiqa["prompt_contract"]
        _require(prompt_contract["id"] == "context_dpo_official_v1", "ConFiQA prompt contract drift")
        _require(prompt_contract["prepared_prompt_key"] == "official_rag", "ConFiQA prompt key drift")
        _require(
            prompt_contract["pair_prompt_normalization"] == "preserve_exact",
            "ConFiQA pair prompt serialization must preserve exact bytes",
        )
        _require(
            prompt_contract["template"] == "{context}\nQ: {question}\nA: ",
            "ConFiQA official prompt template drift",
        )
        _require(
            prompt_contract["source_commit"] == confiqa["source_repository_commit"],
            "ConFiQA prompt source commit drift",
        )
        _require(prompt_contract["byte_exact_assertion"] is True, "ConFiQA byte-exact assertion is required")

    def iter_jobs(self) -> Iterator[SiteJob]:
        tasks = self.site["tasks"]
        selectors = self.site["selectors"]
        run_root = Path(self.site["execution"]["run_root"])
        for matrix_row in self.site["matrix"]:
            dataset = matrix_row["dataset"]
            task = tasks[dataset]
            for subset in task["subsets"]:
                for selector, selector_spec in selectors.items():
                    if selector == "random":
                        replicates = [
                            (f"seed_{seed}", int(seed))
                            for seed in selector_spec["replicate_seeds"]
                        ]
                    else:
                        seed = int(selector_spec.get("seed", selector_spec.get("random_state", 42)))
                        replicates = [("primary", seed)]
                    for replicate, seed in replicates:
                        job_id = "__".join(
                            [dataset, subset, matrix_row["model_key"], selector, replicate]
                        )
                        yield SiteJob(
                            job_id=job_id,
                            dataset=dataset,
                            subset=subset,
                            model_key=matrix_row["model_key"],
                            model=matrix_row["model"],
                            model_revision=matrix_row["revision"],
                            use_chat_template=bool(matrix_row["use_chat_template"]),
                            selector=selector,
                            replicate=replicate,
                            seed=seed,
                            output_dir=run_root / job_id,
                            selector_spec=selector_spec,
                            task=task,
                        )

    def job(self, job_id: str) -> SiteJob:
        matches = [job for job in self.iter_jobs() if job.job_id == job_id]
        _require(len(matches) == 1, f"unknown or duplicate Site job id: {job_id}")
        return matches[0]

    def paths(self, artifact_root: Path, job_id: str) -> SitePaths:
        root = artifact_root.resolve()
        locked_root = Path(self.site["execution"]["run_root"]).resolve()
        _require(root == locked_root, f"artifact root drift: locked={locked_root} requested={root}")
        return SitePaths(root=root, job=self.job(job_id))

    def selector_preprocessing(self, job: SiteJob) -> dict[str, Any]:
        key = "rcm" if job.selector.startswith("rcm_") else job.selector
        return job.task["selector_preprocessing"][key]

    def selector_state_generation(self, job: SiteJob) -> dict[str, Any] | None:
        return job.task.get("selector_state_generation")

    def snapshot_manifest(self, job: SiteJob) -> dict[str, Any]:
        return {
            "protocol_id": self.data["protocol_id"],
            "protocol_version": self.data["version"],
            "config": str(self.config_path),
            "config_sha256": self.config_sha256,
            "job_id": job.job_id,
            "dataset": job.dataset,
            "subset": job.subset,
            "model_key": job.model_key,
            "model": job.model,
            "model_revision": job.model_revision,
            "use_chat_template": job.use_chat_template,
            "selector": job.selector,
            "replicate": job.replicate,
            "seed": job.seed,
            "selection_contract": self.site["selection_contract"],
            "decision_trajectory": self.site["decision_trajectory"],
            "rcm_common": self.site["rcm_common"],
            "selector_spec": job.selector_spec,
            "selector_preprocessing": self.selector_preprocessing(job),
            "selector_state_generation": self.selector_state_generation(job),
            "task_states": job.task["states"],
            "task_score": job.task["score"],
            "downstream_contract": self.site["downstream_contract"],
            "downstream_common": self.site["downstream_common"],
            "downstream": job.task["downstream"],
        }


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    _require(not missing, "missing required artifacts: " + ", ".join(missing))
