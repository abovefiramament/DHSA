#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from screscomp.cecm.scaling import load_open_eval_rows, normalize_answer  # noqa: E402
from screscomp.data import dump_csv, dump_json, dump_jsonl, load_jsonl  # noqa: E402


PROTOCOL_ID = "site_heldout_2048_fixed_protocol_20260726_v1"
PROTOCOL_VERSION = 1
TARGET_ROWS = 2048
SOURCE_SITE_PROTOCOL_ID = "site_position_full_fixed_protocol_20260719_v12"
SOURCE_TLDR_PROTOCOL_ID = "tldr_site_reft_direct_k8_disjoint_alpha_dev_20260725_v6"
TLDR_JUDGE_AMENDMENT_RELATIVE_PATH = "configs/site_heldout_2048_tldr_human_only_judge_amendment_20260727_v1.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl_ids(path: Path) -> set[str]:
    return {str(row["prompt_id"] if "prompt_id" in row else row["sample_id"]) for row in load_jsonl(path)}


class Heldout2048Protocol:
    def __init__(self, config_path: Path, data: dict[str, Any]) -> None:
        self.config_path = config_path.resolve()
        self.data = data
        self.config_sha256 = sha256_file(self.config_path)
        self.repo_root = self.config_path.parent.parent.resolve()
        self.run_root = Path(data["execution"]["run_root"]).resolve()
        self.confiqa_source_config_path = Path(data["confiqa"]["source_config"])
        self.tldr_source_config_path = Path(data["tldr"]["source_config"])
        self.confiqa_source = load_json(self.confiqa_source_config_path)
        self.tldr_source = load_json(self.tldr_source_config_path)

    @classmethod
    def load(cls, config_path: Path) -> "Heldout2048Protocol":
        protocol = cls(config_path, load_json(config_path))
        protocol.validate()
        return protocol

    def snapshot(self) -> dict[str, Any]:
        return {
            "protocol_id": PROTOCOL_ID,
            "version": PROTOCOL_VERSION,
            "config_path": str(self.config_path),
            "config_sha256": self.config_sha256,
        }

    def validate(self) -> None:
        data = self.data
        require(data.get("protocol_id") == PROTOCOL_ID, "unexpected held-out protocol id")
        require(data.get("version") == PROTOCOL_VERSION, "unexpected held-out protocol version")
        require(data.get("status") == "locked_before_gpu_execution", "held-out protocol is not locked")
        require(data.get("remote_source_of_truth") == str(REPO_ROOT), "remote source-of-truth drift")
        require(self.repo_root == REPO_ROOT, "config must be loaded from the remote source-of-truth repository")
        require(int(data.get("target_rows_per_stratum", -1)) == TARGET_ROWS, "held-out target n drift")
        require(
            data.get("documentation") == "docs/SITE_HELDOUT_2048_FIXED_PROTOCOL_20260726_V1.md",
            "documentation path drift",
        )
        require(
            data.get("preparer") == "scripts/paper/39_prepare_site_heldout_2048_protocol.py",
            "preparer path drift",
        )
        require(
            data.get("runner") == "scripts/paper/40_run_site_heldout_2048_serial.py",
            "runner path drift",
        )
        invariants = data["invariants"]
        require(invariants["evaluation_only"] is True, "extension must remain evaluation-only")
        for forbidden in (
            "component_reselection",
            "head_reselection",
            "actuator_retraining",
            "alpha_reselection",
            "prompt_change",
            "decoding_change",
            "metric_change",
            "judge_change",
            "method_specific_sample_drop_or_replacement",
        ):
            require(invariants[forbidden] is False, f"forbidden extension setting enabled: {forbidden}")
        require(invariants["failed_rows_retry_same_sample_only"] is True, "same-sample retry drift")
        require(
            invariants["all_methods_share_identical_sample_ids_within_stratum"] is True,
            "shared-sample contract drift",
        )
        execution = data["execution"]
        require(execution["serial_only"] is True and execution["single_gpu_per_job"] is True, "serial GPU contract drift")
        require(execution["preferred_physical_gpu_indices"] == [2, 3], "GPU candidate set drift")
        require(
            self.run_root == REPO_ROOT / "runs/site_heldout_2048_20260726_v1",
            "held-out run root drift",
        )

        confiqa = data["confiqa"]
        require(sha256_file(self.confiqa_source_config_path) == confiqa["source_config_sha256"], "Site V12 config hash drift")
        source_protocol = Path(confiqa["source_protocol"])
        require(source_protocol.is_file() and sha256_file(source_protocol) == confiqa["source_protocol_sha256"], "Site V12 protocol hash drift")
        require(self.confiqa_source.get("protocol_id") == SOURCE_SITE_PROTOCOL_ID, "Site V12 protocol id drift")
        require(self.confiqa_source.get("version") == 12, "Site V12 version drift")
        source_task = self.confiqa_source["site"]["tasks"]["confiqa"]
        require(confiqa["subsets"] == source_task["subsets"], "ConFiQA subset drift")
        source_models = [
            row["model_key"]
            for row in self.confiqa_source["site"]["matrix"]
            if row["dataset"] == "confiqa"
        ]
        require(confiqa["model_keys"] == source_models, "ConFiQA model matrix drift")
        sample = confiqa["sample_construction"]
        require(int(sample["source_start"]) == 1500, "ConFiQA replication start drift")
        require(int(sample["initial_window_rows"]) == TARGET_ROWS, "ConFiQA initial window drift")
        require(int(sample["target_valid_rows"]) == TARGET_ROWS, "ConFiQA target valid n drift")
        require(sample["fill_rejected_rows"] is True, "ConFiQA valid-row refill is required")
        require(sample["source_order"] == "official_file_order", "ConFiQA source order drift")
        require(sample["prompt_contract"] == source_task["prompt_contract"]["id"], "ConFiQA prompt contract drift")
        require(sample["schema"] == source_task["selector_preprocessing"]["rcm"]["schema"], "ConFiQA schema drift")
        require(sample["alias_policy"] == source_task["selector_preprocessing"]["rcm"]["alias_policy"], "ConFiQA alias policy drift")
        original_test = source_task["downstream"]["heldout_test"]
        require(
            sample["previous_v12_test_source_window"]
            == [int(original_test["source_start"]), int(original_test["source_stop"])],
            "ConFiQA original test provenance drift",
        )
        for subset, source in confiqa["source_files"].items():
            path = Path(source["path"])
            require(path.is_file() and sha256_file(path) == source["sha256"], f"ConFiQA source hash drift: {subset}")
            require(source["sha256"] == source_task["source_file_sha256"][subset], f"ConFiQA registered source mismatch: {subset}")

        tldr = data["tldr"]
        require(sha256_file(self.tldr_source_config_path) == tldr["source_config_sha256"], "TLDR V6 config hash drift")
        tldr_protocol = Path(tldr["source_protocol"])
        require(tldr_protocol.is_file() and sha256_file(tldr_protocol) == tldr["source_protocol_sha256"], "TLDR V6 protocol hash drift")
        require(self.tldr_source.get("protocol_id") == SOURCE_TLDR_PROTOCOL_ID, "TLDR V6 protocol id drift")
        require(self.tldr_source.get("version") == 6, "TLDR V6 version drift")
        reserve = tldr["reserve"]
        reserve_path = Path(reserve["path"])
        require(reserve_path.is_file() and sha256_file(reserve_path) == reserve["sha256"], "TLDR clean reserve hash drift")
        require(int(reserve["rows"]) == 5593, "TLDR reserve row-count provenance drift")
        require(int(reserve["slice_start"]) == 0 and int(reserve["slice_rows"]) == TARGET_ROWS, "TLDR held-out slice drift")
        require(int(self.tldr_source["inputs"]["final"]["rows"]) == 320, "TLDR V6 original final n drift")
        require(
            self.tldr_source["generation"] == load_json(Path(self.tldr_source["source_audit"]["config"]))["generation"],
            "TLDR V4/V6 generation settings drift",
        )
        require(
            self.tldr_source["evaluation"] == {
                **load_json(Path(self.tldr_source["source_audit"]["config"]))["evaluation"],
                "reported_selector_audit": self.tldr_source["evaluation"]["reported_selector_audit"],
            },
            "TLDR V4/V6 evaluator settings drift",
        )


def _write_raw_prefix(source: Path, destination: Path, *, start: int, rows: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    selected: list[bytes] = []
    with source.open("rb") as stream:
        for index, line in enumerate(stream):
            if index < start:
                continue
            if len(selected) == rows:
                break
            require(line.strip(), f"blank row in TLDR reserve at line {index + 1}")
            selected.append(line if line.endswith(b"\n") else line + b"\n")
    require(len(selected) == rows, f"TLDR reserve has only {len(selected)} rows in requested slice")
    with destination.open("wb") as stream:
        stream.writelines(selected)


def _prompt_contract_audit(rows_path: Path, source_task: dict[str, Any]) -> dict[str, Any]:
    contract = source_task["prompt_contract"]
    template = str(contract["template"])
    digest = hashlib.sha256()
    rows = load_jsonl(rows_path)
    require(len(rows) == TARGET_ROWS, f"ConFiQA prepared row-count drift: {rows_path}")
    ids: set[str] = set()
    for row in rows:
        sample_id = str(row["sample_id"])
        require(sample_id not in ids, f"duplicate ConFiQA sample id: {sample_id}")
        ids.add(sample_id)
        expected = template.format(context=row["context"], question=row["question"])
        actual = str(row["prompts"]["official_rag"])
        require(actual == expected, f"ConFiQA prompt contract drift: {sample_id}")
        digest.update(actual.encode("utf-8"))
        digest.update(b"\0")
    return {
        "rows": len(rows),
        "unique_sample_ids": len(ids),
        "prompt_contract": contract["id"],
        "prompt_bytes_sha256": digest.hexdigest(),
        "sample_ids": ids,
    }


def _generation_loader_audit(rows_path: Path) -> dict[str, Any]:
    raw_rows = load_jsonl(rows_path)
    loaded_rows = load_open_eval_rows(
        rows_path,
        prompt_key="official_rag",
        prior_source="dataset_orig",
        split=None,
        start=0,
        max_rows=None,
        val_mod=5,
    )
    raw_ids = [str(row["sample_id"]) for row in raw_rows]
    loaded_ids = [row.sample_id for row in loaded_rows]
    require(len(loaded_rows) == TARGET_ROWS, f"ConFiQA generation-loader n drift: {rows_path}")
    require(loaded_ids == raw_ids, f"ConFiQA generation-loader order/id drift: {rows_path}")
    return {
        "loader": "screscomp.cecm.scaling.load_open_eval_rows",
        "prompt_key": "official_rag",
        "prior_source": "dataset_orig",
        "split": None,
        "rows": len(loaded_rows),
        "sample_id_order_matches_input": True,
    }


def _prepare_confiqa_generation_rows(
    *,
    extension: dict[str, Any],
    sample: dict[str, Any],
    subset: str,
    output: Path,
    audit_dir: Path,
) -> None:
    requested_rows = TARGET_ROWS
    with tempfile.TemporaryDirectory(prefix=f"site_heldout_2048_{subset}_") as temp_root_raw:
        temp_root = Path(temp_root_raw)
        candidate_output = temp_root / "candidate_rows.jsonl"
        candidate_audit = temp_root / "admission_audit"
        while True:
            command = [
                sys.executable,
                "-m",
                "screscomp.cli.prepare_ckplug_open",
                "--dataset",
                "confiqa",
                "--data_json",
                extension["source_files"][subset]["path"],
                "--out_jsonl",
                str(candidate_output),
                "--start",
                str(sample["source_start"]),
                "--max_rows",
                str(sample["initial_window_rows"]),
                "--target_rows",
                str(requested_rows),
                "--audit_dir",
                str(candidate_audit),
                "--schema",
                sample["schema"],
                "--alias_policy",
                sample["alias_policy"],
                "--official_prompt_contract",
                sample["prompt_contract"],
            ]
            subprocess.run(command, cwd=REPO_ROOT, check=True)
            loaded_rows = load_open_eval_rows(
                candidate_output,
                prompt_key="official_rag",
                prior_source="dataset_orig",
                split=None,
                start=0,
                max_rows=None,
                val_mod=5,
            )
            if len(loaded_rows) >= TARGET_ROWS:
                break
            requested_rows += TARGET_ROWS - len(loaded_rows)

        candidate_rows = load_jsonl(candidate_output)
        candidate_by_id = {str(row["sample_id"]): row for row in candidate_rows}
        selected_ids = [row.sample_id for row in loaded_rows[:TARGET_ROWS]]
        require(len(set(selected_ids)) == TARGET_ROWS, f"duplicate ConFiQA generation IDs: {subset}")
        selected_rows = [candidate_by_id[sample_id] for sample_id in selected_ids]
        scan_stop = int(selected_rows[-1]["source_index"]) + 1

        source_audit = load_json(candidate_audit / "sample_admission_manifest.json")
        upstream_rejected = [
            row
            for row in load_jsonl(candidate_audit / "rejected_source_rows.jsonl")
            if int(row["source_index"]) < scan_stop
        ]
        selected_id_set = set(selected_ids)
        downstream_rejected: list[dict[str, Any]] = []
        for row in candidate_rows:
            source_index = int(row["source_index"])
            sample_id = str(row["sample_id"])
            if source_index >= scan_stop or sample_id in selected_id_set:
                continue
            context_norms = {normalize_answer(str(answer)) for answer in row["cf_answers"]}
            prior_norms = {normalize_answer(str(answer)) for answer in row["orig_answers"]}
            overlap = sorted(value for value in context_norms & prior_norms if value)
            require(overlap, f"unclassified ConFiQA generation-loader rejection: {sample_id}")
            downstream_rejected.append(
                {
                    "sample_id": sample_id,
                    "source_index": source_index,
                    "reason": "normalized_context_prior_alias_overlap",
                    "question": row["question"],
                    "orig_answer": row["orig_answer"],
                    "cf_answer": row["cf_answer"],
                    "normalized_overlap": overlap,
                }
            )

        rejected_records = sorted(
            [*upstream_rejected, *downstream_rejected],
            key=lambda row: int(row["source_index"]),
        )
        admitted_records = [
            {
                "sample_id": str(row["sample_id"]),
                "source_index": int(row["source_index"]),
                "admission_role": (
                    "initial_window"
                    if int(row["source_index"]) < int(source_audit["initial_stop_exclusive"])
                    else "replacement"
                ),
                "question": row["question"],
                "orig_answer": row["orig_answer"],
                "cf_answer": row["cf_answer"],
            }
            for row in selected_rows
        ]
        replacement_records = [
            row for row in admitted_records if row["admission_role"] == "replacement"
        ]

        output.parent.mkdir(parents=True, exist_ok=True)
        audit_dir.mkdir(parents=True, exist_ok=True)
        dump_jsonl(output, selected_rows)
        dump_csv(audit_dir / "admitted_source_rows.csv", admitted_records)
        dump_jsonl(audit_dir / "rejected_source_rows.jsonl", rejected_records)
        dump_jsonl(audit_dir / "replacement_source_rows.jsonl", replacement_records)
        dump_json(
            audit_dir / "sample_admission_manifest.json",
            {
                "dataset": "confiqa",
                "source": source_audit["source"],
                "source_sha256": source_audit["source_sha256"],
                "source_rows": int(source_audit["source_rows"]),
                "source_start": int(source_audit["source_start"]),
                "initial_stop_exclusive": int(source_audit["initial_stop_exclusive"]),
                "target_valid_rows": TARGET_ROWS,
                "scan_stop_exclusive": scan_stop,
                "inspected_rows": scan_stop - int(source_audit["source_start"]),
                "admitted_rows": TARGET_ROWS,
                "initial_window_admitted_rows": TARGET_ROWS - len(replacement_records),
                "replacement_rows": len(replacement_records),
                "rejected_rows": len(rejected_records),
                "upstream_admission_rejected_rows": len(upstream_rejected),
                "generation_loader_rejected_rows": len(downstream_rejected),
                "generation_loader": "screscomp.cecm.scaling.load_open_eval_rows",
                "admitted_source_indices": [row["source_index"] for row in admitted_records],
                "rejected_source_indices": [int(row["source_index"]) for row in rejected_records],
                "replacement_source_indices": [row["source_index"] for row in replacement_records],
                "output_jsonl": str(output.resolve()),
            },
        )


def prepare_confiqa(protocol: Heldout2048Protocol) -> list[dict[str, Any]]:
    extension = protocol.data["confiqa"]
    source_task = protocol.confiqa_source["site"]["tasks"]["confiqa"]
    sample = extension["sample_construction"]
    data_root = Path(sample["data_root"])
    source_run_root = Path(extension["source_run_root"])
    manifests: list[dict[str, Any]] = []
    for subset in extension["subsets"]:
        output_dir = data_root / subset
        output = output_dir / "heldout_test_2048.jsonl"
        audit_dir = output_dir / "admission_audit"
        _prepare_confiqa_generation_rows(
            extension=extension,
            sample=sample,
            subset=subset,
            output=output,
            audit_dir=audit_dir,
        )
        audit_path = audit_dir / "sample_admission_manifest.json"
        require(audit_path.is_file(), f"missing ConFiQA admission audit: {subset}")
        admission = load_json(audit_path)
        require(int(admission["admitted_rows"]) == TARGET_ROWS, f"ConFiQA admitted n drift: {subset}")
        require(int(admission["source_start"]) == int(sample["source_start"]), f"ConFiQA source start drift: {subset}")
        require(int(admission["target_valid_rows"]) == TARGET_ROWS, f"ConFiQA target n drift: {subset}")
        prompt_audit = _prompt_contract_audit(output, source_task)
        generation_loader_audit = _generation_loader_audit(output)
        new_ids = prompt_audit.pop("sample_ids")
        role_paths = {
            "v12_selector": source_run_root / "confiqa" / subset / "open_rows.jsonl",
            "v12_alpha_dev": source_run_root / "confiqa" / subset / "alpha_dev_rows.jsonl",
            "v12_heldout_test": source_run_root / "confiqa" / subset / "heldout_test_rows.jsonl",
        }
        intersections = {}
        for role, path in role_paths.items():
            require(path.is_file(), f"missing fixed ConFiQA source role: {path}")
            overlap = new_ids & jsonl_ids(path)
            require(not overlap, f"ConFiQA replication overlaps {role}: {subset} n={len(overlap)}")
            intersections[role] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "prompt_id_intersection": 0,
            }
        manifest = {
            **protocol.snapshot(),
            "dataset": "confiqa",
            "subset": subset,
            "source": extension["source_files"][subset],
            "output": str(output),
            "output_sha256": sha256_file(output),
            "actual_n": TARGET_ROWS,
            "source_start": int(sample["source_start"]),
            "initial_window_rows": int(sample["initial_window_rows"]),
            "scan_stop_exclusive": int(admission["scan_stop_exclusive"]),
            "rejected_rows": int(admission["rejected_rows"]),
            "replacement_rows": int(admission["replacement_rows"]),
            "admission_manifest": str(audit_path),
            "admission_manifest_sha256": sha256_file(audit_path),
            "prompt_audit": prompt_audit,
            "generation_loader_audit": generation_loader_audit,
            "disjointness": intersections,
            "status": "complete",
        }
        manifest_path = output_dir / "input_manifest.json"
        dump_json(manifest_path, manifest)
        manifests.append(manifest)
    return manifests


def _pair_role_ids(source_config: dict[str, Any]) -> dict[str, set[str]]:
    pairs_path = Path(source_config["inputs"]["pairs_csv"])
    train_target = int(source_config["actuator"]["training"]["max_train_rows"])
    val_target = int(source_config["actuator"]["training"]["max_val_rows"])
    selector_target = int(source_config["inputs"]["selector_groups"]["target_unique_groups"])
    train_rows = 0
    val_rows = 0
    selector_ids: set[str] = set()
    train_ids: set[str] = set()
    val_ids: set[str] = set()
    with pairs_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            split = str(row["split"])
            prompt_id = str(row["prompt_id"])
            if split == "train":
                if len(selector_ids) < selector_target:
                    selector_ids.add(prompt_id)
                if train_rows < train_target:
                    train_ids.add(prompt_id)
                    train_rows += 1
            elif split == "val" and val_rows < val_target:
                val_ids.add(prompt_id)
                val_rows += 1
            if train_rows == train_target and val_rows == val_target and len(selector_ids) == selector_target:
                break
    require(train_rows == train_target, f"TLDR train role has only {train_rows} rows")
    require(val_rows == val_target, f"TLDR validation role has only {val_rows} rows")
    require(len(selector_ids) == selector_target, f"TLDR selector role has only {len(selector_ids)} groups")
    return {
        "selector_train_300": selector_ids,
        "reft_train_8192_pairs": train_ids,
        "reft_validation_512_pairs": val_ids,
    }


def prepare_tldr(protocol: Heldout2048Protocol) -> dict[str, Any]:
    extension = protocol.data["tldr"]
    reserve = extension["reserve"]
    source = Path(reserve["path"])
    output = Path(extension["output_file"])
    if not output.is_file():
        _write_raw_prefix(
            source,
            output,
            start=int(reserve["slice_start"]),
            rows=int(reserve["slice_rows"]),
        )
    rows = load_jsonl(output)
    require(len(rows) == TARGET_ROWS, "TLDR expanded final row-count drift")
    new_ids = {str(row["prompt_id"]) for row in rows}
    require(len(new_ids) == TARGET_ROWS, "TLDR expanded final prompt IDs are not unique")
    require(all(str(row.get("split")) == "test" for row in rows), "TLDR expanded final contains non-test rows")
    roles = _pair_role_ids(protocol.tldr_source)
    inputs = protocol.tldr_source["inputs"]
    roles.update(
        {
            "head_audit_dev_40": jsonl_ids(Path(inputs["head_audit_dev"]["path"])),
            "alpha_dev_40": jsonl_ids(Path(inputs["calibration"]["path"])),
            "calibration_reserve_240": jsonl_ids(Path(inputs["calibration_reserve"]["path"])),
            "v6_final_test_320": jsonl_ids(Path(inputs["final"]["path"])),
        }
    )
    require(list(roles) == extension["required_disjoint_roles"], "TLDR disjoint-role order/content drift")
    intersections = []
    for role, ids in roles.items():
        overlap = new_ids & ids
        require(not overlap, f"TLDR expanded final overlaps {role}: n={len(overlap)}")
        intersections.append(
            {
                "role": role,
                "role_unique_prompt_ids": len(ids),
                "prompt_id_intersection": 0,
            }
        )
    manifest = {
        **protocol.snapshot(),
        "dataset": "tldr",
        "reserve": reserve,
        "output": str(output),
        "output_sha256": sha256_file(output),
        "actual_n": TARGET_ROWS,
        "unique_prompt_ids": TARGET_ROWS,
        "all_required_roles_disjoint": True,
        "intersections": intersections,
        "status": "complete",
    }
    manifest_path = output.parent / "input_manifest.json"
    if manifest_path.is_file():
        require(load_json(manifest_path) == manifest, "TLDR expanded input manifest drift")
    else:
        dump_json(manifest_path, manifest)
    return manifest


def _confiqa_jobs(protocol: Heldout2048Protocol) -> Iterable[dict[str, Any]]:
    extension = protocol.data["confiqa"]
    source = protocol.confiqa_source
    selectors = source["site"]["selectors"]
    source_root = Path(extension["source_run_root"])
    for matrix_row in source["site"]["matrix"]:
        if matrix_row["dataset"] != "confiqa":
            continue
        require(matrix_row["model_key"] in extension["model_keys"], "unexpected ConFiQA model key")
        for subset in extension["subsets"]:
            for selector, spec in selectors.items():
                if selector == "random":
                    replicates = [(f"seed_{seed}", int(seed)) for seed in spec["replicate_seeds"]]
                else:
                    replicates = [("primary", int(spec.get("seed", spec.get("random_state", 42))))]
                for replicate, seed in replicates:
                    job_id = "__".join(
                        ["confiqa", subset, matrix_row["model_key"], selector, replicate]
                    )
                    job_root = source_root / job_id
                    requirements = [job_root / rel for rel in extension["source_artifact_requirements"]]
                    yield {
                        "job_id": job_id,
                        "dataset": "confiqa",
                        "subset": subset,
                        "model_key": matrix_row["model_key"],
                        "selector": selector,
                        "replicate": replicate,
                        "seed": seed,
                        "source_ready": int(all(path.is_file() for path in requirements)),
                        "source_job_root": str(job_root),
                    }


def write_protocol_records(
    protocol: Heldout2048Protocol,
    confiqa_manifests: list[dict[str, Any]],
    tldr_manifest: dict[str, Any],
) -> None:
    protocol_dir = protocol.run_root / "protocol"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    fixed_files = {
        "config": protocol.config_path,
        "documentation": REPO_ROOT / protocol.data["documentation"],
        "preparer": REPO_ROOT / protocol.data["preparer"],
        "runner": REPO_ROOT / protocol.data["runner"],
        "source_site_config": protocol.confiqa_source_config_path,
        "source_site_protocol": Path(protocol.data["confiqa"]["source_protocol"]),
        "source_tldr_config": protocol.tldr_source_config_path,
        "source_tldr_protocol": Path(protocol.data["tldr"]["source_protocol"]),
        "tldr_judge_amendment": REPO_ROOT / TLDR_JUDGE_AMENDMENT_RELATIVE_PATH,
    }
    require(all(path.is_file() for path in fixed_files.values()), "fixed protocol implementation is incomplete")
    fixed_manifest = {
        **protocol.snapshot(),
        "files": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in fixed_files.items()
        },
        "status": "complete",
    }
    dump_json(protocol_dir / "fixed_protocol_manifest.json", fixed_manifest)
    shutil.copy2(protocol.config_path, protocol_dir / "scientific_config.json")
    require(sha256_file(protocol_dir / "scientific_config.json") == protocol.config_sha256, "run-local config copy drift")
    jobs = list(_confiqa_jobs(protocol))
    dump_csv(protocol_dir / "confiqa_jobs.csv", jobs)
    dump_json(
        protocol_dir / "input_manifest.json",
        {
            **protocol.snapshot(),
            "target_rows_per_stratum": TARGET_ROWS,
            "confiqa": confiqa_manifests,
            "tldr": tldr_manifest,
            "confiqa_jobs_total": len(jobs),
            "confiqa_jobs_source_ready_at_preflight": sum(int(row["source_ready"]) for row in jobs),
            "status": "complete",
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare fixed 2048-row Site held-out inputs.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = Heldout2048Protocol.load(args.config)
    confiqa = prepare_confiqa(protocol)
    tldr = prepare_tldr(protocol)
    write_protocol_records(protocol, confiqa, tldr)
    print(
        f"[site-heldout-2048-prepare] complete confiqa_subsets={len(confiqa)} "
        f"tldr_n={tldr['actual_n']} config_sha256={protocol.config_sha256}",
        flush=True,
    )


if __name__ == "__main__":
    main()
