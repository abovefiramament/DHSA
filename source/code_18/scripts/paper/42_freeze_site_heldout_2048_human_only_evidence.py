#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_ID = "site_heldout_2048_human_only_evidence_freeze_20260727_v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_order(values: Iterable[str]) -> str:
    payload = "".join(f"{value}\n" for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            require(isinstance(row, dict), f"non-object JSONL row: {path}:{line_number}")
            rows.append(row)
    return rows


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def dump_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def resolve_repo_path(value: str) -> Path:
    path = (REPO_ROOT / value).resolve()
    require(path == REPO_ROOT or REPO_ROOT in path.parents, f"path leaves repo: {value}")
    return path


def source_path(config: dict[str, Any], name: str) -> Path:
    return resolve_repo_path(config["sources"][name]["source"])


def sample_ids(rows: list[dict[str, Any]], path: Path) -> list[str]:
    values: list[str] = []
    for index, row in enumerate(rows):
        require("sample_id" in row, f"missing sample_id: {path}:{index + 1}")
        values.append(str(row["sample_id"]))
    return values


def validate_exact_ids(
    path: Path,
    expected_ids: list[str],
    expected_n: int,
) -> dict[str, Any]:
    rows = load_jsonl(path)
    ids = sample_ids(rows, path)
    require(len(rows) == expected_n, f"unexpected row count: {path}: {len(rows)}")
    require(len(set(ids)) == expected_n, f"duplicate sample IDs: {path}")
    require(ids == expected_ids, f"sample order mismatch: {path}")
    return {
        "path": str(path),
        "rows": len(rows),
        "sha256": sha256_file(path),
        "sample_order_sha256": sha256_order(ids),
    }


def validate_confiqa(
    config: dict[str, Any],
    expected_n: int,
) -> dict[str, Any]:
    split_root = source_path(config, "confiqa_splits")
    run_root = source_path(config, "run") / "confiqa"
    spec = config["validation"]["confiqa"]
    model_key = spec["model_key"]
    subset_reports: dict[str, Any] = {}

    for subset, expected in spec["subsets"].items():
        input_path = split_root / subset / "heldout_test_2048.jsonl"
        manifest_path = split_root / subset / "input_manifest.json"
        require(input_path.is_file(), f"missing ConFiQA split: {input_path}")
        require(manifest_path.is_file(), f"missing ConFiQA split manifest: {manifest_path}")
        rows = load_jsonl(input_path)
        ids = sample_ids(rows, input_path)
        require(len(rows) == expected_n, f"{subset}: expected {expected_n} split rows")
        require(len(set(ids)) == expected_n, f"{subset}: duplicate split sample IDs")
        require(sha256_file(input_path) == expected["output_sha256"], f"{subset}: split hash drift")

        manifest = load_json(manifest_path)
        for key, value in (
            ("status", "complete"),
            ("actual_n", expected_n),
            ("source_start", spec["source_start"]),
            ("scan_stop_exclusive", expected["scan_stop_exclusive"]),
            ("rejected_rows", expected["rejected_rows"]),
            ("replacement_rows", expected["replacement_rows"]),
            ("output_sha256", expected["output_sha256"]),
        ):
            require(manifest.get(key) == value, f"{subset}: input manifest drift for {key}")
        loader_audit = manifest["generation_loader_audit"]
        require(loader_audit["rows"] == expected_n, f"{subset}: loader row-count drift")
        require(loader_audit["sample_id_order_matches_input"] is True, f"{subset}: loader order drift")
        require(
            all(item["prompt_id_intersection"] == 0 for item in manifest["disjointness"].values()),
            f"{subset}: split intersects a version-12 role",
        )

        audit_root = split_root / subset / "admission_audit"
        required_audit = [
            "admitted_source_rows.csv",
            "sample_admission_manifest.json",
            "rejected_source_rows.jsonl",
            "replacement_source_rows.jsonl",
        ]
        require(all((audit_root / name).is_file() for name in required_audit), f"{subset}: incomplete audit")

        generation_root = run_root / subset / model_key
        generation_paths = sorted(generation_root.glob("*/generation_rows.jsonl"))
        require(
            len(generation_paths) == spec["generation_files_per_subset"],
            f"{subset}: expected {spec['generation_files_per_subset']} generation files, got {len(generation_paths)}",
        )
        generations = [
            validate_exact_ids(path, ids, expected_n)
            for path in generation_paths
        ]

        subset_reports[subset] = {
            "input": {
                "path": str(input_path),
                "rows": len(rows),
                "sha256": sha256_file(input_path),
                "sample_order_sha256": sha256_order(ids),
                "source_start": manifest["source_start"],
                "scan_stop_exclusive": manifest["scan_stop_exclusive"],
                "rejected_rows": manifest["rejected_rows"],
                "replacement_rows": manifest["replacement_rows"],
                "prompt_bytes_sha256": manifest["prompt_audit"]["prompt_bytes_sha256"],
            },
            "input_manifest": {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
            },
            "generation_files": generations,
        }

    results_path = run_root / "confiqa_heldout_2048_results.csv"
    results = load_csv(results_path)
    expected_rows = len(spec["subsets"]) * spec["reported_rows_per_subset"]
    require(len(results) == expected_rows, f"ConFiQA result row count: {len(results)}")
    for subset in spec["subsets"]:
        subset_rows = [row for row in results if row["subset"] == subset]
        require(
            len(subset_rows) == spec["reported_rows_per_subset"],
            f"{subset}: reported row-count mismatch",
        )
        require(all(int(row["actual_n"]) == expected_n for row in subset_rows), f"{subset}: actual_n drift")
        require(all(row["model_key"] == model_key for row in subset_rows), f"{subset}: model drift")
        require(
            set(row["selector"] for row in subset_rows) == set(spec["selectors"]),
            f"{subset}: selector set drift",
        )

    return {
        "status": "complete_llama3_only",
        "subsets": subset_reports,
        "results": {
            "path": str(results_path),
            "sha256": sha256_file(results_path),
            "rows": len(results),
            "records": results,
        },
        "incomplete_matrix_cells": [
            "confiqa/qa/mistral7b_v03",
            "confiqa/mr/mistral7b_v03",
            "confiqa/mc/mistral7b_v03",
            "confiqa/qa/qwen25_14b",
            "confiqa/mr/qwen25_14b",
            "confiqa/mc/qwen25_14b",
        ],
    }


def comparison_counts(rows: list[dict[str, Any]], expected_n: int, label: str) -> dict[str, int]:
    require(all(row.get("status") == "ok" for row in rows), f"{label}: non-ok judge rows")
    comparisons = Counter(str(row.get("comparison", "")) for row in rows)
    require(all(name.endswith("_vs_human") for name in comparisons), f"{label}: non-human comparison")
    require(all(count == expected_n for count in comparisons.values()), f"{label}: comparison count drift")
    return dict(sorted(comparisons.items()))


def validate_tldr(
    config: dict[str, Any],
    expected_n: int,
) -> dict[str, Any]:
    split_root = source_path(config, "tldr_split")
    run_root = source_path(config, "run") / "tldr"
    spec = config["validation"]["tldr"]
    input_path = split_root / "final_test_2048.jsonl"
    input_manifest_path = split_root / "input_manifest.json"
    rows = load_jsonl(input_path)
    ids = sample_ids(rows, input_path)
    require(len(rows) == expected_n, f"TLDR split row count: {len(rows)}")
    require(len(set(ids)) == expected_n, "TLDR split contains duplicate sample IDs")
    require(sha256_file(input_path) == spec["input_sha256"], "TLDR split hash drift")
    input_manifest = load_json(input_manifest_path)
    require(input_manifest["status"] == "complete", "TLDR input manifest is incomplete")
    require(input_manifest["actual_n"] == expected_n, "TLDR input manifest row-count drift")
    require(input_manifest["all_required_roles_disjoint"] is True, "TLDR split role overlap")
    require(
        all(item["prompt_id_intersection"] == 0 for item in input_manifest["intersections"]),
        "TLDR split intersects an earlier role",
    )

    generation_paths = sorted((run_root / "generations").glob("*.jsonl"))
    require(len(generation_paths) == spec["generation_files"], "TLDR generation file-count drift")
    generations = [validate_exact_ids(path, ids, expected_n) for path in generation_paths]
    require(
        set(path.stem for path in generation_paths) == set(spec["candidate_names"]),
        "TLDR candidate generation set drift",
    )

    judge_root = run_root / "judge_human_only"
    first_path = judge_root / "judge_first_pass.jsonl"
    review_path = judge_root / "judge_order_swap_review.jsonl"
    packet_path = judge_root / "judge_packets.jsonl"
    review_packet_path = judge_root / "judge_order_swap_packets.jsonl"
    first_rows = load_jsonl(first_path)
    review_rows = load_jsonl(review_path)
    packets = load_jsonl(packet_path)
    review_packets = load_jsonl(review_packet_path)
    require(len(first_rows) == spec["first_pass_rows"], "TLDR first-pass count drift")
    require(len(review_rows) == spec["order_swap_rows"], "TLDR order-swap count drift")
    require(len(packets) == spec["first_pass_rows"], "TLDR first-pass packet count drift")
    require(len(review_packets) == spec["order_swap_rows"], "TLDR order-swap packet count drift")
    first_counts = comparison_counts(first_rows, expected_n, "first pass")
    review_counts = comparison_counts(review_rows, expected_n, "order swap")
    require(set(first_counts) == set(review_counts), "TLDR first/review comparison set mismatch")

    summary_path = judge_root / "pairwise_summary.json"
    summary = load_json(summary_path)
    require(summary["first_pass_ok"] == spec["first_pass_rows"], "TLDR summary first-pass drift")
    require(summary["review_ok"] == spec["order_swap_rows"], "TLDR summary review drift")
    require(len(summary["pairwise"]) == spec["pairwise_rows"], "TLDR pairwise row-count drift")
    require(all(row["right"] == "human" for row in summary["pairwise"]), "TLDR summary has non-human pair")
    require(all(int(row["n"]) == expected_n for row in summary["pairwise"]), "TLDR pairwise n drift")
    require(
        set(row["left"] for row in summary["pairwise"]) == set(spec["candidate_names"]),
        "TLDR summary candidate set drift",
    )

    table_path = run_root / "site_selector_final_table_2048.csv"
    table = load_csv(table_path)
    require(len(table) == spec["pairwise_rows"], "TLDR final-table row-count drift")
    require(all(int(row["actual_n"]) == expected_n for row in table), "TLDR final-table n drift")
    require(set(row["name"] for row in table) == set(spec["candidate_names"]), "TLDR table candidate drift")

    superseded_root = run_root / "judge"
    superseded_first = superseded_root / "judge_first_pass.jsonl"
    superseded_rows = load_jsonl(superseded_first)
    require(
        len(superseded_rows) == spec["superseded_partial_first_pass_rows"],
        "superseded TLDR judge row-count drift",
    )
    require(not (superseded_root / "pairwise_summary.csv").exists(), "superseded judge gained a CSV summary")
    require(not (superseded_root / "pairwise_summary.json").exists(), "superseded judge gained a JSON summary")

    return {
        "status": "complete_human_reference_only",
        "input": {
            "path": str(input_path),
            "rows": len(rows),
            "sha256": sha256_file(input_path),
            "sample_order_sha256": sha256_order(ids),
        },
        "input_manifest": {
            "path": str(input_manifest_path),
            "sha256": sha256_file(input_manifest_path),
            "all_required_roles_disjoint": input_manifest["all_required_roles_disjoint"],
            "intersections": input_manifest["intersections"],
        },
        "generation_files": generations,
        "judge": {
            "formal_path": str(judge_root),
            "first_pass_rows": len(first_rows),
            "order_swap_rows": len(review_rows),
            "first_pass_comparisons": first_counts,
            "order_swap_comparisons": review_counts,
            "pairwise_summary_sha256": sha256_file(summary_path),
            "table_sha256": sha256_file(table_path),
            "table_records": table,
        },
        "superseded_judge": {
            "path": str(superseded_root),
            "first_pass_rows": len(superseded_rows),
            "has_summary": False,
            "evidence_status": "forbidden_failure_provenance_only",
        },
    }


def validate_upstream_references(config: dict[str, Any]) -> dict[str, Any]:
    site_spec = config["site_v12_reference"]
    site_root = resolve_repo_path(site_spec["root"])
    site_manifest_path = resolve_repo_path(site_spec["manifest"])
    require(site_root.is_dir(), f"missing Site v12 snapshot: {site_root}")
    require(sha256_file(site_manifest_path) == site_spec["manifest_sha256"], "Site v12 manifest drift")
    site_manifest = load_json(site_manifest_path)
    for key in ("status", "protocol_id", "protocol_version", "completed_jobs", "expected_jobs_in_scope"):
        require(site_manifest[key] == site_spec[key], f"Site v12 reference drift: {key}")

    tldr_root = source_path(config, "tldr_v6_curated")
    for name, expected_hash in config["tldr_v6_reference_seals"].items():
        path = tldr_root / name
        require(path.is_file(), f"missing TLDR v6 reference: {path}")
        require(sha256_file(path) == expected_hash, f"TLDR v6 reference drift: {name}")

    return {
        "site_v12": {
            "path": str(site_root),
            "manifest": str(site_manifest_path),
            "manifest_sha256": sha256_file(site_manifest_path),
            "completed_jobs": site_manifest["completed_jobs"],
            "expected_jobs_in_scope": site_manifest["expected_jobs_in_scope"],
        },
        "tldr_v6": {
            "path": str(tldr_root),
            "sealed_files": config["tldr_v6_reference_seals"],
        },
    }


def copy_sources(config: dict[str, Any], staging: Path) -> None:
    for item in config["sources"].values():
        source = resolve_repo_path(item["source"])
        destination = staging / item["destination"]
        require(source.is_dir(), f"missing source tree: {source}")
        require(not destination.exists(), f"duplicate destination: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination, copy_function=shutil.copy2)

    site_spec = config["site_v12_reference"]
    site_root = resolve_repo_path(site_spec["root"])
    for relative in site_spec["copy_files"]:
        source = site_root / relative
        destination = staging / "references/site_v12" / relative
        require(source.is_file(), f"missing Site v12 reference file: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    for relative in config["canonical_files"]:
        source = resolve_repo_path(relative)
        destination = staging / "canonical" / relative
        require(source.is_file(), f"missing canonical file: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    documentation = resolve_repo_path(config["documentation"])
    shutil.copy2(documentation, staging / "README.md")


def inventory(staging: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in staging.rglob("*") if item.is_file()):
        relative = path.relative_to(staging).as_posix()
        rows.append(
            {
                "relative_path": relative,
                "category": relative.split("/", 1)[0],
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze completed Site held-out 2048 evidence without experiment mutation."
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    require(config_path.is_file(), f"missing config: {config_path}")
    config = load_json(config_path)
    require(config["protocol_id"] == PROTOCOL_ID, "unexpected freeze protocol")
    require(config["version"] == 1, "unexpected freeze version")
    require(config["status"] == "locked_evidence_only_no_experiment_mutation", "freeze not locked")
    require(Path(config["remote_source_of_truth"]).resolve() == REPO_ROOT, "not remote source of truth")

    for record_name in ("parent_protocol", "judge_amendment"):
        record = config[record_name]
        path = resolve_repo_path(record["path"])
        require(path.is_file(), f"missing {record_name}: {path}")
        require(sha256_file(path) == record["sha256"], f"{record_name} hash drift")

    expected_n = int(config["validation"]["expected_n"])
    confiqa_report = validate_confiqa(config, expected_n)
    tldr_report = validate_tldr(config, expected_n)
    upstream_report = validate_upstream_references(config)

    output_root = resolve_repo_path(config["output_root"])
    staging = output_root.parent / f".{output_root.name}.staging"
    require(not output_root.exists(), f"immutable freeze already exists: {output_root}")
    require(not staging.exists(), f"stale staging directory exists: {staging}")
    staging.mkdir(parents=True)
    try:
        copy_sources(config, staging)
        shutil.copy2(config_path, staging / "scientific_config.json")
        inventory_rows = inventory(staging)
        inventory_path = staging / "file_inventory.csv"
        dump_csv(inventory_path, inventory_rows)

        manifest = {
            "protocol_id": PROTOCOL_ID,
            "version": 1,
            "freeze_date": config["freeze_date"],
            "status": "complete_immutable_evidence_snapshot",
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "runner_path": str(Path(__file__).resolve()),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "documentation_path": str(resolve_repo_path(config["documentation"])),
            "documentation_sha256": sha256_file(resolve_repo_path(config["documentation"])),
            "no_experiment_mutation": config["evidence_policy"],
            "new_data_splits": {
                "confiqa": confiqa_report["subsets"],
                "tldr": {
                    "input": tldr_report["input"],
                    "input_manifest": tldr_report["input_manifest"],
                },
            },
            "evidence": {
                "confiqa": confiqa_report,
                "tldr": tldr_report,
            },
            "upstream_references": upstream_report,
            "claim_boundaries": config["claim_boundaries"],
            "inventory": {
                "path": "file_inventory.csv",
                "sha256": sha256_file(inventory_path),
                "files_excluding_inventory_and_manifest": len(inventory_rows),
                "bytes_excluding_inventory_and_manifest": sum(row["bytes"] for row in inventory_rows),
            },
        }
        dump_json(staging / "evidence_manifest.json", manifest)
        os.replace(staging, output_root)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    frozen_inventory = load_csv(output_root / "file_inventory.csv")
    for row in frozen_inventory:
        path = output_root / row["relative_path"]
        require(path.is_file(), f"frozen file missing: {path}")
        require(sha256_file(path) == row["sha256"], f"frozen file hash drift: {path}")
    print(
        json.dumps(
            {
                "status": "complete_immutable_evidence_snapshot",
                "output_root": str(output_root),
                "inventory_files": len(frozen_inventory),
                "manifest_sha256": sha256_file(output_root / "evidence_manifest.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
