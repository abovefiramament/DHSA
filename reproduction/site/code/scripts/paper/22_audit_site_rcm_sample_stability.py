#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path
from statistics import fmean
from typing import Any


def fail(message: str) -> None:
    raise ValueError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def candidate_mean(row: dict[str, Any], indices: tuple[int, ...]) -> float:
    deltas = row["deltas"]
    return fmean(float(deltas[index]) for index in indices)


def rank(
    rows: list[dict[str, Any]],
    indices: tuple[int, ...],
    *,
    sign: str,
    count: int,
) -> tuple[list[str], dict[str, float]]:
    scored = [(row, candidate_mean(row, indices)) for row in rows]
    if sign == "positive":
        eligible = [(row, value) for row, value in scored if value > 0]
        eligible.sort(
            key=lambda item: (
                -item[1],
                int(item[0]["layer_idx"]),
                -1 if item[0]["head_idx"] is None else int(item[0]["head_idx"]),
            )
        )
    elif sign == "negative":
        eligible = [(row, value) for row, value in scored if value < 0]
        eligible.sort(
            key=lambda item: (
                item[1],
                int(item[0]["layer_idx"]),
                -1 if item[0]["head_idx"] is None else int(item[0]["head_idx"]),
            )
        )
    else:
        fail(f"unknown sign: {sign}")
    selected = eligible[:count]
    return [row["component_id"] for row, _value in selected], {
        row["component_id"]: value for row, value in scored
    }


def compare_selection(
    selected: list[str],
    means: dict[str, float],
    reference: list[str],
    *,
    sign: str,
    count: int,
) -> dict[str, Any]:
    selected_set = set(selected)
    reference_set = set(reference)
    overlap = len(selected_set & reference_set)
    union = len(selected_set | reference_set)
    expected = (lambda value: value > 0) if sign == "positive" else (lambda value: value < 0)
    sign_consistent = sum(expected(means[component]) for component in reference)
    return {
        "selected": selected,
        "complete": len(selected) == count,
        "overlap_count": overlap,
        "reference_recall": overlap / count,
        "jaccard": overlap / union if union else 1.0,
        "set_exact": selected_set == reference_set,
        "ordered_exact": selected == reference,
        "reference_sign_consistency": sign_consistent / count,
    }


def summarize_replicates(
    records: list[dict[str, Any]],
    reference: list[str],
) -> dict[str, Any]:
    fields = [
        "overlap_count",
        "reference_recall",
        "jaccard",
        "reference_sign_consistency",
    ]
    summary: dict[str, Any] = {
        "replicates": len(records),
        "complete_rate": fmean(float(row["complete"]) for row in records),
        "set_exact_rate": fmean(float(row["set_exact"]) for row in records),
        "ordered_exact_rate": fmean(float(row["ordered_exact"]) for row in records),
    }
    for field in fields:
        values = [float(row[field]) for row in records]
        summary[field] = {
            "mean": fmean(values),
            "p025": percentile(values, 0.025),
            "p50": percentile(values, 0.5),
            "p975": percentile(values, 0.975),
            "min": min(values),
            "max": max(values),
        }
    frequencies: dict[str, int] = {}
    for row in records:
        for component in row["selected"]:
            frequencies[component] = frequencies.get(component, 0) + 1
    summary["reference_selection_frequency"] = {
        component: frequencies.get(component, 0) / len(records) for component in reference
    }
    summary["most_frequent_components"] = [
        {"component_id": component, "rate": count / len(records)}
        for component, count in sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))[:12]
    ]
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-hoc no-GPU sample-size stability audit for a saved RCM-zero scan."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scan-dir", type=Path, required=True)
    parser.add_argument("--selector-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sizes", default="30,60,90,120")
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    require(config.get("protocol_id") == "site_position_full_fixed_protocol_20260715_v5", "protocol drift")
    require(config.get("version") == 5, "protocol version drift")
    partition = config["site"]["selection_contract"]["signed_partition"]
    require(partition == {"target_support_count": 4, "competitor_support_count": 4}, "partition drift")
    count = 4

    artifact_paths = {
        "layer_results": args.scan_dir / "layer_results.jsonl",
        "positive_layer_beam": args.scan_dir / "positive_layer_beam.jsonl",
        "negative_layer_beam": args.scan_dir / "negative_layer_beam.jsonl",
        "positive_head_results": args.scan_dir / "positive_head_results.jsonl",
        "negative_head_results": args.scan_dir / "negative_head_results.jsonl",
        "scan_manifest": args.scan_dir / "scan_manifest.json",
        "selector_manifest": args.selector_manifest,
    }
    require(all(path.is_file() for path in artifact_paths.values()), "missing scan artifact")
    layer_rows = load_jsonl(artifact_paths["layer_results"])
    positive_head_rows = load_jsonl(artifact_paths["positive_head_results"])
    negative_head_rows = load_jsonl(artifact_paths["negative_head_results"])
    positive_beam = [row["component_id"] for row in load_jsonl(artifact_paths["positive_layer_beam"])]
    negative_beam = [row["component_id"] for row in load_jsonl(artifact_paths["negative_layer_beam"])]
    selector_manifest = json.loads(args.selector_manifest.read_text(encoding="utf-8"))
    scan_manifest = json.loads(artifact_paths["scan_manifest"].read_text(encoding="utf-8"))

    lengths = {
        len(row["deltas"])
        for rows in (layer_rows, positive_head_rows, negative_head_rows)
        for row in rows
    }
    require(lengths == {120}, f"expected one locked n=120 delta length, got {sorted(lengths)}")
    total_n = lengths.pop()
    sizes = tuple(int(value) for value in args.sizes.split(",") if value.strip())
    require(sizes and tuple(sorted(set(sizes))) == sizes, "sizes must be unique ascending integers")
    require(all(1 <= size <= total_n for size in sizes), "sample size outside saved delta range")
    require(args.replicates > 0, "replicates must be positive")

    full_indices = tuple(range(total_n))
    surfaces = [
        ("layer", "positive", layer_rows, positive_beam),
        ("layer", "negative", layer_rows, negative_beam),
        (
            "head_within_registered_full_beam",
            "positive",
            positive_head_rows,
            selector_manifest["selected_heads"][:count],
        ),
        (
            "head_within_registered_full_beam",
            "negative",
            negative_head_rows,
            selector_manifest["selected_heads"][count:],
        ),
    ]
    for stage, sign, rows, reference in surfaces:
        recomputed, _means = rank(rows, full_indices, sign=sign, count=count)
        require(recomputed == reference, f"full-data {stage}/{sign} ranking does not reproduce saved selection")

    subsets: dict[int, list[tuple[int, ...]]] = {}
    for size in sizes:
        if size == total_n:
            subsets[size] = [full_indices]
            continue
        rng = random.Random(args.seed + size * 1009)
        subsets[size] = [
            tuple(sorted(rng.sample(range(total_n), size))) for _ in range(args.replicates)
        ]

    repeated: list[dict[str, Any]] = []
    prefix: list[dict[str, Any]] = []
    for stage, sign, rows, reference in surfaces:
        for size in sizes:
            records = []
            for indices in subsets[size]:
                selected, means = rank(rows, indices, sign=sign, count=count)
                records.append(
                    compare_selection(selected, means, reference, sign=sign, count=count)
                )
            repeated.append(
                {
                    "stage": stage,
                    "sign": sign,
                    "sample_size": size,
                    "reference": reference,
                    **summarize_replicates(records, reference),
                }
            )
            prefix_indices = tuple(range(size))
            selected, means = rank(rows, prefix_indices, sign=sign, count=count)
            prefix.append(
                {
                    "stage": stage,
                    "sign": sign,
                    "sample_size": size,
                    "indices": [0, size],
                    "reference": reference,
                    **compare_selection(selected, means, reference, sign=sign, count=count),
                }
            )

    split_halves: list[dict[str, Any]] = []
    half_splits = {
        "contiguous": (tuple(range(60)), tuple(range(60, 120))),
        "interleaved": (tuple(range(0, 120, 2)), tuple(range(1, 120, 2))),
    }
    for split_name, (indices_a, indices_b) in half_splits.items():
        for stage, sign, rows, reference in surfaces:
            selected_a, means_a = rank(rows, indices_a, sign=sign, count=count)
            selected_b, means_b = rank(rows, indices_b, sign=sign, count=count)
            set_a, set_b = set(selected_a), set(selected_b)
            overlap = len(set_a & set_b)
            union = len(set_a | set_b)
            split_halves.append(
                {
                    "split": split_name,
                    "stage": stage,
                    "sign": sign,
                    "sample_size_per_half": 60,
                    "half_a": compare_selection(selected_a, means_a, reference, sign=sign, count=count),
                    "half_b": compare_selection(selected_b, means_b, reference, sign=sign, count=count),
                    "between_half_overlap_count": overlap,
                    "between_half_jaccard": overlap / union if union else 1.0,
                }
            )

    config_sha = sha256_file(args.config)
    require(
        scan_manifest["scan_contract"]["config_sha256"] == config_sha,
        "scan/config hash mismatch",
    )
    audit = {
        "audit_id": "site_v5_rcm_zero_sample_stability_qa_llama3_8b",
        "status": "complete_posthoc_no_gpu",
        "protocol_id": config["protocol_id"],
        "protocol_version": config["version"],
        "config_sha256": config_sha,
        "scope": {
            "dataset": selector_manifest["dataset"],
            "subset": selector_manifest["subset"],
            "model_key": selector_manifest["model_key"],
            "selector": selector_manifest["selector"],
            "saved_delta_n": total_n,
            "sizes": list(sizes),
            "subsamples_per_size_below_full_n": args.replicates,
            "subsample_seed": args.seed,
            "subsampling": "uniform_without_replacement_over_saved_paired_delta_indices",
        },
        "interpretation_boundary": [
            "post-hoc audit; it does not alter or gate the locked selector",
            "layer stability is reconstructed over all saved layers",
            "head stability is conditional on the registered full-data layer beams because other layers were not head-scanned",
            "RCM-patch is excluded because its saved aggregate CSV does not contain per-sample candidate deltas",
            "stability within n=120 cannot establish the counterfactual ranking at n=300",
        ],
        "provenance": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in artifact_paths.items()
        },
        "repeated_subsample": repeated,
        "fixed_prefix": prefix,
        "split_half": split_halves,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "audit.json"
    json_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = [
        "stage",
        "sign",
        "sample_size",
        "replicates",
        "complete_rate",
        "set_exact_rate",
        "ordered_exact_rate",
        "mean_overlap_count",
        "mean_jaccard",
        "mean_reference_sign_consistency",
    ]
    with (args.output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in repeated:
            writer.writerow(
                {
                    "stage": row["stage"],
                    "sign": row["sign"],
                    "sample_size": row["sample_size"],
                    "replicates": row["replicates"],
                    "complete_rate": row["complete_rate"],
                    "set_exact_rate": row["set_exact_rate"],
                    "ordered_exact_rate": row["ordered_exact_rate"],
                    "mean_overlap_count": row["overlap_count"]["mean"],
                    "mean_jaccard": row["jaccard"]["mean"],
                    "mean_reference_sign_consistency": row["reference_sign_consistency"]["mean"],
                }
            )
    print(f"wrote audit={json_path}")
    print(f"wrote summary={args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
