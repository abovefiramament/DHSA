"""Recover RCM score operands through the registered shared measurement backend.

This operation writes a separate evidence supplement and never mutates a source
cell, selects positions, trains a controller, or replaces frozen measurements.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from baseline.implementations.loaders import load_model, load_rows
from baseline.implementations.model_runtime import (
    HuggingFaceModelProvider, register_model_paths_from_runtime_config,
    rcm_measure_many,
)
from evaluators.imdb import IMDbSentimentEvaluator


def _save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def _component_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def complete_rcm_evidence(*, source_root: Path, output_root: Path,
                          device: str = "cuda", replay_interventions: bool = False,
                          attention_implementation: str | None = None,
                          score_reduction_dtype: str | None = None) -> dict[str, Any]:
    from experiments.shared.numerical_environment import capture_numerical_environment
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if output_root == source_root or source_root in output_root.parents:
        raise ValueError("RCM supplement must be outside the immutable source cell")
    config = json.loads((source_root / "config/runtime_config.local.json").read_text())
    run = json.loads((source_root / "run_manifest.json").read_text())
    rows = load_rows(source_root / "data/roles/selector.manifest.json", root=source_root)
    by_sample = {str(row["sample_id"]): row for row in rows}
    if len(by_sample) != len(rows):
        raise ValueError("Duplicate source selector rows")
    source_scores = load_rows(source_root / "position/scan/component_scores.jsonl")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_scores:
        groups[str(row["component_id"])].append(row)
    for group in groups.values():
        if [str(row["sample_id"]) for row in group] != list(by_sample):
            raise ValueError("Recorded component row order differs from selector order")
    control = config["position_control"]
    if control["method"] != "rcm_zero":
        raise ValueError("This evidence completion is registered for RCM-zero")
    scan_config = dict(control["scan_config"])
    if score_reduction_dtype is not None:
        if score_reduction_dtype not in ("float32", "model_output"):
            raise ValueError("Unsupported RCM score reduction dtype")
        scan_config["score_reduction_dtype"] = score_reduction_dtype
    dataset = config["identity"]["dataset"]
    output_root.mkdir(parents=True, exist_ok=True)
    _save(output_root / "numerical_environment.json", capture_numerical_environment())
    _save(output_root / "operation.json", {
        "operation": "rcm_score_operand_completion",
        "source_identity": run["identity"], "source_status": run["status"],
        "selector_rows": len(rows), "components": len(groups),
        "settings": "source compiled model, scorer, input order and measurement settings",
        "outputs_replace_source": False,
        "gpu_execution": device,
        "numerical_conditions": {"attention_implementation": attention_implementation,
                                 "score_reduction_dtype": scan_config.get("score_reduction_dtype", "float32")},
        "mode": ("replay_interventions_on_fixed_selector" if replay_interventions else
                 "score_saved_generation_text" if dataset == "imdb" else "replay_fixed_endpoint_measurement"),
        "native_selector_reconstructed": False,
        "component_set": "source_recorded_components; does_not_validate_fresh_parent_selection",
    })
    if dataset == "imdb" and not replay_interventions:
        scorer = scan_config["sentiment_scorer"]
        evaluator = IMDbSentimentEvaluator(device=device)
        def score(texts):
            return evaluator.score_completions(
                texts, path=scorer["local_path"], revision=scorer["revision"],
                batch_size=scorer["reward_batch_size"])
        native_texts = [row["good_state"]["generated_text"] for row in rows]
        native_values = score(native_texts)
        trajectories = load_rows(source_root / "position/scan/trajectory.jsonl")
        texts = {(str(row["sample_id"]), str(row["component_id"])): row["generated_text"]
                 for row in trajectories}
        if len(texts) != len(source_scores):
            raise ValueError("Saved intervention texts do not cover source scores exactly")
        _save(output_root / "native_scores.json", [
            {"sample_id": row["sample_id"], "native_margin": value}
            for row, value in zip(rows, native_values, strict=True)])
    else:
        provider = HuggingFaceModelProvider(device=device, attention_implementation=attention_implementation)
        register_model_paths_from_runtime_config(provider, config)
        model_config = {"model": control["method_config"]["baseline"]["parameters"]["model"]}
        adapter = load_model(provider, model_config, mode="inference")
        _save(output_root / "numerical_environment.json", capture_numerical_environment(adapter.model))

    for index, (component, original) in enumerate(groups.items(), 1):
        checkpoint = output_root / "components" / (component + ".jsonl")
        if checkpoint.exists():
            saved = load_rows(checkpoint)
            if ([str(x["sample_id"]) for x in saved] != list(by_sample)
                    or any(x["component_id"] != component for x in saved)):
                raise ValueError("Evidence checkpoint does not match source component")
            continue
        if dataset == "imdb" and not replay_interventions:
            changed_texts = [texts[(str(row["sample_id"]), component)] for row in rows]
            changed_values = score(changed_texts)
            stripped = [text.strip() for text in changed_texts]
            normalized_values = score(stripped) if stripped != changed_texts else changed_values
            measured = [
                {"native_margin": native, "intervened_margin": zero,
                 "effect": native-zero, "normalized_intervened_margin": normalized,
                 "normalized_effect": native-normalized}
                for native, zero, normalized in zip(
                    native_values, changed_values, normalized_values, strict=True)]
        else:
            layer = int(component[1:].split(".attn", 1)[0])
            head = int(component.split(".h", 1)[1]) if ".h" in component else None
            measured = list(rcm_measure_many(
                adapter, rows, {"component_id": component, "layer_idx": layer,
                                "head_idx": head}, "rcm_zero", scan_config))
        values = []
        for row, measured_row, old in zip(rows, measured, original, strict=True):
            native = float(measured_row["native_margin"])
            zero = float(measured_row["intervened_margin"])
            effect = float(measured_row["effect"])
            if abs((native-zero)-effect) > 1e-12:
                raise ValueError("RCM operand subtraction disagrees with backend effect")
            values.append({
                "sample_id": row["sample_id"], "component_id": component,
                **measured_row, "recorded_effect": float(old["score"]),
                "difference_from_recorded": effect-float(old["score"]),
            })
        _component_rows(checkpoint, values)
        print(f"[rcm evidence] {index}/{len(groups)} {component}", flush=True)

    total = exact = 0
    maximum = 0.0
    means = []
    for component in groups:
        values = load_rows(output_root / "components" / (component + ".jsonl"))
        total += len(values)
        exact += sum(row["difference_from_recorded"] == 0.0 for row in values)
        maximum = max(maximum, *(abs(row["difference_from_recorded"]) for row in values))
        mean = {"component_id": component,
                "recorded_mean_effect": sum(x["recorded_effect"] for x in values)/len(values),
                "recomputed_mean_effect": sum(x["effect"] for x in values)/len(values)}
        if dataset == "imdb" and not replay_interventions:
            mean["normalized_mean_effect"] = sum(x["normalized_effect"] for x in values)/len(values)
        means.append(mean)
    _save(output_root / "component_means.json", means)
    summary = {
        "status": "complete", "source_cell_id": run["identity"]["cell_id"],
        "component_sample_rows": total, "exactly_matching_source_rows": exact,
        "max_absolute_effect_difference": maximum,
        "historical_effects_replaced": False,
        "reconstruction_status": "exact_operands_recovered" if total == exact else "recomputed_operands_with_reported_differences",
        "text_normalization_diagnostic": dataset == "imdb" and not replay_interventions,
        "replay_interventions": replay_interventions,
        "full_pipeline_reproduced": False,
    }
    _save(output_root / "completion.json", summary)
    return summary
