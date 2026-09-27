"""Callback-driven RCM/ITI model scans.

The scanner owns row iteration and portable score artifacts.  The injected
measurement callback owns model-specific forward passes, zero/patch states, or
ITI probes.  This keeps model architecture details out of position selectors.
"""

from __future__ import annotations

import json
import math
import random
from statistics import mean
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .loaders import load_model, load_rows


@dataclass(frozen=True)
class PositionScanRequest:
    method: str
    data_role_manifest: Path
    model_config: Mapping[str, Any]
    scan_config: Mapping[str, Any]
    candidate_positions: tuple[Mapping[str, Any], ...]
    evidence_root: Path
    output_dir: Path


@dataclass(frozen=True)
class PositionScanResult:
    candidate_scores: Path
    execution_manifest: Path
    trajectory_path: Path | None = None
    component_scores_path: Path | None = None


def _sample_id(row: Mapping[str, Any], index: int) -> str:
    value = row.get("sample_id", row.get("id", index))
    return value if isinstance(value, str) and value else str(value)


def _component_id(position: Mapping[str, Any]) -> str:
    value = position.get("component_id")
    if not isinstance(value, str) or not value:
        layer, head = position.get("layer"), position.get("head")
        if isinstance(layer, int) and isinstance(head, int):
            value = f"L{layer}.attn.h{head}"
    if not isinstance(value, str) or not value:
        raise ValueError("candidate position needs component_id or integer layer/head")
    return value


def _numeric_fields(value: Mapping[str, Any]) -> dict[str, float]:
    numeric: dict[str, float] = {}
    for key, item in value.items():
        if isinstance(item, bool):
            continue
        if isinstance(item, (int, float)) and math.isfinite(float(item)):
            numeric[str(key)] = float(item)
    return numeric


def _required_measurement(method: str, result: Mapping[str, Any]) -> None:
    if not _numeric_fields(result):
        raise ValueError("position measurement must return at least one finite numeric field")
    if method in {"rcm_zero", "rcm_patch"}:
        names = {"effect", "mean_score_delta", "selection_score", "ci95_low", "ci95_high"}
    elif method == "iti":
        names = {"probe_score", "accuracy", "ranking_score", "selection_score"}
    else:
        names = set()
    if names and not names.intersection(result):
        raise ValueError(
            f"{method} measurement must expose one of {sorted(names)}"
        )


def _effect_value(row: Mapping[str, Any]) -> float | None:
    for field in ("effect", "mean_score_delta", "selection_score", "probe_score", "accuracy", "ranking_score"):
        value = row.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            return float(value)
    return None


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute a quantile from no values")
    index = probability * (len(ordered) - 1)
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_measurements(rows: Sequence[Mapping[str, Any]], *, method: str, scan_config: Mapping[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["component_id"]), []).append(row)
    bootstrap = scan_config.get("bootstrap")
    if bootstrap is not None and not isinstance(bootstrap, Mapping):
        raise ValueError("scan_config.bootstrap must be an object")
    seed = int(bootstrap["seed"]) if isinstance(bootstrap, Mapping) and "seed" in bootstrap else None
    resamples = int(bootstrap.get("resamples", bootstrap.get("samples", 0))) if isinstance(bootstrap, Mapping) else 0
    percentiles = bootstrap.get("percentiles", (2.5, 97.5)) if isinstance(bootstrap, Mapping) else (2.5, 97.5)
    if not isinstance(percentiles, Sequence) or isinstance(percentiles, (str, bytes)) or len(percentiles) != 2:
        raise ValueError("bootstrap.percentiles must contain exactly two values")
    low_probability, high_probability = float(percentiles[0]) / 100.0, float(percentiles[1]) / 100.0
    if not 0.0 <= low_probability < high_probability <= 1.0:
        raise ValueError("bootstrap.percentiles must be ordered inside [0, 100]")
    if resamples < 0:
        raise ValueError("bootstrap.resamples must be non-negative")
    if resamples and seed is None:
        raise ValueError("bootstrap.seed is required when resamples are enabled")
    output: list[dict[str, Any]] = []
    for component_id, component_rows in grouped.items():
        values = [_effect_value(row) for row in component_rows]
        values = [value for value in values if value is not None]
        if not values:
            raise ValueError(f"component {component_id} has no numeric score")
        first = component_rows[0]
        layer_idx = first.get("layer_idx", first.get("layer"))
        head_idx = first.get("head_idx", first.get("head"))
        if layer_idx is None and component_id.startswith("L"):
            layer_idx = int(component_id[1:].split(".attn", 1)[0])
        if head_idx is None and ".attn.h" in component_id:
            head_idx = int(component_id.split(".attn.h", 1)[1])
        summary: dict[str, Any] = {
            "component_id": component_id,
            "layer_idx": int(layer_idx) if layer_idx is not None else None,
            "head_idx": int(head_idx) if head_idx is not None else None,
            "sample_count": len(values),
            "effect": mean(values),
            "mean_score_delta": mean(values),
            "selection_score": mean(values),
            "method": method,
        }
        if resamples:
            rng = random.Random(seed)
            draws = [mean(rng.choice(values) for _ in values) for _ in range(resamples)]
            summary["ci95_low"] = _quantile(draws, low_probability)
            summary["ci95_high"] = _quantile(draws, high_probability)
        output.append(summary)
    return sorted(output, key=lambda row: (row["layer_idx"] is None, row["layer_idx"], row["head_idx"] is None, row["head_idx"], row["component_id"]))


class CallbackPositionScanner:
    """Execute one real model scan through an injected measurement callback."""

    def __init__(
        self,
        *,
        model_provider: Any,
        measure: Callable[
            [Any, Mapping[str, Any], Mapping[str, Any], str, Mapping[str, Any]],
            Mapping[str, Any],
        ],
    ) -> None:
        self.model_provider = model_provider
        self.measure = measure

    def scan(self, request: PositionScanRequest) -> PositionScanResult:
        method = str(request.method)
        if method not in {"rcm_zero", "rcm_patch", "iti"}:
            raise ValueError("position scan method must be rcm_zero, rcm_patch, or iti")
        if not request.candidate_positions:
            raise ValueError("position scan requires at least one candidate position")
        positions = tuple(
            {**dict(position), "component_id": _component_id(position)}
            for position in request.candidate_positions
        )
        component_ids = [position["component_id"] for position in positions]
        if len(set(component_ids)) != len(component_ids):
            raise ValueError("position scan candidate IDs must be unique")
        model = load_model(self.model_provider, request.model_config, mode="inference")
        rows = load_rows(request.data_role_manifest, root=request.evidence_root)
        if not rows:
            raise ValueError("position scan data role is empty")
        request.output_dir.mkdir(parents=True, exist_ok=True)
        candidate_scores = request.output_dir / "candidate_scores.jsonl"
        sample_scores_path = request.output_dir / "sample_scores.jsonl"
        trajectory = request.output_dir / "trajectory.jsonl"
        component_scores = request.output_dir / "component_scores.jsonl"
        score_rows: list[dict[str, Any]] = []
        trajectory_rows: list[dict[str, Any]] = []
        component_rows: list[dict[str, Any]] = []
        for row_index, row in enumerate(rows):
            sample_id = _sample_id(row, row_index)
            for position in positions:
                measured = self.measure(model, row, position, method, request.scan_config)
                if not isinstance(measured, Mapping):
                    raise ValueError("position measurement callback must return an object")
                _required_measurement(method, measured)
                normalized: dict[str, Any] = {
                    "sample_id": sample_id,
                    "component_id": position["component_id"],
                    "method": method,
                }
                for key, value in measured.items():
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        normalized[str(key)] = value
                score_rows.append(normalized)
                numeric = _numeric_fields(measured)
                component_rows.extend(
                    {
                        "sample_id": sample_id,
                        "component_id": position["component_id"],
                        "score_name": key,
                        "score": value,
                    }
                    for key, value in numeric.items()
                )
                generated_text = measured.get("generated_text", measured.get("text"))
                if isinstance(generated_text, str):
                    trajectory_rows.append(
                        {
                            "sample_id": sample_id,
                            "component_id": position["component_id"],
                            "generated_text": generated_text,
                        }
                    )
        with candidate_scores.open("w", encoding="utf-8") as handle:
            for item in summarize_measurements(score_rows, method=method, scan_config=request.scan_config):
                handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        with sample_scores_path.open("w", encoding="utf-8") as handle:
            for item in score_rows:
                handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        component_path: Path | None = None
        trajectory_path: Path | None = None
        if component_rows:
            component_path = component_scores
            with component_path.open("w", encoding="utf-8") as handle:
                for item in component_rows:
                    handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        if trajectory_rows:
            trajectory_path = trajectory
            with trajectory_path.open("w", encoding="utf-8") as handle:
                for item in trajectory_rows:
                    handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        execution_manifest = request.output_dir / "scan_execution.json"
        execution_manifest.write_text(
            json.dumps(
                {
                    "method": method,
                    "model_id": request.model_config.get("model_id", request.model_config.get("id")),
                    "rows": len(rows),
                    "candidate_positions": len(positions),
                    "measurements": len(score_rows),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return PositionScanResult(
            candidate_scores=candidate_scores,
            execution_manifest=execution_manifest,
            trajectory_path=trajectory_path,
            component_scores_path=component_path,
        )
