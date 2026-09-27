"""Mature-style coarse-layer then full-head RCM scanning."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..loaders import load_model, load_rows, resolve_model_id
from ..position_scanning import PositionScanResult, summarize_measurements


@dataclass(frozen=True)
class RCMScanRequest:
    method: str
    data_role_manifest: Path
    model_config: Mapping[str, Any]
    scan_config: Mapping[str, Any]
    trajectory_config: Mapping[str, Any]
    evidence_root: Path
    output_dir: Path


def _score(row: Mapping[str, Any]) -> float:
    for field in ("effect", "mean_score_delta", "selection_score"):
        value = row.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    raise ValueError(f"RCM measurement has no effect score: {row}")


def _beam_width(config: Mapping[str, Any]) -> int:
    value = config.get(
        "parent_layers_per_role",
        config.get("coarse_beam_width", config.get("coarse_parent_layers_per_role")),
    )
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("RCM scan requires a positive parent_layers_per_role")
    return value


def select_coarse_layer_beams(
    rows: Sequence[Mapping[str, Any]], *, width: int
) -> dict[str, Any]:
    """Keep signed beams when possible; otherwise use disjoint CI-ranked extrema."""
    if width <= 0 or len(rows) < 2 * width:
        raise ValueError("RCM coarse search requires at least twice the beam width in layers")
    layer_ids = [int(row["layer_idx"]) for row in rows]
    if len(set(layer_ids)) != len(layer_ids):
        raise ValueError("RCM coarse search contains duplicate layers")
    for row in rows:
        effect = float(row["mean_score_delta"])
        if not all(math.isfinite(float(value)) for value in (
            effect, row.get("ci95_low", effect), row.get("ci95_high", effect)
        )):
            raise ValueError(f"RCM layer {row['layer_idx']} contains non-finite scores")
    high_key = lambda row: (-float(row.get("ci95_low", row["mean_score_delta"])), int(row["layer_idx"]))
    low_key = lambda row: (float(row.get("ci95_high", row["mean_score_delta"])), int(row["layer_idx"]))
    positive = [row for row in rows if float(row["mean_score_delta"]) > 0.0]
    negative = [row for row in rows if float(row["mean_score_delta"]) < 0.0]
    signed = len(positive) >= width and len(negative) >= width
    high = sorted(positive if signed else rows, key=high_key)[:width]
    high_ids = {int(row["layer_idx"]) for row in high}
    low_pool = negative if signed else [row for row in rows if int(row["layer_idx"]) not in high_ids]
    low = sorted(low_pool, key=low_key)[:width]
    return {
        "policy": "signed_then_extrema_fixed_width",
        "mode": "signed" if signed else "extrema",
        "reason": "both_signed_beams_available" if signed else "insufficient_signed_layers",
        "width": width,
        "sign_counts": {"positive": len(positive), "negative": len(negative),
                        "zero": len(rows) - len(positive) - len(negative)},
        "high_effect_layer_beam": [int(row["layer_idx"]) for row in high],
        "low_effect_layer_beam": [int(row["layer_idx"]) for row in low],
        "positive_layer_beam": [int(row["layer_idx"]) for row in high] if signed else [],
        "negative_layer_beam": [int(row["layer_idx"]) for row in low] if signed else [],
        "tie_break": "lower_layer_index",
        "deduplication": "high_end_first_then_low_end_from_remaining_layers",
        "head_sign_rule": "each_head_own_effect",
    }


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(value), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sample_id(row: Mapping[str, Any], index: int) -> str:
    raw_id = row.get("sample_id", row.get("id", index))
    return raw_id if isinstance(raw_id, str) and raw_id else str(raw_id)


def _checkpoint_name(component_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", component_id) + ".json"


def _load_candidate_checkpoint(
    path: Path,
    *,
    phase: str,
    method: str,
    position: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    expected_ids = [_sample_id(row, index) for index, row in enumerate(rows)]
    if (
        not isinstance(value, Mapping)
        or value.get("status") != "complete"
        or value.get("phase") != phase
        or value.get("method") != method
        or value.get("position") != dict(position)
        or value.get("sample_ids") != expected_ids
    ):
        raise ValueError(
            f"RCM candidate checkpoint conflicts with the registered request: {path}"
        )
    measurements = value.get("measurements")
    trajectories = value.get("trajectories")
    if (
        not isinstance(measurements, list)
        or len(measurements) != len(rows)
        or not all(isinstance(row, dict) for row in measurements)
        or not isinstance(trajectories, list)
        or not all(isinstance(row, dict) for row in trajectories)
    ):
        raise ValueError(f"RCM candidate checkpoint is incomplete: {path}")
    if [row.get("sample_id") for row in measurements] != expected_ids:
        raise ValueError(f"RCM candidate checkpoint sample order drift: {path}")
    return measurements, trajectories


class CoarseToFineRCMScanner:
    """Scan layers, retain signed beams, then scan every head in their union."""

    def __init__(
        self,
        *,
        model_provider: Any,
        layer_count: Callable[[Any], int],
        head_count: Callable[[Any, int], int],
        measure: Callable[
            [Any, Mapping[str, Any], Mapping[str, Any], str, Mapping[str, Any]],
            Mapping[str, Any],
        ],
        measure_many: Callable[
            [Any, Sequence[Mapping[str, Any]], Mapping[str, Any], str, Mapping[str, Any]],
            Sequence[Mapping[str, Any]],
        ] | None = None,
        prepare_patch_prototypes: Callable[
            [Any, Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]], Mapping[str, Any]],
            Mapping[str, Any],
        ] | None = None,
    ) -> None:
        self.model_provider = model_provider
        self.layer_count = layer_count
        self.head_count = head_count
        self.measure = measure
        self.measure_many = measure_many
        self.prepare_patch_prototypes = prepare_patch_prototypes

    def _measure_positions(
        self,
        *,
        model: Any,
        rows: Sequence[Mapping[str, Any]],
        positions: Sequence[Mapping[str, Any]],
        method: str,
        config: Mapping[str, Any],
        output_dir: Path,
        phase: str,
        trajectory_enabled: bool,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        patch_prototypes: Mapping[str, Any] = {}
        if method == "rcm_patch":
            if self.prepare_patch_prototypes is None:
                raise ValueError("RCM-patch scanner lacks a global prototype preparer")
            patch_prototypes = self.prepare_patch_prototypes(model, rows, positions, config)
            if set(patch_prototypes) != {
                str(position["component_id"]) for position in positions
            }:
                raise ValueError("RCM-patch prototype preparer did not cover the candidate set")

        measured_rows: list[dict[str, Any]] = []
        trajectories: list[dict[str, Any]] = []

        checkpoint_dir = output_dir / "candidate_checkpoints" / phase
        progress_path = output_dir / "progress.json"
        for position_index, position in enumerate(positions):
            checkpoint_path = checkpoint_dir / _checkpoint_name(
                str(position["component_id"])
            )
            cached = _load_candidate_checkpoint(
                checkpoint_path,
                phase=phase,
                method=method,
                position=position,
                rows=rows,
            )
            if cached is not None:
                candidate_measurements, candidate_trajectories = cached
                measured_rows.extend(candidate_measurements)
                trajectories.extend(candidate_trajectories)
                print(
                    f"[rcm] resume {phase} candidate {position_index + 1}/{len(positions)} "
                    f"{position['component_id']}",
                    flush=True,
                )
                continue

            measurement_config = dict(config)
            if method == "rcm_patch":
                measurement_config["_rcm_patch_prototype"] = patch_prototypes[
                    str(position["component_id"])
                ]
            candidate_measurements: list[dict[str, Any]] = []
            candidate_trajectories: list[dict[str, Any]] = []

            def append_measurement(
                row_index: int,
                row: Mapping[str, Any],
                measured: Mapping[str, Any],
            ) -> None:
                sample_id = _sample_id(row, row_index)
                effect = _score(measured)
                candidate_measurements.append(
                    {
                        "sample_id": sample_id,
                        "component_id": position["component_id"],
                        "layer_idx": position["layer_idx"],
                        "head_idx": position.get("head_idx"),
                        "effect": effect,
                        "mean_score_delta": effect,
                        "method": method,
                    }
                )
                text = measured.get("generated_text", measured.get("text"))
                if trajectory_enabled and isinstance(text, str):
                    candidate_trajectories.append(
                        {
                            "sample_id": sample_id,
                            "component_id": position["component_id"],
                            "generated_text": text,
                        }
                    )

            if self.measure_many is not None:
                batch = list(
                    self.measure_many(
                        model,
                        rows,
                        position,
                        method,
                        measurement_config,
                    )
                )
                if len(batch) != len(rows):
                    raise ValueError("RCM batch measurement count differs from selector rows")
                for row_index, (row, measured) in enumerate(zip(rows, batch, strict=True)):
                    if not isinstance(measured, Mapping):
                        raise ValueError("RCM batch measurement must return row objects")
                    append_measurement(row_index, row, measured)
            else:
                for row_index, row in enumerate(rows):
                    measured = self.measure(
                        model,
                        row,
                        position,
                        method,
                        measurement_config,
                    )
                    if not isinstance(measured, Mapping):
                        raise ValueError("RCM measurement callback must return an object")
                    append_measurement(row_index, row, measured)
            _atomic_write_json(
                checkpoint_path,
                {
                    "status": "complete",
                    "phase": phase,
                    "method": method,
                    "position": dict(position),
                    "sample_ids": [
                        _sample_id(row, index) for index, row in enumerate(rows)
                    ],
                    "measurements": candidate_measurements,
                    "trajectories": candidate_trajectories,
                },
            )
            measured_rows.extend(candidate_measurements)
            trajectories.extend(candidate_trajectories)
            _atomic_write_json(
                progress_path,
                {
                    "status": "running",
                    "phase": phase,
                    "completed_candidates": position_index + 1,
                    "total_candidates": len(positions),
                    "latest_component_id": position["component_id"],
                    "rows_per_candidate": len(rows),
                },
            )
            print(
                f"[rcm] complete {phase} candidate {position_index + 1}/{len(positions)} "
                f"{position['component_id']} rows={len(rows)}",
                flush=True,
            )

        imdb_states = bool(rows) and all(
            row.get("selector_semantics")
            == "shared_model_native_good_base_complete_states"
            for row in rows
        )
        prototype_record = {
            "prepared": method == "rcm_patch",
            "components": sorted(patch_prototypes),
            "aggregation": (
                [
                    "mean_decision_states_within_sequence",
                    "mean_samples",
                ]
                if imdb_states
                else [
                    "mean_decision_states_within_sequence",
                    "mean_aliases_within_sample",
                    "mean_samples",
                ]
            ) if method == "rcm_patch" else [],
        }
        return measured_rows, trajectories, prototype_record

    def scan(self, request: RCMScanRequest) -> PositionScanResult:
        if request.method not in {"rcm_zero", "rcm_patch"}:
            raise ValueError("RCM scanner method must be rcm_zero or rcm_patch")
        request.output_dir.mkdir(parents=True, exist_ok=True)
        model = load_model(self.model_provider, request.model_config, mode="inference")
        rows = load_rows(request.data_role_manifest, root=request.evidence_root)
        if not rows:
            raise ValueError("RCM selector role is empty")
        num_layers = int(self.layer_count(model))
        if num_layers <= 0:
            raise ValueError("RCM model adapter reported no layers")
        layer_positions = [
            {"component_id": f"L{layer}.attn", "layer_idx": layer, "head_idx": None}
            for layer in range(num_layers)
        ]
        layer_samples, layer_trajectories, layer_prototype_record = self._measure_positions(
            model=model,
            rows=rows,
            positions=layer_positions,
            method=request.method,
            config=request.scan_config,
            output_dir=request.output_dir,
            phase="layers",
            trajectory_enabled=bool(request.trajectory_config.get("enabled", True)),
        )
        layer_results = summarize_measurements(
            layer_samples,
            method=request.method,
            scan_config=request.scan_config,
        )
        width = _beam_width(request.scan_config)
        # Keep layer evidence even when selection or later head scanning cannot finish.
        _write_jsonl(request.output_dir / "layer_candidate_scores.jsonl", layer_results)
        _write_jsonl(request.output_dir / "layer_sample_scores.jsonl", layer_samples)
        layer_selection = select_coarse_layer_beams(layer_results, width=width)
        _atomic_write_json(request.output_dir / "layer_beam_selection.json", layer_selection)
        selected_layers = sorted(
            {*layer_selection["high_effect_layer_beam"], *layer_selection["low_effect_layer_beam"]}
        )
        head_positions: list[dict[str, Any]] = []
        for layer in selected_layers:
            heads = int(self.head_count(model, layer))
            if heads <= 0:
                raise ValueError(f"RCM model adapter reported no heads for layer {layer}")
            head_positions.extend(
                {
                    "component_id": f"L{layer}.attn.h{head}",
                    "layer_idx": layer,
                    "head_idx": head,
                }
                for head in range(heads)
            )
        head_samples, head_trajectories, head_prototype_record = self._measure_positions(
            model=model,
            rows=rows,
            positions=head_positions,
            method=request.method,
            config=request.scan_config,
            output_dir=request.output_dir,
            phase="heads",
            trajectory_enabled=bool(request.trajectory_config.get("enabled", True)),
        )
        head_results = summarize_measurements(
            head_samples,
            method=request.method,
            scan_config=request.scan_config,
        )
        candidate_scores = request.output_dir / "candidate_scores.jsonl"
        layer_scores = request.output_dir / "layer_candidate_scores.jsonl"
        layer_sample_scores = request.output_dir / "layer_sample_scores.jsonl"
        sample_scores = request.output_dir / "sample_scores.jsonl"
        component_scores = request.output_dir / "component_scores.jsonl"
        trajectory = request.output_dir / "trajectory.jsonl"
        _write_jsonl(candidate_scores, head_results)
        _write_jsonl(layer_scores, layer_results)
        _write_jsonl(layer_sample_scores, layer_samples)
        _write_jsonl(sample_scores, head_samples)
        _write_jsonl(
            component_scores,
            [
                {
                    "sample_id": row["sample_id"],
                    "component_id": row["component_id"],
                    "score": row["effect"],
                }
                for row in (*layer_samples, *head_samples)
            ],
        )
        trajectory_rows = (
            [*layer_trajectories, *head_trajectories]
            if bool(request.trajectory_config.get("enabled", True))
            else []
        )
        trajectory_path: Path | None = None
        if trajectory_rows:
            trajectory_path = trajectory
            _write_jsonl(trajectory, trajectory_rows)
        execution = request.output_dir / "scan_execution.json"
        execution_value = {
            "method": request.method,
            "model_id": resolve_model_id(request.model_config, mode="inference"),
            "rows": len(rows),
            "layers_scanned": num_layers,
            "coarse_layer_selection": layer_selection,
            "positive_layer_beam": layer_selection["positive_layer_beam"],
            "negative_layer_beam": layer_selection["negative_layer_beam"],
            "high_effect_layer_beam": layer_selection["high_effect_layer_beam"],
            "low_effect_layer_beam": layer_selection["low_effect_layer_beam"],
            "heads_scanned": len(head_positions),
            "patch_prototypes": {
                "layer_scan": layer_prototype_record,
                "head_scan": head_prototype_record,
            },
        }
        _atomic_write_json(execution, execution_value)
        _atomic_write_json(
            request.output_dir / "progress.json",
            {
                "status": "complete",
                "phase": "complete",
                "completed_layer_candidates": len(layer_positions),
                "completed_head_candidates": len(head_positions),
                "rows_per_candidate": len(rows),
            },
        )
        return PositionScanResult(
            candidate_scores=candidate_scores,
            execution_manifest=execution,
            trajectory_path=trajectory_path,
            component_scores_path=component_scores,
        )
