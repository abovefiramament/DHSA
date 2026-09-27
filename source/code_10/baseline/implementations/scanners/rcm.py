"""Mature-style coarse-layer then full-head RCM scanning."""

from __future__ import annotations

import json
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


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


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

        def append_measurement(
            row_index: int,
            row: Mapping[str, Any],
            position: Mapping[str, Any],
            measured: Mapping[str, Any],
        ) -> None:
            raw_id = row.get("sample_id", row.get("id", row_index))
            sample_id = raw_id if isinstance(raw_id, str) and raw_id else str(raw_id)
            effect = _score(measured)
            measured_rows.append(
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
            if isinstance(text, str):
                trajectories.append(
                    {
                        "sample_id": sample_id,
                        "component_id": position["component_id"],
                        "generated_text": text,
                    }
                )

        for position in positions:
            measurement_config = dict(config)
            if method == "rcm_patch":
                measurement_config["_rcm_patch_prototype"] = patch_prototypes[
                    str(position["component_id"])
                ]
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
                    append_measurement(row_index, row, position, measured)
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
                    append_measurement(row_index, row, position, measured)

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
        )
        layer_results = summarize_measurements(
            layer_samples,
            method=request.method,
            scan_config=request.scan_config,
        )
        width = _beam_width(request.scan_config)
        positive = sorted(
            (row for row in layer_results if float(row["mean_score_delta"]) > 0.0),
            key=lambda row: (-float(row.get("ci95_low", row["mean_score_delta"])), int(row["layer_idx"])),
        )[:width]
        negative = sorted(
            (row for row in layer_results if float(row["mean_score_delta"]) < 0.0),
            key=lambda row: (float(row.get("ci95_high", row["mean_score_delta"])), int(row["layer_idx"])),
        )[:width]
        if len(positive) != width or len(negative) != width:
            raise RuntimeError("RCM layer scan cannot form both registered signed beams")
        selected_layers = sorted(
            {int(row["layer_idx"]) for row in (*positive, *negative)}
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
        )
        head_results = summarize_measurements(
            head_samples,
            method=request.method,
            scan_config=request.scan_config,
        )
        request.output_dir.mkdir(parents=True, exist_ok=True)
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
        execution.write_text(
            json.dumps(
                {
                    "method": request.method,
                    "model_id": resolve_model_id(request.model_config, mode="inference"),
                    "rows": len(rows),
                    "layers_scanned": num_layers,
                    "positive_layer_beam": [row["layer_idx"] for row in positive],
                    "negative_layer_beam": [row["layer_idx"] for row in negative],
                    "heads_scanned": len(head_positions),
                    "patch_prototypes": {
                        "layer_scan": layer_prototype_record,
                        "head_scan": head_prototype_record,
                    },
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
            execution_manifest=execution,
            trajectory_path=trajectory_path,
            component_scores_path=component_scores,
        )
