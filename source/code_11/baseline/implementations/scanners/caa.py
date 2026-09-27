"""Official-flow CAA localization for the native BiPO baseline."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from baseline.implementations.loaders import load_model, load_rows
from baseline.implementations.position_scanning import PositionScanResult
from experiments.shared.trajectory_controller import normalize_trajectory_config


@dataclass(frozen=True)
class CAAScanRequest:
    training_data_manifest: Path
    selector_data_manifest: Path
    model_config: Mapping[str, Any]
    scan_config: Mapping[str, Any]
    trajectory_config: Mapping[str, Any]
    evidence_root: Path
    output_dir: Path


class FullDepthCAAScanner:
    """Construct CAA on train rows and evaluate every layer on selector rows."""

    def __init__(
        self,
        *,
        model_provider: Any,
        layer_count: Callable[[Any], int],
        capture_pair: Callable[[Any, Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]],
        measure_many: Callable[
            [Any, Sequence[Mapping[str, Any]], int, np.ndarray, Mapping[str, Any]],
            Sequence[Mapping[str, Any]],
        ],
    ) -> None:
        self.model_provider = model_provider
        self.layer_count = layer_count
        self.capture_pair = capture_pair
        self.measure_many = measure_many

    def scan(self, request: CAAScanRequest) -> PositionScanResult:
        model = load_model(self.model_provider, request.model_config, mode="inference")
        training = load_rows(request.training_data_manifest, root=request.evidence_root)
        selector = load_rows(request.selector_data_manifest, root=request.evidence_root)
        if not training or not selector:
            raise ValueError("CAA requires non-empty training and selector roles")
        layers = int(self.layer_count(model))
        save_trace = normalize_trajectory_config(request.trajectory_config)["enabled"]
        delta_sum: np.ndarray | None = None
        capture_trace: list[dict[str, Any]] = []
        for index, row in enumerate(training):
            captured = self.capture_pair(model, row, request.scan_config)
            positive = np.asarray(captured["positive_activation"], dtype=np.float32)
            negative = np.asarray(captured["negative_activation"], dtype=np.float32)
            if positive.shape != negative.shape or positive.shape[0] != layers:
                raise ValueError("CAA capture geometry drift")
            if delta_sum is None:
                delta_sum = np.zeros_like(positive, dtype=np.float64)
            delta_sum += positive.astype(np.float64) - negative
            if save_trace:
                capture_trace.append(
                {
                    "sample_id": str(row.get("sample_id", row.get("id", index))),
                    "captured_layers": layers,
                }
            )
        vectors = (delta_sum / len(training)).astype(np.float32)
        request.output_dir.mkdir(parents=True, exist_ok=True)
        vector_dir = request.output_dir / "caa_vectors"
        vector_dir.mkdir(exist_ok=True)
        score_rows: list[dict[str, Any]] = []
        trajectory_rows: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        for layer, vector in enumerate(vectors):
            np.save(vector_dir / f"layer_{layer}.npy", vector)
            measured = list(
                self.measure_many(model, selector, layer, vector, request.scan_config)
            )
            if len(measured) != len(selector):
                raise ValueError("CAA selector measurement count drift")
            values: list[float] = []
            for index, (row, item) in enumerate(zip(selector, measured, strict=True)):
                value = item.get("selection_score")
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError("CAA measurement requires numeric selection_score")
                values.append(float(value))
                record = {
                    "sample_id": str(row.get("sample_id", row.get("id", index))),
                    "component_id": f"L{layer}.block",
                    "layer_idx": layer,
                    "selection_score": float(value),
                }
                if save_trace:
                    score_rows.append(record)
                text = item.get("generated_text")
                if save_trace and isinstance(text, str):
                    trajectory_rows.append({
                        **record, "generated_text": text,
                        "negative_generated_text": item.get("negative_generated_text", ""),
                    })
            summaries.append(
                {
                    "component_id": f"L{layer}.block",
                    "layer_idx": layer,
                    "sample_count": len(values),
                    "selection_score": mean(values),
                    "effect": mean(values),
                    "vector_file": f"caa_vectors/layer_{layer}.npy",
                    "method": "caa",
                }
            )
        candidate_scores = request.output_dir / "candidate_scores.jsonl"
        sample_scores = request.output_dir / "sample_scores.jsonl"
        trajectory = request.output_dir / "trajectory.jsonl"
        outputs = [(candidate_scores, summaries)]
        if save_trace:
            outputs.extend([
            (sample_scores, score_rows),
            (trajectory, [*capture_trace, *trajectory_rows]),
            ])
        for path, rows in outputs:
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        manifest = request.output_dir / "scan_execution.json"
        manifest.write_text(
            json.dumps(
                {
                    "method": "caa",
                    "training_rows": len(training),
                    "selector_rows": len(selector),
                    "layers": layers,
                    "vector_construction": "mean_positive_minus_negative_final_response_state",
                    "layer_selection_factor": 1.0,
                    "directions": [-1.0, 1.0],
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
            execution_manifest=manifest,
            trajectory_path=trajectory if save_trace else None,
            component_scores_path=sample_scores if save_trace else None,
        )
