"""Reusable evaluation primitives, independent of the intervention method."""

from __future__ import annotations

import json
import math
from statistics import mean
from typing import Any, Callable, Mapping, Sequence

from baseline.implementations.loaders import load_rows
from experiments.shared.contracts import EvaluationRequest, EvaluationResult


AlignedRows = list[tuple[Mapping[str, Any], Mapping[str, Any]]]
ScoredRows = Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]


def align_prediction_references(request: EvaluationRequest) -> AlignedRows:
    """Load prediction/reference artifacts and align them by source sample id."""

    predictions = load_rows(request.predictions, root=request.evidence_root)
    references = load_rows(request.references, root=request.evidence_root)
    if not predictions or not references:
        raise ValueError("evaluation predictions and references must be non-empty")
    reference_by_id: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(references):
        sample_id = str(row.get("sample_id", index))
        if sample_id in reference_by_id:
            raise ValueError(f"duplicate reference sample_id={sample_id}")
        reference_by_id[sample_id] = row
    aligned: AlignedRows = []
    seen_predictions: set[str] = set()
    for index, prediction in enumerate(predictions):
        sample_id = str(prediction.get("sample_id", index))
        if sample_id in seen_predictions:
            raise ValueError(f"duplicate prediction sample_id={sample_id}")
        seen_predictions.add(sample_id)
        source_sample_id = str(prediction.get("source_sample_id", sample_id))
        reference = reference_by_id.get(source_sample_id)
        if reference is None:
            raise ValueError(f"missing reference for sample_id={source_sample_id}")
        aligned.append((prediction, reference))
    return aligned


def write_evaluation(
    request: EvaluationRequest,
    scored_rows: ScoredRows,
) -> EvaluationResult:
    """Persist numeric per-sample scores and their alpha-aware aggregate."""

    if not scored_rows:
        raise ValueError("evaluation scorer produced no rows")
    rows: list[dict[str, Any]] = []
    for index, (prediction, scored) in enumerate(scored_rows):
        if not isinstance(scored, Mapping):
            raise ValueError("evaluation scorer must return an object")
        sample_id = str(prediction.get("sample_id", index))
        normalized: dict[str, Any] = {"sample_id": sample_id}
        if "alpha" in prediction:
            normalized["alpha"] = prediction["alpha"]
        for key, value in scored.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"evaluation score {key!r} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"evaluation score {key!r} must be finite")
            normalized[str(key)] = float(value)
        rows.append(normalized)
    request.output_dir.mkdir(parents=True, exist_ok=True)
    per_sample = request.output_dir / "per_sample_scores.jsonl"
    with per_sample.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    metric_names = sorted(
        {key for row in rows for key in row if key not in {"sample_id", "alpha"}}
    )
    if any("alpha" in row for row in rows):
        alpha_values = sorted({float(row["alpha"]) for row in rows})
        summary: dict[str, Any] = {
            "rows": [
                {
                    "alpha": alpha,
                    **{
                        name: mean(
                            float(row[name])
                            for row in rows
                            if float(row["alpha"]) == alpha and name in row
                        )
                        for name in metric_names
                    },
                }
                for alpha in alpha_values
            ],
            "prediction_rows": len(rows),
        }
    else:
        summary = {
            "rows": len(rows),
            "metrics": {
                name: mean(float(row[name]) for row in rows if name in row)
                for name in metric_names
            },
        }
    summary_path = request.output_dir / "summary_metrics.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return EvaluationResult(per_sample_scores=per_sample, summary_metrics=summary_path)


class CallbackEvaluator:
    """Apply one task row scorer to aligned prediction/reference rows."""

    def __init__(
        self,
        score: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]],
    ) -> None:
        self.score = score

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        return write_evaluation(
            request,
            [
                (prediction, self.score(prediction, reference))
                for prediction, reference in align_prediction_references(request)
            ],
        )
