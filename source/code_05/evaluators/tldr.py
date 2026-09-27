"""Human-reference judge interface for the formal TL;DR evaluator."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from experiments.shared.contracts import EvaluationRequest, EvaluationResult

from .callback import align_prediction_references, write_evaluation
from .tldr_health import summarize_tldr_health


class HumanReferenceJudge(Protocol):
    """External TL;DR Appendix C.2 judge supplied by its evaluator backend."""

    def evaluate_pair(
        self,
        *,
        prediction: Mapping[str, Any],
        human_reference: Mapping[str, Any],
        evaluation_config: Mapping[str, Any],
        checkpoint_dir: Path | None = None,
        checkpoint_key: str | None = None,
    ) -> Mapping[str, Any]:
        """Return paired numeric wins plus an inspectable two-order trace."""


def _numeric_verdict(verdict: Mapping[str, Any]) -> dict[str, float]:
    values = {
        "first_pass_win": verdict.get("first_pass_win"),
        "order_swap_win": verdict.get("order_swap_win"),
        "order_swap_agreement": verdict.get("order_swap_agreement"),
    }
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in values.values()
    ):
        raise ValueError("TLDR human judge must return numeric first/swap/agreement values")
    return {name: float(value) for name, value in values.items()}


def _write_trace(output_dir: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "judge_trace.jsonl"
    newline = chr(10)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + newline)
    return path


def _alpha_groups(
    rows: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> dict[float | None, list[tuple[Mapping[str, Any], Mapping[str, Any]]]]:
    groups: dict[float | None, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = {}
    for prediction, reference in rows:
        value = prediction.get("alpha")
        if value is None:
            key = None
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            key = float(value)
        else:
            raise ValueError("TLDR prediction alpha must be numeric when present")
        groups.setdefault(key, []).append((prediction, reference))
    return groups


def _write_health_summary(
    *,
    request: EvaluationRequest,
    evaluation_result: EvaluationResult,
    aligned: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> Path:
    groups = _alpha_groups(aligned)
    summaries = {alpha: summarize_tldr_health(rows) for alpha, rows in groups.items()}
    summary_path = evaluation_result.summary_metrics
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    detail_rows: list[dict[str, Any]] = []

    if "rows" in summary and isinstance(summary["rows"], list):
        for row in summary["rows"]:
            alpha = row.get("alpha")
            if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
                raise ValueError("TLDR alpha summary requires numeric alpha")
            metrics, details = summaries[float(alpha)]
            row.update(metrics)
            detail_rows.append({"alpha": float(alpha), "health": details})
    elif set(groups) == {None}:
        metrics, details = summaries[None]
        summary.setdefault("metrics", {}).update(metrics)
        detail_rows.append({"alpha": None, "health": details})
    else:
        raise ValueError("TLDR alpha-group health cannot match evaluator summary")

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + chr(10),
        encoding="utf-8",
    )
    detail_path = request.output_dir / "tldr_health_details.json"
    detail_path.write_text(
        json.dumps({"groups": detail_rows}, ensure_ascii=False, indent=2, sort_keys=True)
        + chr(10),
        encoding="utf-8",
    )
    return detail_path


class TLDRHumanReferenceEvaluator:
    """Wrap a real human-reference judge; it has no lexical fallback."""

    def __init__(self, judge: HumanReferenceJudge) -> None:
        self.judge = judge

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        aligned = align_prediction_references(request)
        profile = request.evaluation_config.get("profile")
        if not isinstance(profile, Mapping):
            raise ValueError("TLDR evaluation.profile must be an object")
        judge_config = profile.get("judge")
        if not isinstance(judge_config, Mapping):
            raise ValueError("TLDR evaluation.profile.judge must be an object")
        request_config = judge_config.get("request")
        if not isinstance(request_config, Mapping):
            raise ValueError("TLDR judge.request must be an object")
        concurrency = request_config.get("concurrency")
        if (
            not isinstance(concurrency, int)
            or isinstance(concurrency, bool)
            or concurrency < 1
        ):
            raise ValueError("TLDR judge.request.concurrency must be a positive integer")

        def evaluate_one(
            indexed_pair: tuple[int, tuple[Mapping[str, Any], Mapping[str, Any]]],
        ) -> tuple[Mapping[str, Any], dict[str, float], Mapping[str, Any]]:
            index, pair = indexed_pair
            prediction, reference = pair
            verdict = self.judge.evaluate_pair(
                prediction=prediction,
                human_reference=reference,
                evaluation_config=request.evaluation_config,
                checkpoint_dir=request.output_dir / "judge_calls",
                checkpoint_key=f"row_{index:06d}",
            )
            scores = _numeric_verdict(verdict)
            trace = verdict.get("trace")
            if not isinstance(trace, Mapping):
                raise ValueError("TLDR human judge must return an inspectable trace")
            return prediction, scores, trace

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            evaluated = list(executor.map(evaluate_one, enumerate(aligned)))

        scored: list[tuple[Mapping[str, Any], Mapping[str, float]]] = []
        trace_rows: list[Mapping[str, Any]] = []
        for prediction, scores, trace in evaluated:
            scored.append(
                (
                    prediction,
                    {
                        "first_pass_win_rate": scores["first_pass_win"],
                        "order_swap_win_rate": scores["order_swap_win"],
                        "balanced_win_rate_against_human_summary": (
                            scores["first_pass_win"] + scores["order_swap_win"]
                        )
                        / 2.0,
                        "order_swap_agreement": scores["order_swap_agreement"],
                    },
                )
            )
            trace_rows.append(
                {
                    "sample_id": str(prediction.get("sample_id", "")),
                    "source_sample_id": str(
                        prediction.get(
                            "source_sample_id",
                            prediction.get("sample_id", ""),
                        )
                    ),
                    "judge_trace": dict(trace),
                }
            )
        result = write_evaluation(request, scored)
        health_details = _write_health_summary(
            request=request,
            evaluation_result=result,
            aligned=aligned,
        )
        return EvaluationResult(
            per_sample_scores=result.per_sample_scores,
            summary_metrics=result.summary_metrics,
            auxiliary_artifacts=(
                _write_trace(request.output_dir, trace_rows),
                health_details,
            ),
        )
