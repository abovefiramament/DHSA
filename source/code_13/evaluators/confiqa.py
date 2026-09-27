"""Protocol-defined ConFiQA open-generation metrics."""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

from experiments.shared.contracts import EvaluationRequest, EvaluationResult

from .callback import align_prediction_references, write_evaluation


def _values(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        if value.strip():
            yield value
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _values(item)


def _aliases(row: Mapping[str, Any], keys: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for key in keys:
        for value in _values(row.get(key)):
            normalized = _normalize(value)
            if normalized and normalized not in seen:
                seen.add(normalized)
                result.append(normalized)
    return tuple(result)


def _normalize(value: str) -> str:
    compact = re.sub(r"[^\w\s]", " ", value.casefold())
    return re.sub(r"\s+", " ", compact).strip()


def _contains_answer(text: str, aliases: tuple[str, ...]) -> bool:
    padded = f" {_normalize(text)} "
    return any(f" {answer} " in padded for answer in aliases)


def score_confiqa(
    prediction: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Score one answer under the locked context-versus-prior metric."""

    text = str(prediction.get("generated_text", "") or "").strip()
    target_aliases = _aliases(
        reference,
        ("cf_answers", "cf_answer", "chosen", "target", "answer"),
    )
    prior_aliases = _aliases(
        reference,
        ("orig_answers", "orig_answer", "prior_answer", "rejected", "non_target"),
    )
    if not target_aliases or not prior_aliases:
        raise ValueError("ConFiQA evaluation row requires context and prior answer aliases")
    normalized_text = _normalize(text)
    context_hit = float(_contains_answer(text, target_aliases))
    prior_hit = float(_contains_answer(text, prior_aliases))
    context_only = context_hit * (1.0 - prior_hit)
    prior_only = prior_hit * (1.0 - context_hit)
    both = context_hit * prior_hit
    neither = (1.0 - context_hit) * (1.0 - prior_hit)
    exact = float(normalized_text in target_aliases)
    # Exact target-only answers are the registered short-exact context category.
    short_exact_context = exact
    logic_score = (
        context_only
        + short_exact_context
        - prior_hit
        - neither
        - float(len(text)) / 400.0
    )
    return {
        "logic_score": logic_score,
        "pc": context_hit,
        "po": prior_hit,
        "mr": 1.0 - prior_hit,
        "em": exact,
        "context_only": context_only,
        "prior_only": prior_only,
        "both": both,
        "neither": neither,
        "short_exact_context": short_exact_context,
        "mean_chars": float(len(text)),
        "health": float(bool(text)),
        "actual_n": 1.0,
    }


class ConfiQAFormalEvaluator:
    """Formal ConFiQA evaluator, separate from generation and CAST."""

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        scored_rows = []
        for prediction, reference in align_prediction_references(request):
            scores = dict(score_confiqa(prediction, reference))
            subset = reference.get("source_subset")
            if subset is not None:
                if subset not in {"qa", "mr", "mc"}:
                    raise ValueError(f"unknown ConFiQA source_subset: {subset!r}")
                scores.update(
                    {f"{subset}_{name}": value for name, value in scores.items()}
                )
            scored_rows.append((prediction, scores))
        return write_evaluation(request, scored_rows)
