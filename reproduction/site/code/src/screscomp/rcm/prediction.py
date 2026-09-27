from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any


@dataclass(slots=True)
class PredictionMetric:
    target: str
    score_name: str
    n: int
    positives: int
    positive_rate: float
    auroc: float | None
    average_precision: float | None
    top_bottom_gap: float | None
    decile_monotonicity: float | None
    mean_score_positive: float | None
    mean_score_negative: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n", ""}:
            return False
    return bool(value)


def _average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def auroc(labels: list[bool], scores: list[float]) -> float | None:
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = _average_ranks(scores)
    pos_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision(labels: list[bool], scores: list[float]) -> float | None:
    n_pos = sum(labels)
    if n_pos == 0:
        return None
    ordered = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    hits = 0
    precisions: list[float] = []
    for rank, (_score, label) in enumerate(ordered, start=1):
        if label:
            hits += 1
            precisions.append(hits / rank)
    return sum(precisions) / n_pos


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = mean(xs)
    mean_y = mean(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = sum((x - mean_x) ** 2 for x in xs)
    den_y = sum((y - mean_y) ** 2 for y in ys)
    if den_x <= 1e-12 or den_y <= 1e-12:
        return None
    return num / ((den_x * den_y) ** 0.5)


def decile_profile(labels: list[bool], scores: list[float], *, bins: int = 10) -> list[dict[str, float]]:
    if not labels or len(labels) != len(scores):
        return []
    ordered = sorted(zip(scores, labels), key=lambda item: item[0])
    output: list[dict[str, float]] = []
    for idx in range(bins):
        start = int(idx * len(ordered) / bins)
        end = int((idx + 1) * len(ordered) / bins)
        local = ordered[start:end]
        if not local:
            continue
        local_labels = [label for _score, label in local]
        local_scores = [score for score, _label in local]
        output.append(
            {
                "decile": idx + 1,
                "n": len(local),
                "positive_rate": sum(local_labels) / len(local_labels),
                "mean_score": mean(local_scores),
            }
        )
    return output


def top_bottom_gap(labels: list[bool], scores: list[float]) -> float | None:
    profile = decile_profile(labels, scores)
    if len(profile) < 2:
        return None
    return profile[-1]["positive_rate"] - profile[0]["positive_rate"]


def decile_monotonicity(labels: list[bool], scores: list[float]) -> float | None:
    profile = decile_profile(labels, scores)
    if len(profile) < 2:
        return None
    return _pearson(
        [float(row["decile"]) for row in profile],
        [float(row["positive_rate"]) for row in profile],
    )


def summarize_prediction_scores(
    rows: list[dict[str, Any]],
    *,
    target_scores: dict[str, list[str]],
) -> list[PredictionMetric]:
    output: list[PredictionMetric] = []
    for target, score_names in target_scores.items():
        labels = [_as_bool(row.get(target)) for row in rows]
        positives = sum(labels)
        for score_name in score_names:
            scored = [(label, float(row[score_name])) for label, row in zip(labels, rows) if row.get(score_name) not in ("", None)]
            if not scored:
                continue
            local_labels = [label for label, _score in scored]
            scores = [score for _label, score in scored]
            local_positives = sum(local_labels)
            output.append(
                PredictionMetric(
                    target=target,
                    score_name=score_name,
                    n=len(scored),
                    positives=local_positives,
                    positive_rate=local_positives / len(scored) if scored else 0.0,
                    auroc=auroc(local_labels, scores),
                    average_precision=average_precision(local_labels, scores),
                    top_bottom_gap=top_bottom_gap(local_labels, scores),
                    decile_monotonicity=decile_monotonicity(local_labels, scores),
                    mean_score_positive=mean(score for label, score in scored if label) if local_positives else None,
                    mean_score_negative=mean(score for label, score in scored if not label)
                    if local_positives < len(scored)
                    else None,
                )
            )
    return output
