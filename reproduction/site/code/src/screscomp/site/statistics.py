from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import mean
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class CandidateResult:
    component_id: str
    layer_idx: int
    head_idx: int | None
    deltas: tuple[float, ...]
    mean_score_delta: float
    ci95_low: float
    ci95_high: float


def percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        return math.nan
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def paired_bootstrap_interval(
    deltas: Iterable[float],
    *,
    samples: int,
    seed: int,
    interval_percentiles: tuple[float, float],
) -> tuple[float, float]:
    values = tuple(float(value) for value in deltas)
    if not values:
        return math.nan, math.nan
    rng = random.Random(seed)
    estimates = sorted(
        mean(values[rng.randrange(len(values))] for _ in range(len(values)))
        for _ in range(samples)
    )
    low, high = interval_percentiles
    return percentile(estimates, low / 100.0), percentile(estimates, high / 100.0)


def candidate_result(
    component_id: str,
    layer_idx: int,
    head_idx: int | None,
    deltas: Iterable[float],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
    interval_percentiles: tuple[float, float],
) -> CandidateResult:
    values = tuple(float(value) for value in deltas)
    low, high = paired_bootstrap_interval(
        values,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
        interval_percentiles=interval_percentiles,
    )
    return CandidateResult(
        component_id=component_id,
        layer_idx=layer_idx,
        head_idx=head_idx,
        deltas=values,
        mean_score_delta=mean(values) if values else math.nan,
        ci95_low=low,
        ci95_high=high,
    )


def role_oriented_ci95_lower_bound(result: CandidateResult) -> float:
    if result.mean_score_delta > 0:
        return result.ci95_low
    if result.mean_score_delta < 0:
        return -result.ci95_high
    return math.nan


def rank_positive(results: Iterable[CandidateResult], count: int) -> list[CandidateResult]:
    positive = [row for row in results if row.mean_score_delta > 0]
    positive.sort(
        key=lambda row: (
            -row.ci95_low,
            row.layer_idx,
            -1 if row.head_idx is None else row.head_idx,
        )
    )
    return positive[:count]


def rank_negative(results: Iterable[CandidateResult], count: int) -> list[CandidateResult]:
    negative = [row for row in results if row.mean_score_delta < 0]
    negative.sort(
        key=lambda row: (
            row.ci95_high,
            row.layer_idx,
            -1 if row.head_idx is None else row.head_idx,
        )
    )
    return negative[:count]


def result_rows(results: Iterable[CandidateResult]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        rows.append(
            {
                "component_id": result.component_id,
                "layer_idx": result.layer_idx,
                "head_idx": "" if result.head_idx is None else result.head_idx,
                "mean_score_delta": result.mean_score_delta,
                "ci95_low": result.ci95_low,
                "ci95_high": result.ci95_high,
                "role_oriented_ci95_lower_bound": role_oriented_ci95_lower_bound(result),
                "ci_crosses_zero": result.ci95_low <= 0.0 <= result.ci95_high,
                "n": len(result.deltas),
            }
        )
    return rows
