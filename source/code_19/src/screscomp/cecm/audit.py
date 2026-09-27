from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Callable, Iterable


@dataclass(frozen=True, slots=True)
class ScreenRow:
    component_id: str
    layer_idx: str
    component_type: str
    mean_delta: float
    abs_mean_delta: float
    stderr_delta: float
    positive_rate: float
    negative_rate: float
    sign_consistency: float
    n: int

    @property
    def sign(self) -> int:
        if self.mean_delta > 0:
            return 1
        if self.mean_delta < 0:
            return -1
        return 0


def load_component_screen(path: Path) -> dict[str, ScreenRow]:
    rows: dict[str, ScreenRow] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            component_id = str(row["component_id"])
            mean_delta = float(row["mean_delta"])
            rows[component_id] = ScreenRow(
                component_id=component_id,
                layer_idx=str(row.get("layer_idx", "")),
                component_type=str(row.get("component_type", "")),
                mean_delta=mean_delta,
                abs_mean_delta=abs(mean_delta),
                stderr_delta=float(row.get("stderr_delta", "nan")),
                positive_rate=float(row.get("positive_rate", "nan")),
                negative_rate=float(row.get("negative_rate", "nan")),
                sign_consistency=float(row.get("sign_consistency", "nan")),
                n=int(float(row.get("n", "0"))),
            )
    return rows


def z_for_alpha(alpha: float) -> float:
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0,1), got {alpha}")
    return NormalDist().inv_cdf(1.0 - alpha / 2.0)


def ci(row: ScreenRow, *, alpha: float) -> tuple[float, float]:
    half_width = z_for_alpha(alpha) * row.stderr_delta
    return row.mean_delta - half_width, row.mean_delta + half_width


def ci_excludes_zero(row: ScreenRow, *, alpha: float) -> bool:
    low, high = ci(row, alpha=alpha)
    return (low > 0 and high > 0) or (low < 0 and high < 0)


def accepted(row: ScreenRow, *, alpha: float, tau_delta: float) -> bool:
    return ci_excludes_zero(row, alpha=alpha) and abs(row.mean_delta) >= tau_delta and row.sign != 0


def split_status(
    a: ScreenRow | None,
    b: ScreenRow | None,
    *,
    alpha: float,
    tau_delta: float,
) -> str:
    if a is None or b is None:
        return "missing"
    if not accepted(a, alpha=alpha, tau_delta=tau_delta):
        return "split_a_not_accepted"
    if not accepted(b, alpha=alpha, tau_delta=tau_delta):
        return "split_b_not_accepted"
    if a.sign != b.sign:
        return "split_sign_flip"
    return "stable_positive" if a.sign > 0 else "stable_negative"


def same_sign_weak(a: ScreenRow | None, b: ScreenRow | None, *, tau_delta: float) -> bool:
    if a is None or b is None:
        return False
    if a.sign == 0 or b.sign == 0 or a.sign != b.sign:
        return False
    return abs(a.mean_delta) >= tau_delta and abs(b.mean_delta) >= tau_delta


def robust_status(
    avg_status: str,
    margin_status: str,
    *,
    avg_a: ScreenRow | None,
    avg_b: ScreenRow | None,
    margin_a: ScreenRow | None,
    margin_b: ScreenRow | None,
    tau_delta: float,
) -> tuple[str, str]:
    stable_avg = avg_status in {"stable_positive", "stable_negative"}
    stable_margin = margin_status in {"stable_positive", "stable_negative"}
    if stable_avg and stable_margin:
        if avg_status == margin_status:
            return (
                "robust_positive" if avg_status == "stable_positive" else "robust_negative",
                "high",
            )
        return "score_operator_sign_flip", "reject"
    if stable_avg:
        if same_sign_weak(margin_a, margin_b, tau_delta=tau_delta):
            avg_sign = 1 if avg_status == "stable_positive" else -1
            if margin_a is not None and margin_a.sign == avg_sign:
                return "avglogp_stable_margin_weak_same_sign", "medium"
            return "avglogp_stable_margin_opposite_weak", "low"
        return "avglogp_only_stable", "low"
    if stable_margin:
        if same_sign_weak(avg_a, avg_b, tau_delta=tau_delta):
            margin_sign = 1 if margin_status == "stable_positive" else -1
            if avg_a is not None and avg_a.sign == margin_sign:
                return "margin_stable_avglogp_weak_same_sign", "medium"
            return "margin_stable_avglogp_opposite_weak", "low"
        return "margin_only_stable", "low"
    return "not_stable", "reject"


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else math.nan


def dot(x: list[float], y: list[float]) -> float:
    return sum(a * b for a, b in zip(x, y))


def cosine(x: list[float], y: list[float]) -> float:
    denom = math.sqrt(dot(x, x)) * math.sqrt(dot(y, y))
    return dot(x, y) / denom if denom else math.nan


def pearson(x: list[float], y: list[float]) -> float:
    mx = mean(x)
    my = mean(y)
    return cosine([v - mx for v in x], [v - my for v in y])


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            out[order[k]] = rank
        i = j
    return out


def spearman(x: list[float], y: list[float]) -> float:
    return pearson(ranks(x), ranks(y))


MetricFn = Callable[[list[float], list[float]], float]


def bootstrap_ci(
    x: list[float],
    y: list[float],
    *,
    metric_fn: MetricFn,
    n_bootstrap: int,
    rng: random.Random,
) -> tuple[float, float]:
    if n_bootstrap <= 0 or not x:
        return math.nan, math.nan
    n = len(x)
    values: list[float] = []
    for _ in range(n_bootstrap):
        idx = [rng.randrange(n) for _ in range(n)]
        value = metric_fn([x[i] for i in idx], [y[i] for i in idx])
        if not math.isnan(value):
            values.append(value)
    if not values:
        return math.nan, math.nan
    values.sort()
    low_i = min(max(int(0.025 * len(values)), 0), len(values) - 1)
    high_i = min(max(int(0.975 * len(values)), 0), len(values) - 1)
    return values[low_i], values[high_i]


def permutation_p(
    x: list[float],
    y: list[float],
    *,
    observed: float,
    metric_fn: MetricFn,
    n_permutations: int,
    rng: random.Random,
) -> tuple[float, float]:
    if n_permutations <= 0 or not x:
        return math.nan, math.nan
    right = 0
    absolute = 0
    y_perm = list(y)
    for _ in range(n_permutations):
        rng.shuffle(y_perm)
        value = metric_fn(x, y_perm)
        if value >= observed:
            right += 1
        if abs(value) >= abs(observed):
            absolute += 1
    return (right + 1) / (n_permutations + 1), (absolute + 1) / (n_permutations + 1)


def paired_values(
    left: dict[str, ScreenRow],
    right: dict[str, ScreenRow],
) -> tuple[list[str], list[float], list[float]]:
    ids = sorted(set(left) & set(right))
    return ids, [left[cid].mean_delta for cid in ids], [right[cid].mean_delta for cid in ids]


def topk_ids(rows: dict[str, ScreenRow], *, direction: str, k: int) -> list[str]:
    if direction == "positive":
        candidates = [row for row in rows.values() if row.mean_delta > 0]
        candidates.sort(key=lambda row: (-row.mean_delta, row.component_id))
    elif direction == "negative":
        candidates = [row for row in rows.values() if row.mean_delta < 0]
        candidates.sort(key=lambda row: (row.mean_delta, row.component_id))
    else:
        raise ValueError(f"direction must be positive or negative, got {direction}")
    return [row.component_id for row in candidates[:k]]


def hypergeom_overlap_p(*, population: int, set_a: int, set_b: int, overlap: int) -> float:
    if population <= 0:
        return math.nan
    denom = math.comb(population, set_b)
    if denom == 0:
        return math.nan
    total = 0
    upper = min(set_a, set_b)
    for i in range(overlap, upper + 1):
        if set_b - i > population - set_a:
            continue
        total += math.comb(set_a, i) * math.comb(population - set_a, set_b - i)
    return total / denom


def topk_overlap_row(
    *,
    comparison: str,
    left: dict[str, ScreenRow],
    right: dict[str, ScreenRow],
    direction: str,
    k: int,
) -> dict[str, object]:
    left_top = topk_ids(left, direction=direction, k=k)
    right_top = topk_ids(right, direction=direction, k=k)
    left_set = set(left_top)
    right_set = set(right_top)
    overlap = left_set & right_set
    union = left_set | right_set
    population = len(set(left) & set(right))
    return {
        "comparison": comparison,
        "direction": direction,
        "k": k,
        "population": population,
        "left_top": ";".join(left_top),
        "right_top": ";".join(right_top),
        "overlap_ids": ";".join(sorted(overlap)),
        "overlap_count": len(overlap),
        "jaccard": len(overlap) / len(union) if union else math.nan,
        "hypergeom_p_ge_overlap": hypergeom_overlap_p(
            population=population,
            set_a=len(left_set),
            set_b=len(right_set),
            overlap=len(overlap),
        ),
    }
