"""Method-neutral selection among preregistered candidate validation curves."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from .contracts import AlphaSelectionRequest
from .evaluation_controller import DeterministicMetricAlphaSelector


def select_candidate_curves(
    curves: Mapping[str, Sequence[Mapping[str, Any]]],
    *, candidate_order: Sequence[str], alpha_grid: Sequence[float], metric: str,
) -> dict[str, Any]:
    """Choose each curve's peak with the shared selector, then the first best candidate."""
    if not candidate_order or len(set(candidate_order)) != len(candidate_order) or set(candidate_order) != set(curves):
        raise ValueError("candidate curves must cover exactly the registered order")
    rule = dict(alpha_field="alpha", alpha_grid=list(alpha_grid), primary_metric=metric,
                direction="maximize", constraints=[], near_best_tolerance=0.0,
                tie_break="smallest_alpha")
    selector = DeterministicMetricAlphaSelector()
    trace = [dict(candidate_id=name, **selector.select(AlphaSelectionRequest(
        curve_rows=tuple(curves[name]), selection_config=rule))) for name in candidate_order]
    winner = max(trace, key=lambda row: row["best_primary_metric"])
    return dict(status="frozen", selected_candidate=winner["candidate_id"],
                validation_peak_alpha=winner["selected_alpha"], primary_metric=metric,
                candidate_order=list(candidate_order), alpha_grid=list(alpha_grid),
                tie_break="first_in_registered_order_then_smallest_alpha", trace=trace)
