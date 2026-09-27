"""CAA layer selector used by the native BiPO baseline.

The model-facing scanner constructs and evaluates the CAA vectors.  This file
only applies the preregistered deterministic top-k rule.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .position_selection import SelectionResult, load_rows, plan_parameters, run_request


def _score(row: Mapping[str, Any]) -> float:
    value = row.get("selection_score")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("CAA candidate requires a numeric selection_score")
    return float(value)


def _layer(row: Mapping[str, Any]) -> int:
    value = row.get("layer_idx", row.get("layer"))
    if isinstance(value, bool) or not isinstance(value, int):
        component = row.get("component_id")
        if isinstance(component, str) and component.startswith("L") and component.endswith(".block"):
            value = int(component[1:-6])
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("CAA candidate requires a non-negative layer_idx")
    return value


def select(
    rows: Sequence[Mapping[str, Any]], *, count: int
) -> SelectionResult:
    if count <= 0:
        raise ValueError("CAA candidate count must be positive")
    candidates = [dict(row) for row in rows]
    if len(candidates) < count:
        raise ValueError("CAA layer pool is smaller than the requested count")
    component_ids = [str(row.get("component_id", "")) for row in candidates]
    if any(not value for value in component_ids) or len(set(component_ids)) != len(component_ids):
        raise ValueError("CAA layer candidates need distinct component_id values")
    ranked = sorted(candidates, key=lambda row: (-_score(row), _layer(row)))
    selected = tuple(
        {
            **row,
            "component_id": str(row["component_id"]),
            "layer_idx": _layer(row),
            "selection_role": "native_candidate",
            "rank": rank,
            "ranking_score": _score(row),
        }
        for rank, row in enumerate(ranked[:count], start=1)
    )
    selected_ids = {str(row["component_id"]) for row in selected}
    trace = tuple(
        {
            "component_id": str(row["component_id"]),
            "layer_idx": _layer(row),
            "selected": str(row["component_id"]) in selected_ids,
            "effect": _score(row),
            "role": "native_candidate",
        }
        for row in ranked
    )
    return SelectionResult("caa", selected, tuple(ranked), trace)


def select_positions(request_or_rows: Any, **kwargs: Any) -> Any:
    if hasattr(request_or_rows, "selector_data_manifest"):
        parameters = plan_parameters(request_or_rows.method_plan)
        search = parameters.get("position_search", parameters)
        rows = load_rows(
            request_or_rows.selector_data_manifest,
            root=getattr(request_or_rows, "evidence_root", None),
        )
        result = select(rows, count=int(search["candidate_count"]))
        return run_request(request_or_rows, result)
    return list(select(request_or_rows, count=int(kwargs["candidate_count"])).positions)
