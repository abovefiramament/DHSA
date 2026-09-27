"""ITI position selector using precomputed grouped two-fold probe scores."""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

from .position_selection import SelectionResult, load_rows, plan_parameters, run_request, select_iti


def resolve_plan(config: Mapping[str, Any]) -> dict[str, Any]:
    baseline = config.get("baseline") if isinstance(config, Mapping) else None
    if not isinstance(baseline, Mapping) or baseline.get("method") != "iti":
        raise ValueError("ITI config must select baseline.method=iti")
    parameters = baseline.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("ITI baseline.parameters must be an object")
    return {"method": "iti", "method_kind": "position_baseline", **copy.deepcopy(dict(parameters))}


def select(rows: Sequence[Mapping[str, Any]], *, count: int, score_fields: Sequence[str] = ("ranking_score", "mean_grouped_twofold_heldout_accuracy", "selection_score")) -> SelectionResult:
    return select_iti(rows, count=count, score_fields=score_fields)


def select_positions(request_or_rows: Any, **kwargs: Any) -> Any:
    if hasattr(request_or_rows, "selector_data_manifest"):
        parameters = plan_parameters(request_or_rows.method_plan)
        search = parameters.get("position_search", parameters)
        rows = load_rows(request_or_rows.selector_data_manifest, root=getattr(request_or_rows, "evidence_root", None))
        result = select(rows, count=int(search.get("count", search.get("top_k", 8))), score_fields=tuple(search.get("score_fields", ("ranking_score", "mean_grouped_twofold_heldout_accuracy", "selection_score"))))
        return run_request(request_or_rows, result)
    parameters = kwargs or {}
    return list(select(request_or_rows, count=int(parameters["count"]), score_fields=tuple(parameters.get("score_fields", ("ranking_score", "mean_grouped_twofold_heldout_accuracy", "selection_score")))).positions)
