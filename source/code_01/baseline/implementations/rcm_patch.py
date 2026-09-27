"""RCM-patch position selector for the shared Site/Performance flow."""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

from .position_selection import SelectionResult, load_rows, plan_parameters, run_request, select_rcm


def resolve_plan(config: Mapping[str, Any]) -> dict[str, Any]:
    baseline = config.get("baseline") if isinstance(config, Mapping) else None
    if not isinstance(baseline, Mapping) or baseline.get("method") != "rcm_patch":
        raise ValueError("RCM-patch config must select baseline.method=rcm_patch")
    parameters = baseline.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("RCM-patch baseline.parameters must be an object")
    return {"method": "rcm_patch", "method_kind": "position_baseline", **copy.deepcopy(dict(parameters))}


def select(
    rows: Sequence[Mapping[str, Any]],
    *,
    positive_count: int,
    negative_count: int,
    parent_layers_per_role: int = 4,
    score_fields: Sequence[str] = ("selection_score", "mean_score_delta", "effect"),
) -> SelectionResult:
    return select_rcm(rows, method="rcm_patch", positive_count=positive_count, negative_count=negative_count, parent_layers_per_role=parent_layers_per_role, score_fields=score_fields)


def select_positions(request_or_rows: Any, **kwargs: Any) -> Any:
    if hasattr(request_or_rows, "selector_data_manifest"):
        parameters = plan_parameters(request_or_rows.method_plan)
        search = parameters.get("position_search", parameters)
        rows = load_rows(request_or_rows.selector_data_manifest, root=getattr(request_or_rows, "evidence_root", None))
        quotas = search.get("quotas", search.get("signed_quota", {}))
        positive = int(search.get("positive_count", quotas.get("target_support", 4))) if isinstance(quotas, Mapping) else int(search.get("positive_count", 4))
        negative = int(search.get("negative_count", quotas.get("competitor_support", 4))) if isinstance(quotas, Mapping) else int(search.get("negative_count", 4))
        result = select(rows, positive_count=positive, negative_count=negative, parent_layers_per_role=int(search.get("parent_layers_per_role", search.get("coarse_beam_width", 4))), score_fields=tuple(search.get("score_fields", ("selection_score", "mean_score_delta", "effect"))))
        return run_request(request_or_rows, result)
    parameters = kwargs or {}
    return list(select(request_or_rows, positive_count=int(parameters["positive_count"]), negative_count=int(parameters["negative_count"]), parent_layers_per_role=int(parameters.get("parent_layers_per_role", 4)), score_fields=tuple(parameters.get("score_fields", ("selection_score", "mean_score_delta", "effect")))).positions)
