"""Random position selector using explicit seed and no replacement."""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

from .position_selection import SelectionResult, load_rows, plan_parameters, run_request, select_random


def resolve_plan(config: Mapping[str, Any]) -> dict[str, Any]:
    baseline = config.get("baseline") if isinstance(config, Mapping) else None
    if not isinstance(baseline, Mapping) or baseline.get("method") != "random":
        raise ValueError("Random config must select baseline.method=random")
    parameters = baseline.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("Random baseline.parameters must be an object")
    return {"method": "random", "method_kind": "position_baseline", **copy.deepcopy(dict(parameters))}


def select(rows: Sequence[Mapping[str, Any]], *, count: int, seed: int) -> SelectionResult:
    return select_random(rows, count=count, seed=seed)


def select_positions(request_or_rows: Any, **kwargs: Any) -> Any:
    if hasattr(request_or_rows, "selector_data_manifest"):
        parameters = plan_parameters(request_or_rows.method_plan)
        search = parameters.get("position_search", parameters)
        rows = load_rows(request_or_rows.selector_data_manifest, root=getattr(request_or_rows, "evidence_root", None))
        if "seed" not in search:
            raise ValueError("Random position selection requires an explicit seed")
        result = select(rows, count=int(search.get("count", search.get("top_k", 8))), seed=int(search["seed"]))
        return run_request(request_or_rows, result)
    parameters = kwargs or {}
    return list(select(request_or_rows, count=int(parameters["count"]), seed=int(parameters["seed"])).positions)
