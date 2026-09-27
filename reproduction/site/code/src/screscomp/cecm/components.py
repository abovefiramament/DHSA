from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from screscomp.cecm.specs import EdgeClause, ObjectiveSpec


SOURCE_CONTEXT_OVER_PRIOR = "source_context_over_prior"


@dataclass(frozen=True, slots=True)
class ComponentSpec:
    component_id: str
    layer_idx: int
    component_type: str


@dataclass(frozen=True, slots=True)
class CoreSelectionConfig:
    mode: str = SOURCE_CONTEXT_OVER_PRIOR
    max_core_index: int = 5
    require_axis: str = "source_identity"
    context_event: str = "context_hit"
    prior_event: str = "prior_hit"
    context_edge_sign: str = "need_positive"
    prior_edge_sign: str = "suff_positive"
    edge_direction: str = "positive"


@dataclass(frozen=True, slots=True)
class SelectedComponent:
    component_id: str
    layer_idx: int
    component_type: str
    context_core_index: int
    prior_core_index: int
    context_role: str
    prior_role: str
    context_weight: float
    prior_weight: float
    rank: int

    def to_row(self) -> dict[str, object]:
        return {
            "component_id": self.component_id,
            "layer_idx": self.layer_idx,
            "component_type": self.component_type,
            "context_core_index": self.context_core_index,
            "prior_core_index": self.prior_core_index,
            "context_role": self.context_role,
            "prior_role": self.prior_role,
            "context_weight": self.context_weight,
            "prior_weight": self.prior_weight,
            "rank": self.rank,
            "mode": SOURCE_CONTEXT_OVER_PRIOR,
            "selection_rule": (
                "intersection(context_hit need_positive positive, "
                "prior_hit suff_positive positive)"
            ),
        }


@dataclass(frozen=True, slots=True)
class GenericSelectedComponent:
    component_id: str
    layer_idx: int
    component_type: str
    rank: int
    objective_id: str
    clause_rows: tuple[tuple[EdgeClause, dict[str, Any]], ...]

    def to_row(self) -> dict[str, object]:
        row: dict[str, object] = {
            "component_id": self.component_id,
            "layer_idx": self.layer_idx,
            "component_type": self.component_type,
            "rank": self.rank,
            "objective_id": self.objective_id,
            "num_required_clauses": len(self.clause_rows),
            "selection_rule": "component must satisfy every objective component_clause",
        }
        for clause, edge_row in self.clause_rows:
            prefix = f"clause_{clause.name}"
            row[f"{prefix}_event"] = _event(edge_row)
            row[f"{prefix}_axis"] = _axis(edge_row)
            row[f"{prefix}_edge_sign"] = _edge_sign(edge_row)
            row[f"{prefix}_edge_direction"] = _edge_direction(edge_row)
            row[f"{prefix}_core_index"] = _core_index(edge_row)
            row[f"{prefix}_weight"] = _edge_weight(edge_row)
        return row


def _get(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _int_or_large(value: Any) -> int:
    try:
        text = str(value).strip()
        if not text:
            return 10**9
        return int(float(text))
    except Exception:
        return 10**9


def _float_or_zero(value: Any) -> float:
    try:
        text = str(value).strip()
        if not text:
            return 0.0
        return float(text)
    except Exception:
        return 0.0


def component_id_from_row(row: dict[str, Any]) -> str:
    component_id = _get(row, "component_id", "component", "component_name", "node")
    if component_id:
        return component_id
    layer = _get(row, "layer_idx", "layer", "layer_index")
    kind = _get(row, "component_type", "kind", "module_type")
    if layer and kind:
        return f"L{layer}.{kind}"
    return ""


def parse_component_id(component_id: str) -> ComponentSpec:
    match = re.fullmatch(r"L(\d+)\.(attn|mlp)", component_id.strip())
    if not match:
        raise ValueError(f"Unsupported component_id: {component_id!r}; expected L<layer>.attn or L<layer>.mlp")
    return ComponentSpec(
        component_id=component_id.strip(),
        layer_idx=int(match.group(1)),
        component_type=match.group(2),
    )


def component_spec_from_row(row: dict[str, Any]) -> ComponentSpec:
    component_id = component_id_from_row(row)
    if component_id:
        parsed = parse_component_id(component_id)
        layer_raw = _get(row, "layer_idx", "layer", "layer_index")
        type_raw = _get(row, "component_type", "kind", "module_type")
        if layer_raw or type_raw:
            return ComponentSpec(
                component_id=parsed.component_id,
                layer_idx=int(layer_raw) if layer_raw else parsed.layer_idx,
                component_type=type_raw if type_raw else parsed.component_type,
            )
        return parsed
    raise ValueError(f"Could not infer component from row: {row}")


def _axis(row: dict[str, Any]) -> str:
    return _get(row, "axis", "axis_name", "event_axis")


def _event(row: dict[str, Any]) -> str:
    return _get(row, "micro_event", "event", "event_variable", "variable")


def _edge_sign(row: dict[str, Any]) -> str:
    return _get(row, "edge_sign", "role", "edge_role", "sign_role")


def _edge_direction(row: dict[str, Any]) -> str:
    return _get(row, "edge_direction", "direction", "sign")


def _core_index(row: dict[str, Any]) -> int:
    return _int_or_large(_get(row, "core_index", "rank", "component_rank"))


def _edge_weight(row: dict[str, Any]) -> float:
    return _float_or_zero(_get(row, "weight", "edge_weight", "abs_weight", "mean_abs_delta"))


def _matches_source_edge(
    row: dict[str, Any],
    *,
    config: CoreSelectionConfig,
    event: str,
    edge_sign: str,
) -> bool:
    axis = _axis(row)
    if axis and axis != config.require_axis:
        return False
    if _event(row) != event:
        return False
    if _edge_sign(row) != edge_sign:
        return False
    if config.edge_direction and _edge_direction(row) != config.edge_direction:
        return False
    if _core_index(row) > config.max_core_index:
        return False
    return bool(component_id_from_row(row))


def _matches_clause(row: dict[str, Any], clause: EdgeClause) -> bool:
    if clause.axis and _axis(row) != clause.axis:
        return False
    if clause.event and _event(row) != clause.event:
        return False
    if clause.edge_sign and _edge_sign(row) != clause.edge_sign:
        return False
    if clause.edge_direction and _edge_direction(row) != clause.edge_direction:
        return False
    if clause.max_core_index is not None and _core_index(row) > clause.max_core_index:
        return False
    return bool(component_id_from_row(row))


def select_components_by_clauses(
    rows: Iterable[dict[str, Any]],
    *,
    spec: ObjectiveSpec,
) -> tuple[list[GenericSelectedComponent], list[dict[str, object]]]:
    if not spec.component_clauses:
        raise ValueError("objective spec must define component_clauses for component selection")

    rows_list = list(rows)
    matches_by_clause: dict[str, dict[str, dict[str, Any]]] = {
        clause.name: {} for clause in spec.component_clauses
    }
    audit_edges: list[dict[str, object]] = []
    for row in rows_list:
        component_id = component_id_from_row(row)
        if not component_id:
            continue
        for clause in spec.component_clauses:
            if not _matches_clause(row, clause):
                continue
            matches_by_clause[clause.name][component_id] = row
            audit_edges.append({"clause_name": clause.name, "objective_id": spec.objective_id, **row})

    shared_ids: set[str] | None = None
    for clause in spec.component_clauses:
        ids = set(matches_by_clause[clause.name])
        shared_ids = ids if shared_ids is None else shared_ids & ids
    shared_ids = shared_ids or set()

    def rank_key(component_id: str) -> tuple[int, int, str]:
        clause_rows = [matches_by_clause[clause.name][component_id] for clause in spec.component_clauses]
        core_indices = [_core_index(row) for row in clause_rows]
        return max(core_indices), sum(core_indices), component_id

    selected: list[GenericSelectedComponent] = []
    for rank, component_id in enumerate(sorted(shared_ids, key=rank_key), start=1):
        first_row = matches_by_clause[spec.component_clauses[0].name][component_id]
        component = component_spec_from_row(first_row)
        clause_rows = tuple(
            (clause, matches_by_clause[clause.name][component_id])
            for clause in spec.component_clauses
        )
        selected.append(
            GenericSelectedComponent(
                component_id=component.component_id,
                layer_idx=component.layer_idx,
                component_type=component.component_type,
                rank=rank,
                objective_id=spec.objective_id,
                clause_rows=clause_rows,
            )
        )
    return selected, audit_edges


def summarize_generic_component_selection(
    *,
    total_edges: int,
    selected: list[GenericSelectedComponent],
    audit_edges: list[dict[str, object]],
    spec: ObjectiveSpec,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        {"metric": "objective_id", "value": spec.objective_id},
        {"metric": "total_manifest_edges", "value": total_edges},
        {"metric": "selected_components", "value": len(selected)},
        {"metric": "required_clauses", "value": len(spec.component_clauses)},
    ]
    for clause in spec.component_clauses:
        count = sum(1 for row in audit_edges if row.get("clause_name") == clause.name)
        rows.append({"metric": f"matching_edges_{clause.name}", "value": count})
    return rows


def select_source_context_arbitration_core(
    rows: Iterable[dict[str, Any]],
    *,
    config: CoreSelectionConfig | None = None,
) -> tuple[list[SelectedComponent], list[dict[str, object]]]:
    """Compatibility wrapper for the built-in source objective.

    New code should call select_components_by_clauses with an ObjectiveSpec so every
    axis uses the same component-selection algorithm.
    """
    config = config or CoreSelectionConfig()
    if config.mode != SOURCE_CONTEXT_OVER_PRIOR:
        raise ValueError(f"Unsupported component selection mode: {config.mode}")

    context_rows: dict[str, dict[str, Any]] = {}
    prior_rows: dict[str, dict[str, Any]] = {}
    audit_edges: list[dict[str, object]] = []

    for row in rows:
        component_id = component_id_from_row(row)
        if not component_id:
            continue
        if _matches_source_edge(
            row,
            config=config,
            event=config.context_event,
            edge_sign=config.context_edge_sign,
        ):
            context_rows[component_id] = row
            audit_edges.append({"edge_group": "context", **row})
        if _matches_source_edge(
            row,
            config=config,
            event=config.prior_event,
            edge_sign=config.prior_edge_sign,
        ):
            prior_rows[component_id] = row
            audit_edges.append({"edge_group": "prior", **row})

    shared_ids = sorted(
        set(context_rows) & set(prior_rows),
        key=lambda cid: (
            max(_core_index(context_rows[cid]), _core_index(prior_rows[cid])),
            _core_index(context_rows[cid]) + _core_index(prior_rows[cid]),
            cid,
        ),
    )
    selected: list[SelectedComponent] = []
    for rank, component_id in enumerate(shared_ids, start=1):
        context_row = context_rows[component_id]
        prior_row = prior_rows[component_id]
        spec = component_spec_from_row(context_row)
        selected.append(
            SelectedComponent(
                component_id=spec.component_id,
                layer_idx=spec.layer_idx,
                component_type=spec.component_type,
                context_core_index=_core_index(context_row),
                prior_core_index=_core_index(prior_row),
                context_role=_edge_sign(context_row),
                prior_role=_edge_sign(prior_row),
                context_weight=_edge_weight(context_row),
                prior_weight=_edge_weight(prior_row),
                rank=rank,
            )
        )
    return selected, audit_edges


def summarize_component_selection(
    *,
    total_edges: int,
    selected: list[SelectedComponent],
    audit_edges: list[dict[str, object]],
    config: CoreSelectionConfig,
) -> list[dict[str, object]]:
    context_count = sum(1 for row in audit_edges if row.get("edge_group") == "context")
    prior_count = sum(1 for row in audit_edges if row.get("edge_group") == "prior")
    return [
        {"metric": "total_manifest_edges", "value": total_edges},
        {"metric": "matching_context_edges", "value": context_count},
        {"metric": "matching_prior_edges", "value": prior_count},
        {"metric": "selected_shared_components", "value": len(selected)},
        {"metric": "max_core_index", "value": config.max_core_index},
        {"metric": "selection_mode", "value": config.mode},
    ]
