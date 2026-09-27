from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .graph import ComponentNode, RCMGraph


CONTROL_OPERATIONS = {
    "amplify_up",
    "suppress_up",
    "amplify_down",
    "suppress_down",
    "target",
    "anti_target",
}


@dataclass(slots=True)
class ControlTerm:
    axis: str
    component_id: str
    layer_idx: int
    component_type: str
    sign: str
    operation: str
    alpha: float
    timing: str
    weight: float
    expected_effect: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ComponentDecision:
    component_id: str
    layer_idx: int
    component_type: str
    axes: str
    operations: str
    signs: str
    timings: str
    term_count: int
    signed_alpha_sum: float
    relation_kind: str
    resolution_rule: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ControlPlan:
    name: str
    graph_name: str
    operations: dict[str, list[str]]
    terms: list[ControlTerm]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "graph_name": self.graph_name,
            "operations": self.operations,
            "terms": [term.to_dict() for term in self.terms],
            "metadata": self.metadata,
        }

    def addition_specs(self) -> list[dict[str, Any]]:
        """Return tensor-free addition specs used by generation wrappers."""

        return [
            {
                "axis": term.axis,
                "component_id": term.component_id,
                "layer_idx": term.layer_idx,
                "component_type": term.component_type,
                "alpha_multiplier": term.alpha,
                "apply_mode": term.timing,
                "operation": term.operation,
                "sign": term.sign,
                "expected_effect": term.expected_effect,
            }
            for term in self.terms
        ]

    def component_decision_rows(self) -> list[dict[str, Any]]:
        grouped: dict[str, list[ControlTerm]] = {}
        for term in self.terms:
            grouped.setdefault(term.component_id, []).append(term)

        rows: list[dict[str, Any]] = []
        for component_id, terms in sorted(grouped.items()):
            axes = sorted({term.axis for term in terms})
            operations = sorted({term.operation for term in terms})
            signs = sorted({term.sign for term in terms})
            timings = sorted({term.timing for term in terms})
            if len(axes) == 1:
                relation_kind = "single_axis"
                resolution_rule = "apply_axis_term"
            elif len(signs) == 1 and len(timings) == 1:
                relation_kind = "aligned_multi_axis"
                resolution_rule = "compose_axis_terms"
            elif len(signs) > 1 and len(timings) == 1:
                relation_kind = "opposite_sign_multi_axis"
                resolution_rule = "compose_but_flag_for_validation"
            else:
                relation_kind = "mixed_timing_multi_axis"
                resolution_rule = "stage_terms_by_timing_and_flag_for_validation"
            first = terms[0]
            rows.append(
                ComponentDecision(
                    component_id=component_id,
                    layer_idx=first.layer_idx,
                    component_type=first.component_type,
                    axes=",".join(axes),
                    operations=",".join(operations),
                    signs=",".join(signs),
                    timings=",".join(timings),
                    term_count=len(terms),
                    signed_alpha_sum=sum(term.alpha for term in terms),
                    relation_kind=relation_kind,
                    resolution_rule=resolution_rule,
                ).to_dict()
            )
        return rows


def _expand_operation(operation: str) -> list[str]:
    if operation == "target":
        return ["amplify_up", "suppress_down"]
    if operation == "anti_target":
        return ["suppress_up", "amplify_down"]
    if operation not in CONTROL_OPERATIONS:
        raise ValueError(f"Unknown control operation: {operation}")
    return [operation]


def _operation_applies(operation: str, node: ComponentNode) -> bool:
    return (
        (operation.endswith("_up") and node.sign == "up")
        or (operation.endswith("_down") and node.sign == "down")
    )


def _alpha_multiplier(operation: str) -> float:
    if operation.startswith("amplify"):
        return 1.0
    if operation.startswith("suppress"):
        return -1.0
    raise ValueError(f"Cannot map operation to alpha: {operation}")


def _expected_effect(operation: str) -> str:
    if operation in {"amplify_up", "suppress_down"}:
        return "raise_target_event"
    if operation in {"suppress_up", "amplify_down"}:
        return "lower_target_event"
    raise ValueError(f"Cannot map operation to effect: {operation}")


def build_control_plan(
    graph: RCMGraph,
    *,
    axis_operations: dict[str, list[str]] | None = None,
    name: str = "rcm_control_plan",
    global_alpha: float = 1.0,
) -> ControlPlan:
    if axis_operations is None:
        axis_operations = {axis: ["target"] for axis in graph.axes}

    expanded: dict[str, list[str]] = {}
    terms: list[ControlTerm] = []
    for axis, operations in axis_operations.items():
        if axis not in graph.axes:
            raise ValueError(f"Operation specified for axis not in graph: {axis}")
        expanded_ops = [op for operation in operations for op in _expand_operation(operation)]
        expanded[axis] = expanded_ops
        for node in graph.by_axis(axis):
            for operation in expanded_ops:
                if not _operation_applies(operation, node):
                    continue
                terms.append(
                    ControlTerm(
                        axis=node.axis,
                        component_id=node.component_id,
                        layer_idx=node.layer_idx,
                        component_type=node.component_type,
                        sign=node.sign,
                        operation=operation,
                        alpha=global_alpha * node.alpha * _alpha_multiplier(operation),
                        timing=node.timing,
                        weight=node.weight,
                        expected_effect=_expected_effect(operation),
                    )
                )

    return ControlPlan(
        name=name,
        graph_name=graph.name,
        operations=expanded,
        terms=terms,
        metadata={"global_alpha": global_alpha},
    )
