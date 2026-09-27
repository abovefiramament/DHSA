from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SOURCE_CONTEXT_OVER_PRIOR = "source_context_over_prior"


@dataclass(frozen=True, slots=True)
class EdgeClause:
    name: str
    axis: str = ""
    event: str = ""
    edge_sign: str = ""
    edge_direction: str = ""
    max_core_index: int | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EdgeClause":
        return cls(
            name=str(raw["name"]),
            axis=str(raw.get("axis", "")),
            event=str(raw.get("event", raw.get("micro_event", ""))),
            edge_sign=str(raw.get("edge_sign", raw.get("role", ""))),
            edge_direction=str(raw.get("edge_direction", raw.get("direction", ""))),
            max_core_index=(
                None
                if raw.get("max_core_index") in (None, "")
                else int(raw.get("max_core_index"))
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "axis": self.axis,
            "event": self.event,
            "edge_sign": self.edge_sign,
            "edge_direction": self.edge_direction,
            "max_core_index": self.max_core_index,
        }


@dataclass(frozen=True, slots=True)
class ObjectiveSpec:
    objective_id: str
    prompt_key: str
    y_plus_keys: tuple[str, ...]
    y_minus_keys: tuple[str, ...]
    y_minus_fallback_keys: tuple[str, ...] = field(default_factory=tuple)
    component_clauses: tuple[EdgeClause, ...] = field(default_factory=tuple)
    continuation_prefix: str = " "
    val_mod: int = 5
    description: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ObjectiveSpec":
        clauses = tuple(EdgeClause.from_dict(item) for item in raw.get("component_clauses", []))
        return cls(
            objective_id=str(raw.get("objective_id", raw.get("event", ""))),
            prompt_key=str(raw.get("prompt_key", "base_rag")),
            y_plus_keys=tuple(str(item) for item in raw.get("y_plus_keys", [])),
            y_minus_keys=tuple(str(item) for item in raw.get("y_minus_keys", [])),
            y_minus_fallback_keys=tuple(str(item) for item in raw.get("y_minus_fallback_keys", [])),
            component_clauses=clauses,
            continuation_prefix=str(raw.get("continuation_prefix", " ")),
            val_mod=int(raw.get("val_mod", 5)),
            description=str(raw.get("description", "")),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "objective_id": self.objective_id,
            "prompt_key": self.prompt_key,
            "y_plus_keys": list(self.y_plus_keys),
            "y_minus_keys": list(self.y_minus_keys),
            "y_minus_fallback_keys": list(self.y_minus_fallback_keys),
            "component_clauses": [clause.to_dict() for clause in self.component_clauses],
            "continuation_prefix": self.continuation_prefix,
            "val_mod": self.val_mod,
            "description": self.description,
        }


def builtin_objective(objective_id: str) -> ObjectiveSpec:
    if objective_id != SOURCE_CONTEXT_OVER_PRIOR:
        raise ValueError(
            f"Unknown built-in CECM objective: {objective_id}. "
            "Pass --spec-json for a new axis/event without changing the algorithm."
        )
    return ObjectiveSpec(
        objective_id=SOURCE_CONTEXT_OVER_PRIOR,
        prompt_key="base_rag",
        y_plus_keys=(
            "cf_answer",
            "context_answer",
            "counterfactual_answer",
            "target_answer",
            "evidence_answer",
            "cf_answers",
            "context_answers",
            "counterfactual_answers",
            "target_answers",
            "evidence_answers",
        ),
        y_minus_keys=(
            "orig_answer",
            "prior_answer",
            "parametric_answer",
            "source_answer",
            "baseline_answer",
            "orig_answers",
            "prior_answers",
            "parametric_answers",
            "source_answers",
            "baseline_answers",
        ),
        component_clauses=(
            EdgeClause(
                name="context_support_edge",
                axis="source_identity",
                event="context_hit",
                edge_sign="need_positive",
                edge_direction="positive",
                max_core_index=5,
            ),
            EdgeClause(
                name="prior_support_edge",
                axis="source_identity",
                event="prior_hit",
                edge_sign="suff_positive",
                edge_direction="positive",
                max_core_index=5,
            ),
        ),
        description=(
            "Preference objective whose positive endpoint is the context-supported answer "
            "and whose negative endpoint is the prior/original answer."
        ),
    )


def load_objective_spec(
    *,
    objective_id: str,
    spec_json: Path | str | None = None,
) -> ObjectiveSpec:
    if spec_json is None:
        return builtin_objective(objective_id)
    spec_path = Path(spec_json)
    raw = json.loads(spec_path.read_text(encoding="utf-8"))
    spec = ObjectiveSpec.from_dict(raw)
    if objective_id and spec.objective_id and objective_id != spec.objective_id:
        raise ValueError(f"Objective mismatch: CLI requested {objective_id}, spec has {spec.objective_id}")
    if not spec.objective_id:
        raise ValueError("objective spec must define objective_id")
    if not spec.y_plus_keys or not spec.y_minus_keys:
        raise ValueError("objective spec must define non-empty y_plus_keys and y_minus_keys")
    return spec
