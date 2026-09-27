from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class EventSpec:
    """A measurable generation event used by RCM."""

    name: str
    axis: str
    target: str
    contrast: str
    description: str
    positive_field: str
    positive_values: tuple[Any, ...] = (True,)
    score_fields: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def read(self, row: dict[str, Any]) -> bool:
        value = row.get(self.positive_field)
        if isinstance(value, str):
            return value in {str(item) for item in self.positive_values}
        return value in self.positive_values


@dataclass(frozen=True, slots=True)
class AxisSpec:
    """A factor axis composed of target and contrast behavioral attractors."""

    name: str
    target_event: str
    contrast_event: str
    target_attractor: str
    contrast_attractor: str
    default_timing: str
    default_alpha: float
    selection_metric: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_event_specs() -> dict[str, EventSpec]:
    return {
        "context_adoption": EventSpec(
            name="context_adoption",
            axis="source_identity",
            target="context_answer",
            contrast="prior_answer",
            description="The generation adopts the context/counterfactual answer only.",
            positive_field="outcome",
            positive_values=("context_only",),
            score_fields=("ci_score_context", "margin_score_context", "source_score"),
        ),
        "prior_leakage": EventSpec(
            name="prior_leakage",
            axis="source_identity",
            target="prior_answer",
            contrast="context_answer",
            description="The generation leaks or adopts the parametric/prior answer.",
            positive_field="orig_hit",
            positive_values=(True,),
            score_fields=("ci_score_prior", "margin_score_prior"),
        ),
        "short_exact_form": EventSpec(
            name="short_exact_form",
            axis="form",
            target="short_exact_answer",
            contrast="verbose_or_freeform_answer",
            description="The generation gives a concise first answer matching the context answer.",
            positive_field="first_answer_cf_em",
            positive_values=(True,),
            score_fields=("form_score",),
        ),
        "single_commitment": EventSpec(
            name="single_commitment",
            axis="commitment",
            target="single_committed_answer",
            contrast="mixed_or_hedged_answer",
            description="The generation gives one committed answer rather than both/mixed/neither.",
            positive_field="outcome",
            positive_values=("context_only", "prior_only"),
            score_fields=("commitment_score",),
        ),
        "prior_suppression": EventSpec(
            name="prior_suppression",
            axis="prior_suppression",
            target="no_prior_memory",
            contrast="prior_memory",
            description="The generation avoids relying on the prior-memory answer.",
            positive_field="orig_hit",
            positive_values=(False,),
            score_fields=("prior_suppression_score",),
        ),
    }


def default_axis_specs() -> dict[str, AxisSpec]:
    return {
        "source_identity": AxisSpec(
            name="source_identity",
            target_event="context_adoption",
            contrast_event="prior_leakage",
            target_attractor="context_answer",
            contrast_attractor="prior_answer",
            default_timing="all",
            default_alpha=1.0,
            selection_metric="source_score",
            description="Which source identity the model adopts under context/prior conflict.",
        ),
        "form": AxisSpec(
            name="form",
            target_event="short_exact_form",
            contrast_event="verbose_or_freeform_answer",
            target_attractor="short_exact_answer",
            contrast_attractor="verbose_or_freeform_answer",
            default_timing="prefill",
            default_alpha=0.5,
            selection_metric="form_score",
            description="Whether the answer is concise/exact or verbose/free-form.",
        ),
        "commitment": AxisSpec(
            name="commitment",
            target_event="single_commitment",
            contrast_event="mixed_or_hedged_answer",
            target_attractor="single_committed_answer",
            contrast_attractor="mixed_or_hedged_answer",
            default_timing="first_2_decode",
            default_alpha=0.75,
            selection_metric="commitment_score",
            description="Whether the model commits to one answer or hedges/mentions alternatives.",
        ),
        "prior_suppression": AxisSpec(
            name="prior_suppression",
            target_event="prior_suppression",
            contrast_event="prior_memory",
            target_attractor="no_prior_memory",
            contrast_attractor="prior_memory",
            default_timing="all",
            default_alpha=0.5,
            selection_metric="prior_suppression_score",
            description="Whether prior-memory components are suppressed during context following.",
        ),
    }


def get_event_spec(name: str) -> EventSpec:
    specs = default_event_specs()
    if name not in specs:
        raise KeyError(f"Unknown event spec: {name}")
    return specs[name]


def get_axis_spec(name: str) -> AxisSpec:
    specs = default_axis_specs()
    if name not in specs:
        raise KeyError(f"Unknown axis spec: {name}")
    return specs[name]
