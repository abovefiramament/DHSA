"""Shared semantic contract for RCM state contrasts and CAST control.

This module contains no model code and no scientific defaults.  It only
normalizes the common target-vs-non-target advantage and validates the two
method families that derive from it:

* ``state_contrast`` is the RCM family (Zero or Patch);
* ``relative_advantage`` is the CAST family (external residual control).

The returned plan records the semantic choice so a backend cannot silently
reinterpret a registered experiment.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping


class ControlObjectiveError(ValueError):
    """Raised when a control objective is missing or semantically incomplete."""


STATE_CONTRAST_VARIANTS = frozenset({"zero", "patch"})
CONTROL_FAMILIES = frozenset({"state_contrast", "relative_advantage"})


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ControlObjectiveError(f"{field} must be an object")
    return value


def _string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ControlObjectiveError(f"{field} must be a non-empty string")
    return value.strip()


def _bool(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ControlObjectiveError(f"{field} must be a boolean")
    return value


def _advantage(value: Any) -> dict[str, Any]:
    raw = _mapping(value, field="control_objective.advantage")
    required = ("metric", "target", "non_target", "aggregation", "same_input_and_decode")
    for key in required:
        if key not in raw:
            raise ControlObjectiveError(f"control_objective.advantage.{key} is required")
    resolved = {
        "metric": _string(raw["metric"], field="control_objective.advantage.metric"),
        "target": _string(raw["target"], field="control_objective.advantage.target"),
        "non_target": _string(
            raw["non_target"], field="control_objective.advantage.non_target"
        ),
        "aggregation": _string(
            raw["aggregation"], field="control_objective.advantage.aggregation"
        ),
        "same_input_and_decode": _bool(
            raw["same_input_and_decode"],
            field="control_objective.advantage.same_input_and_decode",
        ),
    }
    if resolved["aggregation"] != "mean":
        raise ControlObjectiveError(
            "control_objective.advantage.aggregation must be mean"
        )
    if not resolved["same_input_and_decode"]:
        raise ControlObjectiveError(
            "control_objective.advantage.same_input_and_decode must be true"
        )
    return resolved


def _state_contrast(raw: Mapping[str, Any], *, expected_variant: str | None) -> dict[str, Any]:
    variant = _string(raw.get("variant"), field="control_objective.variant")
    if variant not in STATE_CONTRAST_VARIANTS:
        raise ControlObjectiveError(f"unsupported state-contrast variant: {variant}")
    if expected_variant is not None and variant != expected_variant:
        raise ControlObjectiveError(
            f"control_objective.variant {variant!r} does not match method {expected_variant!r}"
        )
    contrast = _mapping(
        raw.get("contrast"), field="control_objective.contrast"
    )
    required = (
        "state_a",
        "state_b",
        "direction",
        "effect_type",
        "single_component",
        "recompute_downstream",
    )
    for key in required:
        if key not in contrast:
            raise ControlObjectiveError(f"control_objective.contrast.{key} is required")
    resolved_contrast = {
        "state_a": _string(contrast["state_a"], field="control_objective.contrast.state_a"),
        "state_b": _string(contrast["state_b"], field="control_objective.contrast.state_b"),
        "direction": _string(
            contrast["direction"], field="control_objective.contrast.direction"
        ),
        "effect_type": _string(
            contrast["effect_type"], field="control_objective.contrast.effect_type"
        ),
        "single_component": _bool(
            contrast["single_component"],
            field="control_objective.contrast.single_component",
        ),
        "recompute_downstream": _bool(
            contrast["recompute_downstream"],
            field="control_objective.contrast.recompute_downstream",
        ),
    }
    if not resolved_contrast["single_component"]:
        raise ControlObjectiveError("RCM state contrast must change one component at a time")
    if not resolved_contrast["recompute_downstream"]:
        raise ControlObjectiveError("RCM state contrast must recompute downstream computation")
    expected = {
        "zero": {
            "state_a": "zero_missing",
            "state_b": "native_present",
            "direction": "zero_to_native",
            "effect_type": "existence",
        },
        "patch": {
            "state_a": "reference",
            "state_b": "target_prototype",
            "direction": "reference_to_target",
            "effect_type": "replacement",
        },
    }[variant]
    for key, value in expected.items():
        if resolved_contrast[key] != value:
            raise ControlObjectiveError(
                f"RCM {variant} contrast.{key} must be {value!r}"
            )
    return {
        "family": "state_contrast",
        "variant": variant,
        "advantage": _advantage(raw.get("advantage")),
        "contrast": resolved_contrast,
    }


def _relative_advantage(raw: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "reference_policy",
        "controlled_policy",
        "surrogate",
        "base_frozen",
        "residual_operation",
    )
    for key in required:
        if key not in raw:
            raise ControlObjectiveError(f"control_objective.{key} is required")
    resolved = {
        "family": "relative_advantage",
        "advantage": _advantage(raw.get("advantage")),
        "reference_policy": _string(
            raw["reference_policy"], field="control_objective.reference_policy"
        ),
        "controlled_policy": _string(
            raw["controlled_policy"], field="control_objective.controlled_policy"
        ),
        "surrogate": _string(raw["surrogate"], field="control_objective.surrogate"),
        "base_frozen": _bool(raw["base_frozen"], field="control_objective.base_frozen"),
        "residual_operation": _string(
            raw["residual_operation"], field="control_objective.residual_operation"
        ),
    }
    if resolved["reference_policy"] != "frozen_base":
        raise ControlObjectiveError(
            "CAST reference_policy must be frozen_base"
        )
    if resolved["controlled_policy"] != "base_plus_external_residual":
        raise ControlObjectiveError(
            "CAST controlled_policy must be base_plus_external_residual"
        )
    if resolved["surrogate"] != "dpo_relative_log_odds":
        raise ControlObjectiveError(
            "CAST surrogate must be dpo_relative_log_odds"
        )
    if resolved["residual_operation"] != "add_external_delta_at_registered_write":
        raise ControlObjectiveError(
            "CAST residual_operation must add_external_delta_at_registered_write"
        )
    if not resolved["base_frozen"]:
        raise ControlObjectiveError("CAST base_frozen must be true")
    return resolved


def resolve_control_objective(
    value: Any,
    *,
    expected_family: str | None = None,
    expected_variant: str | None = None,
) -> dict[str, Any]:
    """Validate one explicit RCM or CAST semantic objective."""

    raw = _mapping(value, field="control_objective")
    family = _string(raw.get("family"), field="control_objective.family")
    if family not in CONTROL_FAMILIES:
        raise ControlObjectiveError(f"unsupported control objective family: {family}")
    if expected_family is not None and family != expected_family:
        raise ControlObjectiveError(
            f"control_objective.family {family!r} does not match {expected_family!r}"
        )
    if family == "state_contrast":
        if expected_variant is None and "variant" not in raw:
            raise ControlObjectiveError("state_contrast requires an explicit variant")
        resolved = _state_contrast(raw, expected_variant=expected_variant)
    else:
        if expected_variant is not None:
            raise ControlObjectiveError("relative_advantage cannot declare an RCM variant")
        resolved = _relative_advantage(raw)
    return copy.deepcopy(resolved)
