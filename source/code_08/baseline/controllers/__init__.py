"""Configuration controllers shared by all baseline methods."""

from .bank_controller import build_bank_plan, build_bank_plan_from_position_freeze
from .bank_execution_controller import (
    compose_audited_banks,
    execute_bank_plan,
    execute_registered_bank_plan,
    finalize_audited_bank_plan,
)
from .parameter_controller import resolve_method_parameters
from .control_objective_controller import resolve_control_objective
from .payload_controller import (
    compose_controller_manifest,
    register_component_payload,
    register_position_vector_pair,
    validate_component_payload,
    validate_position_vector_pair,
)
from .post_training_audit_controller import (
    build_post_training_audit_plan,
    freeze_post_training_selection,
)
from .position_controller import resolve_positions
from .position_freeze_controller import freeze_position_plan, validate_position_freeze

__all__ = [
    "build_bank_plan",
    "build_bank_plan_from_position_freeze",
    "build_post_training_audit_plan",
    "compose_controller_manifest",
    "compose_audited_banks",
    "execute_bank_plan",
    "execute_registered_bank_plan",
    "finalize_audited_bank_plan",
    "freeze_post_training_selection",
    "register_component_payload",
    "register_position_vector_pair",
    "resolve_method_parameters",
    "resolve_control_objective",
    "resolve_positions",
    "freeze_position_plan",
    "validate_position_freeze",
    "validate_component_payload",
    "validate_position_vector_pair",
]
