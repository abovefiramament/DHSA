"""Registration of controller-native components, separate from methods."""

from __future__ import annotations

from .component_registry import (
    COMPONENT_REGISTRY,
    ComponentRegistry,
    ComponentRegistryError,
)
from .evaluation_controller import DeterministicMetricAlphaSelector


def register_builtin_components(
    registry: ComponentRegistry = COMPONENT_REGISTRY,
) -> None:
    """Idempotently register implementations owned by the shared framework."""

    try:
        registry.register(
            kind="alpha_selector",
            component_id="deterministic_metric",
            version=1,
            implementation=DeterministicMetricAlphaSelector(),
            operations=("select",),
            input_contract="AlphaSelectionRequest/v1",
            output_contract="AlphaDecision/v1",
        )
    except ComponentRegistryError as exc:
        if "duplicate component registration" not in str(exc):
            raise
