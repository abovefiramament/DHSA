"""Load every code-owned component registration exactly once per process."""

from __future__ import annotations

from baseline.implementations.component_bindings import register_baseline_components
from data.adapter_bindings import register_data_components

from .builtin_components import register_builtin_components
from .component_registry import COMPONENT_REGISTRY, ComponentRegistry


def load_component_catalog(
    registry: ComponentRegistry = COMPONENT_REGISTRY,
) -> ComponentRegistry:
    """Load framework and mature-method bindings without reading experiment config."""

    register_builtin_components(registry)
    register_baseline_components(registry)
    register_data_components(registry)
    return registry
