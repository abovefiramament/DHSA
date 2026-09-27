"""Bind reusable CAST runtime components to a supplied model provider."""

from __future__ import annotations

from typing import Any

from experiments.shared.component_catalog import load_component_catalog
from experiments.shared.component_registry import COMPONENT_REGISTRY, ComponentRegistry

from .component_bindings import (
    build_attention_head_scanners,
    register_execution_components,
)
from .model_runtime import (
    cast_generation_callback_factory,
    iti_capture_pair,
    iti_train_probes,
    rcm_measure,
    rcm_measure_many,
    rcm_prepare_patch_prototypes,
)


def register_cast_runtime_components(
    model_provider: Any,
    *,
    registry: ComponentRegistry = COMPONENT_REGISTRY,
) -> dict[str, Any]:
    """Register CAST and position-scanner capabilities without task bindings.

    The caller supplies the model provider. Dataset adapters and evaluators are
    registered elsewhere, so the same CAST runtime can be used by Site and
    Performance without a second implementation.
    """

    load_component_catalog(registry)
    scanners = build_attention_head_scanners(
        model_provider=model_provider,
        layer_count=lambda model: model.layers,
        head_count=lambda model, _layer: model.attention_heads_per_layer,
        rcm_measure=rcm_measure,
        rcm_measure_many=rcm_measure_many,
        rcm_prepare_patch_prototypes=rcm_prepare_patch_prototypes,
        iti_capture_pair=iti_capture_pair,
        iti_train_probes=iti_train_probes,
    )
    return register_execution_components(
        model_provider=model_provider,
        generation_callback=cast_generation_callback_factory(model_provider),
        bank_component_id="cast",
        generation_component_id="cast_generation",
        position_scanners=scanners,
        registry=registry,
    )
