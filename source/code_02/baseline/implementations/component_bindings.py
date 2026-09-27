"""Register reusable baseline implementations without machine bindings.

Selectors are code-owned and registered eagerly. Model-dependent execution
components are bound only after a caller provides a model provider. This module
never owns a checkpoint, dataset, evaluator, device, or machine path.
"""

from __future__ import annotations

from typing import Any, Mapping

from experiments.shared.component_registry import ComponentRegistry
from . import caa, iti, random as random_selector, rcm_patch, rcm_zero
from .scanners.imported_rcm import ExternalRCMImportScanner


class _PositionBackend:
    def __init__(self, selector):
        self.select_positions = selector


def register_baseline_components(registry: ComponentRegistry) -> None:
    """Register reusable model-free position backends exactly once."""

    entries = {
        "rcm_zero": rcm_zero.select_positions,
        "rcm_patch": rcm_patch.select_positions,
        "iti": iti.select_positions,
        "random": random_selector.select_positions,
        "caa": caa.select_positions,
    }
    for component_id, selector in entries.items():
        try:
            registry.register(
                kind="position_backend",
                component_id=component_id,
                version=1,
                implementation=_PositionBackend(selector),
                operations=("select_positions",),
                input_contract="position_rows_and_selection_parameters/v1",
                output_contract="selected_positions_and_execution_artifacts/v1",
            )
        except ValueError as exc:
            if "duplicate component registration" not in str(exc):
                raise
    try:
        registry.register(
            kind="position_scanner",
            component_id="external_rcm_import",
            version=1,
            implementation=ExternalRCMImportScanner(),
            operations=("scan",),
            input_contract="RCMScanRequest/v1+ExternalRCMImportManifest/v1",
            output_contract="PositionScanResult/v1",
        )
    except ValueError as exc:
        if "duplicate component registration" not in str(exc):
            raise


def build_attention_head_scanners(
    *,
    model_provider: Any,
    layer_count: Any,
    head_count: Any,
    rcm_measure: Any,
    rcm_measure_many: Any,
    rcm_prepare_patch_prototypes: Any,
    iti_capture_pair: Any,
    iti_train_probes: Any,
) -> dict[str, Any]:
    """Build four attention-head scanners from execution-environment adapters.

    The returned objects contain no dataset split, checkpoint path, device, or
    scientific selection rule. Those values remain in an experiment bundle;
    these adapters expose only model geometry, measurement, activation capture,
    and the probe trainer.
    """

    from .scanners.iti import GroupedTwoFoldITIScanner
    from .scanners.random import ModelGeometryScanner
    from .scanners.rcm import CoarseToFineRCMScanner

    geometry = ModelGeometryScanner(
        model_provider=model_provider,
        layer_count=layer_count,
        head_count=head_count,
    )
    return {
        "rcm_zero_scanner": CoarseToFineRCMScanner(
            model_provider=model_provider,
            layer_count=layer_count,
            head_count=head_count,
            measure=rcm_measure,
            measure_many=rcm_measure_many,
        ),
        "rcm_patch_scanner": CoarseToFineRCMScanner(
            model_provider=model_provider,
            layer_count=layer_count,
            head_count=head_count,
            measure=rcm_measure,
            measure_many=rcm_measure_many,
            prepare_patch_prototypes=rcm_prepare_patch_prototypes,
        ),
        "iti_scanner": GroupedTwoFoldITIScanner(
            model_provider=model_provider,
            capture_pair=iti_capture_pair,
            train_probes=iti_train_probes,
        ),
        "random_scanner": geometry,
    }


def build_caa_scanner(
    *,
    model_provider: Any,
    layer_count: Any,
    capture_pair: Any,
    measure_many: Any,
) -> Any:
    from .scanners.caa import FullDepthCAAScanner

    return FullDepthCAAScanner(
        model_provider=model_provider,
        layer_count=layer_count,
        capture_pair=capture_pair,
        measure_many=measure_many,
    )


def register_execution_components(
    *,
    model_provider: Any,
    generation_callback: Any = None,
    bank_component_id: str = "cast",
    generation_component_id: str = "cast_generation",
    position_scanners: Any = None,
    registry: ComponentRegistry,
) -> dict[str, Any]:
    """Bind CAST/generation implementations to external callables.

    This is a process-local binding point. It deliberately is not called by the
    repository catalog because the required model and callbacks belong to the
    execution environment, not to a method or public configuration file.
    """

    from .cast import CastBackend
    from .generation_backend import CallbackGenerationBackend

    if model_provider is None or not callable(getattr(model_provider, "load", None)):
        raise ValueError("register_execution_components requires a model provider")

    def ensure(kind: str, component_id: str, implementation: Any, operations: tuple[str, ...], input_contract: str, output_contract: str) -> Any:
        try:
            return registry.register(
                kind=kind,
                component_id=component_id,
                version=1,
                implementation=implementation,
                operations=operations,
                input_contract=input_contract,
                output_contract=output_contract,
            )
        except ValueError as exc:
            if "duplicate component registration" not in str(exc):
                raise
            return registry.resolve(
                {"id": component_id, "version": 1},
                expected_kind=kind,
            )

    registrations: dict[str, Any] = {
        "cast": ensure(
            "bank_backend",
            bank_component_id,
            CastBackend(model_provider),
            ("train_bank", "compose_bank"),
            "BankTrainRequest+BankComposeRequest/v1",
            "BankTrainResult/v1",
        )
    }
    if position_scanners is not None:
        if not isinstance(position_scanners, Mapping):
            raise ValueError("position_scanners must map registered IDs to scanner objects")
        scanner_registrations: dict[str, Any] = {}
        for component_id, scanner in position_scanners.items():
            if not isinstance(component_id, str) or not component_id:
                raise ValueError("position scanner IDs must be non-empty strings")
            scanner_registrations[component_id] = ensure(
                "position_scanner",
                component_id,
                scanner,
                ("scan",),
                "RCMScanRequest|ITIScanRequest/v1",
                "PositionScanResult/v1",
            )
        registrations["position_scanners"] = scanner_registrations
    if generation_callback is not None:
        registrations["generation"] = ensure(
            "generation_backend",
            generation_component_id,
            CallbackGenerationBackend(
                model_provider=model_provider,
                generate=generation_callback,
            ),
            ("generate",),
            "GenerationRequest/v1",
            "GenerationResult/v1",
        )
    return registrations


def register_loreft_components(
    *,
    model_provider: Any,
    registry: ComponentRegistry,
) -> dict[str, Any]:
    """Bind official LoReFT training and inference to the supplied model provider."""

    from .loreft import LoReFTBackend, loreft_generation_backend

    if model_provider is None or not callable(getattr(model_provider, "load_fresh", None)):
        raise ValueError("register_loreft_components requires load_fresh()")

    def ensure(kind: str, component_id: str, implementation: Any, operations: tuple[str, ...]):
        try:
            return registry.register(
                kind=kind,
                component_id=component_id,
                version=1,
                implementation=implementation,
                operations=operations,
                input_contract="native_loreft/v1",
                output_contract="native_loreft_evidence/v1",
            )
        except ValueError as exc:
            if "duplicate component registration" not in str(exc):
                raise
            return registry.resolve(
                {"id": component_id, "version": 1}, expected_kind=kind
            )

    return {
        "training": ensure(
            "native_baseline_backend",
            "loreft",
            LoReFTBackend(model_provider),
            ("train_candidate",),
        ),
        "generation": ensure(
            "generation_backend",
            "loreft_generation",
            loreft_generation_backend(model_provider),
            ("generate",),
        ),
    }


def register_bipo_components(
    *,
    model_provider: Any,
    caa_scanner: Any,
    registry: ComponentRegistry,
) -> dict[str, Any]:
    """Bind CAA localization and the official BiPO trainer."""

    from .bipo import BiPOBackend, bipo_generation_backend

    def ensure(kind: str, component_id: str, implementation: Any, operations: tuple[str, ...]):
        try:
            return registry.register(
                kind=kind,
                component_id=component_id,
                version=1,
                implementation=implementation,
                operations=operations,
                input_contract="native_bipo/v1",
                output_contract="native_bipo_evidence/v1",
            )
        except ValueError as exc:
            if "duplicate component registration" not in str(exc):
                raise
            return registry.resolve(
                {"id": component_id, "version": 1}, expected_kind=kind
            )

    return {
        "scanner": ensure("position_scanner", "caa_scanner", caa_scanner, ("scan",)),
        "training": ensure(
            "native_baseline_backend", "bipo", BiPOBackend(model_provider), ("train_candidate",)
        ),
        "generation": ensure(
            "generation_backend",
            "bipo_generation",
            bipo_generation_backend(model_provider),
            ("generate",),
        ),
    }
