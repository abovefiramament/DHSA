"""Fixed function facade for dataset, position, bank, and payload components."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from baseline.controllers import (
    build_bank_plan,
    build_post_training_audit_plan,
    compose_controller_manifest,
    freeze_post_training_selection,
    register_component_payload,
    register_position_vector_pair,
    resolve_method_parameters,
    resolve_positions,
    validate_component_payload,
    validate_position_vector_pair,
)
from data.controller_registry import get_dataset_controller
from experiments.shared.component_registry import COMPONENT_REGISTRY, ComponentRegistry
from experiments.shared.contracts import DataMaterializationRequest, DataMaterializationResult


class ComponentRuntimeError(ValueError):
    """Raised when a fixed component function cannot satisfy its contract."""


def _write_exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise ComponentRuntimeError(f"component artifact already exists: {path}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(rendered)


def _load_object(path: Path, *, field: str) -> dict[str, Any]:
    if not path.is_file():
        raise ComponentRuntimeError(f"{field} does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ComponentRuntimeError(f"{field} must contain one JSON object")
    return value


def materialize_dataset(
    dataset: str,
    data_spec: Mapping[str, Any],
    *,
    output_dir: Path,
    execution_config: Mapping[str, Any] | None = None,
    evidence_root: Path | None = None,
    registry: ComponentRegistry = COMPONENT_REGISTRY,
) -> dict[str, Any]:
    """Execute one registered dataset adapter without method-specific logic."""

    binding = data_spec.get("backend_binding")
    if binding is not None:
        if not isinstance(binding, Mapping):
            raise ComponentRuntimeError("data.backend_binding must be an object")
        if execution_config is None or evidence_root is None:
            raise ComponentRuntimeError(
                "typed data adapter requires execution_config and evidence_root"
            )
        registration = registry.resolve(
            binding,
            expected_kind="data_adapter",
            required_operations=("materialize",),
        )
        result = registration.implementation.materialize(
            DataMaterializationRequest(
                dataset=dataset,
                data_spec=dict(data_spec),
                execution_config=dict(execution_config),
                evidence_root=evidence_root.resolve(),
                output_dir=output_dir.resolve(),
            )
        )
        if not isinstance(result, DataMaterializationResult):
            raise ComponentRuntimeError(
                "data adapter must return DataMaterializationResult"
            )
        manifest_path = result.dataset_manifest.resolve()
        if manifest_path != (output_dir.resolve() / "dataset_manifest.json"):
            raise ComponentRuntimeError(
                "data adapter must own output_dir/dataset_manifest.json"
            )
        manifest = _load_object(manifest_path, field="dataset manifest")
        if (
            manifest.get("manifest_type") != "data_freeze"
            or manifest.get("status") != "frozen"
            or manifest.get("dataset") != dataset
        ):
            raise ComponentRuntimeError("typed data adapter returned an invalid freeze")
        return manifest

    try:
        controller = get_dataset_controller(dataset)
    except ValueError as exc:
        raise ComponentRuntimeError(str(exc)) from exc
    return controller.execute(data_spec, output_dir)


def prepare_intervention(
    config: Mapping[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Resolve method, positions, and bank map through one fixed function."""

    method_plan = resolve_method_parameters(config)
    position_plan = resolve_positions(config)
    bank_plan = build_bank_plan(position_plan, method_plan)
    _write_exclusive_json(output_dir / "method_plan.json", method_plan)
    _write_exclusive_json(output_dir / "position_plan.json", position_plan)
    _write_exclusive_json(output_dir / "bank_plan.json", bank_plan)
    return {
        "method_plan": method_plan,
        "position_plan": position_plan,
        "bank_plan": bank_plan,
    }


def register_bank_result(
    *,
    bank_plan_path: Path,
    method_plan_path: Path,
    bank_id: str,
    payload_path: Path,
    training_manifest_path: Path,
    evidence_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Bind one completed bank payload to its exact intervention identity."""

    bank_plan = _load_object(bank_plan_path, field="bank plan")
    method_plan = _load_object(method_plan_path, field="method plan")
    pair = register_position_vector_pair(
        bank_plan,
        method_plan,
        bank_id=bank_id,
        payload_path=payload_path,
        training_manifest_path=training_manifest_path,
        evidence_root=evidence_root,
    )
    validate_position_vector_pair(pair, evidence_root=evidence_root)
    _write_exclusive_json(output_path, pair)
    return pair


def compose_bank_results(
    *,
    bank_plan_path: Path,
    pair_manifest_paths: Sequence[Path],
    evidence_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Validate all position-vector pairs and seal their inference composition."""

    bank_plan = _load_object(bank_plan_path, field="bank plan")
    pairs = [
        _load_object(path, field=f"position-vector pair {index}")
        for index, path in enumerate(pair_manifest_paths)
    ]
    for pair in pairs:
        validate_position_vector_pair(pair, evidence_root=evidence_root)
    controller = compose_controller_manifest(pairs, bank_plan)
    _write_exclusive_json(output_path, controller)
    return controller


def register_component_result(
    *,
    bank_pair_path: Path,
    component_id: str,
    payload_path: Path,
    extraction_manifest_path: Path,
    evidence_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Seal one method-extracted component view from a trained bank."""

    bank_pair = _load_object(bank_pair_path, field="bank pair")
    manifest = register_component_payload(
        bank_pair,
        component_id=component_id,
        payload_path=payload_path,
        extraction_manifest_path=extraction_manifest_path,
        evidence_root=evidence_root,
    )
    validate_component_payload(manifest, evidence_root=evidence_root)
    _write_exclusive_json(output_path, manifest)
    return manifest


def prepare_post_training_audit(
    *,
    bank_plan_path: Path,
    component_manifest_paths: Sequence[Path],
    audit_config: Mapping[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    """Create the immutable audit-unit plan after candidate-bank training."""

    bank_plan = _load_object(bank_plan_path, field="bank plan")
    components = [
        _load_object(path, field=f"component manifest {index}")
        for index, path in enumerate(component_manifest_paths)
    ]
    plan = build_post_training_audit_plan(bank_plan, components, audit_config)
    _write_exclusive_json(output_path, plan)
    return plan


def finalize_post_training_audit(
    *,
    audit_plan_path: Path,
    source_bank_plan_path: Path,
    decision_manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Freeze retained components and emit the final position/bank plans."""

    audit_plan = _load_object(audit_plan_path, field="post-training audit plan")
    source_bank_plan = _load_object(source_bank_plan_path, field="source bank plan")
    frozen = freeze_post_training_selection(
        audit_plan,
        source_bank_plan,
        decision_manifest_path=decision_manifest_path,
    )
    _write_exclusive_json(output_path, frozen)
    return frozen
