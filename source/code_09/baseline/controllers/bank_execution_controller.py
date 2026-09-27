"""Execute a complete bank lifecycle through one registered backend.

The controller owns ordering, per-bank isolation, payload registration,
optional component extraction, and deterministic inference composition.  It
does not implement a training objective or manipulate method-owned tensors.
"""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.shared.component_registry import (
    COMPONENT_REGISTRY,
    ComponentRegistry,
)
from experiments.shared.contracts import (
    BankComposeRequest,
    BankTrainRequest,
    BankTrainResult,
)

from .payload_controller import (
    compose_controller_manifest,
    register_component_payload,
    register_position_vector_pair,
    validate_component_payload,
    validate_position_vector_pair,
)


BANK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class BankExecutionError(ValueError):
    """Raised when a backend cannot complete the registered bank plan."""


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BankExecutionError(f"{field} must be an object")
    return value


def _string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BankExecutionError(f"{field} must be a non-empty string")
    return value


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise BankExecutionError(f"immutable bank artifact already exists: {path}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(rendered)


def _validate_result(result: Any, *, operation: str) -> BankTrainResult:
    if not isinstance(result, BankTrainResult):
        raise BankExecutionError(
            f"bank backend {operation} must return BankTrainResult"
        )
    if (
        not isinstance(result.optimizer_steps, int)
        or isinstance(result.optimizer_steps, bool)
        or result.optimizer_steps < 0
    ):
        raise BankExecutionError("bank result optimizer_steps must be non-negative")
    for field, path in (
        ("payload_path", result.payload_path),
        ("training_manifest_path", result.training_manifest_path),
    ):
        if not isinstance(path, Path) or not path.is_file():
            raise BankExecutionError(f"bank result {field} is unavailable: {path}")
    return result


def _bank_rows(bank_plan: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = bank_plan.get("banks")
    if not isinstance(raw, list) or not raw:
        raise BankExecutionError("bank_plan.banks must be non-empty")
    rows = [_mapping(item, field="bank_plan.banks[]") for item in raw]
    expected = bank_plan.get("composition", {}).get("bank_order")
    actual = [row.get("bank_id") for row in rows]
    if not isinstance(expected, list) or actual != expected:
        raise BankExecutionError("bank order disagrees with composition.bank_order")
    if len(actual) != len(set(actual)):
        raise BankExecutionError("bank plan contains duplicate bank IDs")
    for bank_id in actual:
        if not isinstance(bank_id, str) or BANK_ID.fullmatch(bank_id) is None:
            raise BankExecutionError(f"unsafe bank ID: {bank_id!r}")
    return rows


def _validate_data_role(
    path: Path,
    *,
    field: str,
    allowed_purposes: set[str],
    expected_dataset: str | None,
) -> dict[str, Any]:
    if not path.is_file():
        raise BankExecutionError(f"{field} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BankExecutionError(f"{field} is not a JSON role manifest") from exc
    if not isinstance(value, dict):
        raise BankExecutionError(f"{field} must contain one object")
    if value.get("purpose") not in allowed_purposes:
        raise BankExecutionError(
            f"{field} purpose must be one of {sorted(allowed_purposes)}"
        )
    if expected_dataset is not None and value.get("dataset") != expected_dataset:
        raise BankExecutionError(f"{field} belongs to another dataset")
    return value


def _artifact_closure(
    pairs: Sequence[Mapping[str, Any]],
    components: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    closure: list[dict[str, Any]] = []
    for pair in pairs:
        for field in ("vector_payload", "training_manifest"):
            closure.append({"owner": pair["pair_id"], "field": field, **pair[field]})
    for component in components:
        for field in ("component_payload", "extraction_manifest"):
            closure.append(
                {
                    "owner": component["component_id"],
                    "field": field,
                    **component[field],
                }
            )
    return closure


def _register_bank_result(
    *,
    bank_plan: Mapping[str, Any],
    method_plan: Mapping[str, Any],
    bank: Mapping[str, Any],
    result: BankTrainResult,
    evidence_root: Path,
    bank_dir: Path,
    require_component_payloads: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    bank_id = str(bank["bank_id"])
    pair = register_position_vector_pair(
        bank_plan,
        method_plan,
        bank_id=bank_id,
        payload_path=result.payload_path,
        training_manifest_path=result.training_manifest_path,
        evidence_root=evidence_root,
    )
    validate_position_vector_pair(pair, evidence_root=evidence_root)
    _write_exclusive(bank_dir / "position_vector_pair.json", pair)

    expected_components = [row["component_id"] for row in bank["positions"]]
    returned_components = [item.component_id for item in result.component_payloads]
    if returned_components and returned_components != expected_components:
        raise BankExecutionError(
            f"bank {bank_id!r} component payload order/coverage drift"
        )
    if require_component_payloads and returned_components != expected_components:
        raise BankExecutionError(
            f"bank {bank_id!r} must expose every trained component payload"
        )
    manifests: list[dict[str, Any]] = []
    for index, component in enumerate(result.component_payloads):
        manifest = register_component_payload(
            pair,
            component_id=component.component_id,
            payload_path=component.payload_path,
            extraction_manifest_path=component.extraction_manifest_path,
            evidence_root=evidence_root,
        )
        validate_component_payload(manifest, evidence_root=evidence_root)
        _write_exclusive(
            bank_dir / "components" / f"component_{index:04d}.json", manifest
        )
        manifests.append(manifest)
    return pair, manifests


def execute_bank_plan(
    bank_plan: Mapping[str, Any],
    method_plan: Mapping[str, Any],
    *,
    backend: Any,
    backend_identity: Mapping[str, Any],
    training_data_manifest: Path,
    validation_data_manifest: Path | None,
    evidence_root: Path,
    output_dir: Path,
    execution_config: Mapping[str, Any],
    require_component_payloads: bool = False,
) -> dict[str, Any]:
    """Train every planned bank exactly once and freeze its composition."""

    position_freeze = _mapping(
        bank_plan.get("position_freeze"), field="bank_plan.position_freeze"
    )
    expected_dataset = position_freeze.get("dataset")
    training_role = _validate_data_role(
        training_data_manifest,
        field="training_data_manifest",
        allowed_purposes={"training"},
        expected_dataset=expected_dataset,
    )
    validation_role = None
    if validation_data_manifest is not None:
        validation_role = _validate_data_role(
            validation_data_manifest,
            field="validation_data_manifest",
            allowed_purposes={"training_validation"},
            expected_dataset=expected_dataset,
        )
    banks = _bank_rows(bank_plan)
    pairs: list[dict[str, Any]] = []
    components: list[dict[str, Any]] = []
    executions: list[dict[str, Any]] = []
    for bank in banks:
        bank_id = str(bank["bank_id"])
        bank_dir = output_dir / "banks" / bank_id
        request = BankTrainRequest(
            bank_id=bank_id,
            ordered_positions=tuple(copy.deepcopy(bank["positions"])),
            method_plan=copy.deepcopy(method_plan),
            execution_config=copy.deepcopy(execution_config),
            training_data_manifest=training_data_manifest.resolve(),
            validation_data_manifest=(
                validation_data_manifest.resolve()
                if validation_data_manifest is not None
                else None
            ),
            evidence_root=evidence_root.resolve(),
            output_dir=bank_dir,
        )
        result = _validate_result(backend.train_bank(request), operation="train_bank")
        pair, bank_components = _register_bank_result(
            bank_plan=bank_plan,
            method_plan=method_plan,
            bank=bank,
            result=result,
            evidence_root=evidence_root,
            bank_dir=bank_dir,
            require_component_payloads=require_component_payloads,
        )
        pairs.append(pair)
        components.extend(bank_components)
        executions.append(
            {
                "bank_id": bank_id,
                "operation": "train_bank",
                "optimizer_steps": result.optimizer_steps,
                "pair_id": pair["pair_id"],
                "component_ids": [item["component_id"] for item in bank_components],
            }
        )
    controller = compose_controller_manifest(pairs, bank_plan)
    controller_path = output_dir / "controller_manifest.json"
    _write_exclusive(controller_path, controller)
    execution = {
        "schema_version": 1,
        "status": "complete",
        "phase": "bank_training",
        "backend": copy.deepcopy(dict(backend_identity)),
        "method": method_plan.get("method"),
        "execution_profile": execution_config.get("profile"),
        "dataset": expected_dataset,
        "training_data_role": training_role["data_role"],
        "validation_data_role": (
            validation_role["data_role"] if validation_role is not None else None
        ),
        "require_component_payloads": require_component_payloads,
        "banks": executions,
        "controller_manifest": {
            "relative_path": controller_path.resolve()
            .relative_to(evidence_root.resolve())
            .as_posix(),
            "size_bytes": controller_path.stat().st_size,
        },
        "artifact_closure": _artifact_closure(pairs, components),
    }
    execution_path = output_dir / "bank_execution_manifest.json"
    _write_exclusive(execution_path, execution)
    return {
        "bank_execution_manifest": execution,
        "bank_execution_manifest_path": execution_path,
        "controller_manifest": controller,
        "controller_manifest_path": controller_path,
        "position_vector_pairs": pairs,
        "component_payload_manifests": components,
    }


def execute_registered_bank_plan(
    bank_plan: Mapping[str, Any],
    method_plan: Mapping[str, Any],
    *,
    backend_binding: Mapping[str, Any],
    training_data_manifest: Path,
    validation_data_manifest: Path | None,
    evidence_root: Path,
    output_dir: Path,
    execution_config: Mapping[str, Any],
    require_component_payloads: bool = False,
    registry: ComponentRegistry = COMPONENT_REGISTRY,
) -> dict[str, Any]:
    registration = registry.resolve(
        backend_binding,
        expected_kind="bank_backend",
        required_operations=("train_bank",),
    )
    return execute_bank_plan(
        bank_plan,
        method_plan,
        backend=registration.implementation,
        backend_identity={
            "kind": registration.kind,
            "id": registration.component_id,
            "version": registration.version,
            "input_contract": registration.input_contract,
            "output_contract": registration.output_contract,
        },
        training_data_manifest=training_data_manifest,
        validation_data_manifest=validation_data_manifest,
        evidence_root=evidence_root,
        output_dir=output_dir,
        execution_config=execution_config,
        require_component_payloads=require_component_payloads,
    )


def compose_audited_banks(
    audit_freeze: Mapping[str, Any],
    method_plan: Mapping[str, Any],
    component_payload_manifests: Sequence[Mapping[str, Any]],
    *,
    backend_binding: Mapping[str, Any],
    evidence_root: Path,
    output_dir: Path,
    execution_config: Mapping[str, Any],
    registry: ComponentRegistry = COMPONENT_REGISTRY,
) -> dict[str, Any]:
    """Compose retained component payloads with exactly zero optimizer steps."""

    if audit_freeze.get("status") != "frozen":
        raise BankExecutionError("post-training audit selection is not frozen")
    finalization = _mapping(
        audit_freeze.get("finalization"), field="audit_freeze.finalization"
    )
    if finalization.get("mode") != "reuse_component_payloads":
        raise BankExecutionError(
            "compose_audited_banks requires reuse_component_payloads"
        )
    registration = registry.resolve(
        backend_binding,
        expected_kind="bank_backend",
        required_operations=("compose_bank",),
    )
    bank_plan = _mapping(
        audit_freeze.get("final_bank_plan"), field="audit_freeze.final_bank_plan"
    )
    retained = audit_freeze.get("retained_component_ids")
    if not isinstance(retained, list) or not retained:
        raise BankExecutionError("audit freeze has no retained components")
    by_component: dict[str, Mapping[str, Any]] = {}
    for manifest in component_payload_manifests:
        item = _mapping(manifest, field="component_payload_manifests[]")
        component_id = _string(item.get("component_id"), field="component_id")
        if component_id in by_component:
            raise BankExecutionError(f"duplicate component payload: {component_id}")
        validate_component_payload(item, evidence_root=evidence_root)
        by_component[component_id] = item
    if not set(retained).issubset(by_component):
        raise BankExecutionError("retained component payload coverage is incomplete")

    pairs: list[dict[str, Any]] = []
    executions: list[dict[str, Any]] = []
    for bank in _bank_rows(bank_plan):
        bank_id = str(bank["bank_id"])
        component_ids = tuple(row["component_id"] for row in bank["positions"])
        bank_dir = output_dir / "banks" / bank_id
        request = BankComposeRequest(
            bank_id=bank_id,
            ordered_component_ids=component_ids,
            component_manifests=tuple(
                copy.deepcopy(by_component[component_id])
                for component_id in component_ids
            ),
            method_plan=copy.deepcopy(method_plan),
            execution_config=copy.deepcopy(execution_config),
            evidence_root=evidence_root.resolve(),
            output_dir=bank_dir,
        )
        result = _validate_result(
            registration.implementation.compose_bank(request),
            operation="compose_bank",
        )
        if result.optimizer_steps != 0:
            raise BankExecutionError(
                "reuse_component_payloads composition must report zero optimizer steps"
            )
        pair, unexpected_components = _register_bank_result(
            bank_plan=bank_plan,
            method_plan=method_plan,
            bank=bank,
            result=result,
            evidence_root=evidence_root,
            bank_dir=bank_dir,
            require_component_payloads=False,
        )
        if unexpected_components:
            raise BankExecutionError("compose_bank must not create new component payloads")
        pairs.append(pair)
        executions.append(
            {
                "bank_id": bank_id,
                "operation": "compose_bank",
                "optimizer_steps": 0,
                "source_component_ids": list(component_ids),
                "pair_id": pair["pair_id"],
            }
        )
    controller = compose_controller_manifest(pairs, bank_plan)
    controller_path = output_dir / "controller_manifest.json"
    _write_exclusive(controller_path, controller)
    execution = {
        "schema_version": 1,
        "status": "complete",
        "phase": "post_audit_zero_step_composition",
        "backend": {
            "kind": registration.kind,
            "id": registration.component_id,
            "version": registration.version,
        },
        "method": method_plan.get("method"),
        "execution_profile": execution_config.get("profile"),
        "banks": executions,
        "artifact_closure": _artifact_closure(pairs, []),
    }
    execution_path = output_dir / "bank_execution_manifest.json"
    _write_exclusive(execution_path, execution)
    return {
        "bank_execution_manifest": execution,
        "bank_execution_manifest_path": execution_path,
        "controller_manifest": controller,
        "controller_manifest_path": controller_path,
        "position_vector_pairs": pairs,
    }


def finalize_audited_bank_plan(
    audit_freeze: Mapping[str, Any],
    method_plan: Mapping[str, Any],
    *,
    backend_binding: Mapping[str, Any],
    evidence_root: Path,
    output_dir: Path,
    execution_config: Mapping[str, Any],
    component_payload_manifests: Sequence[Mapping[str, Any]] = (),
    training_data_manifest: Path | None = None,
    validation_data_manifest: Path | None = None,
    registry: ComponentRegistry = COMPONENT_REGISTRY,
) -> dict[str, Any]:
    """Complete either allowed post-audit path without an implicit choice."""

    finalization = _mapping(
        audit_freeze.get("finalization"), field="audit_freeze.finalization"
    )
    mode = finalization.get("mode")
    if mode == "reuse_component_payloads":
        if training_data_manifest is not None or validation_data_manifest is not None:
            raise BankExecutionError(
                "reuse_component_payloads cannot receive retraining data"
            )
        return compose_audited_banks(
            audit_freeze,
            method_plan,
            component_payload_manifests,
            backend_binding=backend_binding,
            evidence_root=evidence_root,
            output_dir=output_dir,
            execution_config=execution_config,
            registry=registry,
        )
    if mode == "retrain_selected_banks":
        if component_payload_manifests:
            raise BankExecutionError(
                "retrain_selected_banks cannot consume candidate component payloads"
            )
        if training_data_manifest is None:
            raise BankExecutionError(
                "retrain_selected_banks requires the registered training data"
            )
        final_bank_plan = _mapping(
            audit_freeze.get("final_bank_plan"),
            field="audit_freeze.final_bank_plan",
        )
        return execute_registered_bank_plan(
            final_bank_plan,
            method_plan,
            backend_binding=backend_binding,
            training_data_manifest=training_data_manifest,
            validation_data_manifest=validation_data_manifest,
            evidence_root=evidence_root,
            output_dir=output_dir,
            execution_config=execution_config,
            require_component_payloads=False,
            registry=registry,
        )
    raise BankExecutionError(f"unsupported audit finalization mode: {mode!r}")
