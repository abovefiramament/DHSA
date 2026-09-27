"""Bind trained vector payloads to positions, models, timings, and banks."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping, Sequence


class PayloadContractError(ValueError):
    """Raised when a vector payload is not compatible with its position plan."""


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PayloadContractError(f"{field} must be an object")
    return value


def _string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PayloadContractError(f"{field} must be a non-empty string")
    return value


def _artifact(path: Path, evidence_root: Path, *, field: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    root = evidence_root.expanduser().resolve()
    if not resolved.is_file():
        raise PayloadContractError(f"{field} does not exist: {resolved}")
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise PayloadContractError(f"{field} must remain under the evidence root") from exc
    return {
        "relative_path": relative.as_posix(),
        "size_bytes": resolved.stat().st_size,
    }


def register_position_vector_pair(
    bank_plan: Mapping[str, Any],
    method_plan: Mapping[str, Any],
    *,
    bank_id: str,
    payload_path: Path,
    training_manifest_path: Path,
    evidence_root: Path,
) -> dict[str, Any]:
    """Create one immutable identity binding for a trained bank payload."""

    banks = bank_plan.get("banks")
    if not isinstance(banks, list):
        raise PayloadContractError("bank_plan.banks must be a list")
    matches = [bank for bank in banks if bank.get("bank_id") == bank_id]
    if len(matches) != 1:
        raise PayloadContractError(f"unknown or ambiguous bank_id: {bank_id}")
    bank = _mapping(matches[0], field=f"bank[{bank_id}]")
    positions = bank.get("positions")
    if not isinstance(positions, list) or not positions:
        raise PayloadContractError(f"bank {bank_id!r} has no positions")
    component_ids = [
        _string(_mapping(row, field="bank.positions[]").get("component_id"), field="component_id")
        for row in positions
    ]
    controller = _mapping(method_plan.get("controller"), field="method_plan.controller")
    model = _mapping(method_plan.get("model"), field="method_plan.model")
    model_identity: dict[str, dict[str, str]] = {}
    for role, raw_model in model.items():
        item = _mapping(raw_model, field=f"method_plan.model.{role}")
        model_identity[role] = {
            "checkpoint": _string(
                item.get("checkpoint"), field=f"method_plan.model.{role}.checkpoint"
            ),
            "revision": _string(
                item.get("revision"), field=f"method_plan.model.{role}.revision"
            ),
        }
    legacy_timings = controller.get("timings")
    training_timings = controller.get("training_timings", legacy_timings)
    inference_timings = controller.get("inference_timings", legacy_timings)
    if not isinstance(training_timings, list) or not training_timings:
        raise PayloadContractError(
            "method_plan.controller.training_timings must be non-empty"
        )
    if not isinstance(inference_timings, list) or not inference_timings:
        raise PayloadContractError(
            "method_plan.controller.inference_timings must be non-empty"
        )
    raw_transfer = _mapping(method_plan.get("transfer"), field="method_plan.transfer")
    transfer = {"mode": _string(raw_transfer.get("mode"), field="transfer.mode")}
    if "compatibility" in raw_transfer:
        compatibility = dict(
            _mapping(raw_transfer["compatibility"], field="transfer.compatibility")
        )
        if compatibility != {
            "component_mapping": "identity_by_layer_and_head",
            "architecture_match": True,
        }:
            raise PayloadContractError("transfer compatibility declaration drift")
        transfer["compatibility"] = compatibility

    pair = {
        "schema_version": 1,
        "pair_id": f"{method_plan.get('method')}::{bank_id}",
        "method": _string(method_plan.get("method"), field="method_plan.method"),
        "operator_family": _string(controller.get("family"), field="controller.family"),
        "bank_id": bank_id,
        "component_type": _string(
            bank_plan.get("component_type"), field="bank_plan.component_type"
        ),
        "ordered_component_ids": component_ids,
        "vector_payload": _artifact(payload_path, evidence_root, field="payload_path"),
        "training_manifest": _artifact(
            training_manifest_path, evidence_root, field="training_manifest_path"
        ),
        "model": model_identity,
        "training_timings": copy.deepcopy(training_timings),
        "inference_timings": copy.deepcopy(inference_timings),
        "transfer": transfer,
    }
    if legacy_timings is not None:
        pair["timings"] = copy.deepcopy(legacy_timings)
    if "position_freeze" in bank_plan:
        pair["position_freeze"] = copy.deepcopy(bank_plan["position_freeze"])
    return pair


def validate_position_vector_pair(
    pair: Mapping[str, Any],
    *,
    evidence_root: Path,
) -> None:
    """Validate pair identity and referenced payload files."""

    if pair.get("schema_version") != 1:
        raise PayloadContractError("position-vector pair schema is invalid")
    root = evidence_root.expanduser().resolve()
    for field in ("vector_payload", "training_manifest"):
        artifact = _mapping(pair.get(field), field=field)
        relative = Path(_string(artifact.get("relative_path"), field=f"{field}.relative_path"))
        if relative.is_absolute() or ".." in relative.parts:
            raise PayloadContractError(f"{field} escapes the evidence root")
        path = root / relative
        if not path.is_file() or path.stat().st_size != artifact.get("size_bytes"):
            raise PayloadContractError(f"{field} is missing or has the wrong size")


def register_component_payload(
    bank_pair: Mapping[str, Any],
    *,
    component_id: str,
    payload_path: Path,
    extraction_manifest_path: Path,
    evidence_root: Path,
) -> dict[str, Any]:
    """Bind one extracted component payload to its jointly trained source bank."""

    component = _string(component_id, field="component_id")
    ordered = bank_pair.get("ordered_component_ids")
    if not isinstance(ordered, list) or component not in ordered:
        raise PayloadContractError(
            f"component {component!r} is absent from the source bank pair"
        )
    manifest = {
        "schema_version": 1,
        "component_id": component,
        "bank_id": _string(bank_pair.get("bank_id"), field="bank_pair.bank_id"),
        "source_pair_id": _string(bank_pair.get("pair_id"), field="bank_pair.pair_id"),
        "component_payload": _artifact(
            payload_path, evidence_root, field="component_payload"
        ),
        "extraction_manifest": _artifact(
            extraction_manifest_path,
            evidence_root,
            field="extraction_manifest",
        ),
    }
    return manifest


def validate_component_payload(
    manifest: Mapping[str, Any],
    *,
    evidence_root: Path,
) -> None:
    """Verify an extracted component payload and its source-binding manifest."""

    if manifest.get("schema_version") != 1:
        raise PayloadContractError("component payload schema is invalid")
    root = evidence_root.expanduser().resolve()
    for field in ("component_payload", "extraction_manifest"):
        artifact = _mapping(manifest.get(field), field=field)
        relative = Path(
            _string(artifact.get("relative_path"), field=f"{field}.relative_path")
        )
        if relative.is_absolute() or ".." in relative.parts:
            raise PayloadContractError(f"{field} escapes the evidence root")
        path = root / relative
        if not path.is_file() or path.stat().st_size != artifact.get("size_bytes"):
            raise PayloadContractError(f"{field} is missing or has the wrong size")


def compose_controller_manifest(
    pairs: Sequence[Mapping[str, Any]],
    bank_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate all banks and emit the deterministic inference composition."""

    expected = bank_plan["composition"]["bank_order"]
    by_id = {pair.get("bank_id"): dict(pair) for pair in pairs}
    if len(by_id) != len(pairs) or set(by_id) != set(expected):
        raise PayloadContractError("pair manifests do not match the registered bank set")
    planned = {bank["bank_id"]: bank for bank in bank_plan.get("banks", [])}
    for bank_id in expected:
        pair_ids = by_id[bank_id].get("ordered_component_ids")
        planned_ids = [row["component_id"] for row in planned[bank_id]["positions"]]
        if pair_ids != planned_ids:
            raise PayloadContractError(f"bank {bank_id!r} component order changed")
    manifest = {
        "schema_version": 1,
        "manifest_type": "controller_freeze",
        "status": "frozen",
        "composition": copy.deepcopy(bank_plan["composition"]),
        "ordered_pairs": [
            {
                "bank_id": bank_id,
                "pair_id": by_id[bank_id]["pair_id"],
                "ordered_component_ids": copy.deepcopy(
                    by_id[bank_id]["ordered_component_ids"]
                ),
                "vector_payload": copy.deepcopy(by_id[bank_id]["vector_payload"]),
            }
            for bank_id in expected
        ],
    }
    if "position_freeze" in bank_plan:
        manifest["position_freeze"] = copy.deepcopy(bank_plan["position_freeze"])
    manifest["artifact_closure"] = [
        copy.deepcopy(item["vector_payload"])
        for item in manifest["ordered_pairs"]
    ]
    return manifest
