"""Validate one experiment/data/model-family configuration bundle."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]



class BundleValidationError(ValueError):
    """Raised when a configuration bundle is incomplete or drifts."""


FORBIDDEN_CHILD_FIELDS = frozenset(
    {
        "experiment",
        "dataset",
        "model_family",
        "protocol",
        "shared",
        "data",
        "models",
        "flow",
        "evaluation",
        "matrix",
    }
)
REQUIRED_SHARED_FIELDS = frozenset(
    {"data", "models", "flow", "selection_calibration_test", "evaluation", "execution"}
)


def _object(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BundleValidationError(f"{field} must be an object")
    return value


def _reject_machine_paths(value: Any, *, field: str) -> None:
    if isinstance(value, str):
        if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
            raise BundleValidationError(
                f"{field} contains an absolute machine path; use registry:// instead"
            )
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_machine_paths(item, field=f"{field}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_machine_paths(item, field=f"{field}[{index}]")

def _portable_bundle_directory(bundle_dir: Path) -> dict[str, str]:
    try:
        relative = bundle_dir.relative_to(REPO_ROOT)
    except ValueError:
        return {
            "scope": "external",
            "name": bundle_dir.name,
        }
    return {"scope": "repository", "path": relative.as_posix()}



def _string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BundleValidationError(f"{field} must be a non-empty string")
    return value


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleValidationError(f"cannot load JSON config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BundleValidationError(f"{path} must contain one JSON object")
    return value


def _local_file(bundle_dir: Path, filename: Any, *, field: str) -> Path:
    name = _string(filename, field=field)
    candidate = Path(name)
    if candidate.name != name or candidate.is_absolute():
        raise BundleValidationError(f"{field} must be a plain filename in the bundle directory")
    resolved = (bundle_dir / candidate).resolve()
    if resolved.parent != bundle_dir.resolve():
        raise BundleValidationError(f"{field} escapes the bundle directory")
    if not resolved.is_file():
        raise BundleValidationError(f"referenced config file does not exist: {resolved}")
    return resolved


def _plan_files(
    bundle_dir: Path,
    bundle_id: str,
    plans: Mapping[str, Any],
    *,
    plan_type: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    loaded: dict[str, dict[str, Any]] = {}
    filenames: dict[str, str] = {}
    for plan_id, filename in plans.items():
        plan_name = _string(plan_id, field=f"{plan_type} plan id")
        path = _local_file(
            bundle_dir,
            filename,
            field=f"plans.{plan_type}.{plan_name}",
        )
        payload = _load(path)
        _reject_machine_paths(payload, field=path.name)
        if payload.get("bundle_id") != bundle_id:
            raise BundleValidationError(f"{path.name} bundle_id mismatch")
        if payload.get("plan_type") != plan_type:
            raise BundleValidationError(f"{path.name} plan_type must be {plan_type!r}")
        if payload.get("plan_id") != plan_name:
            raise BundleValidationError(f"{path.name} plan_id mismatch")
        forbidden = sorted(FORBIDDEN_CHILD_FIELDS.intersection(payload))
        if forbidden:
            raise BundleValidationError(
                f"{path.name} duplicates bundle-owned fields: {forbidden}"
            )
        payload_field = "baseline" if plan_type == "method" else "position_control"
        _object(payload.get(payload_field), field=f"{path.name}.{payload_field}")
        loaded[plan_name] = payload
        filenames[plan_name] = path.name
    return loaded, filenames


def validate_bundle(bundle_dir: Path) -> dict[str, Any]:
    """Validate and resolve a flat dataset/model-family bundle."""

    bundle_dir = bundle_dir.expanduser().resolve()
    if not bundle_dir.is_dir():
        raise BundleValidationError(f"bundle directory does not exist: {bundle_dir}")
    master_path = bundle_dir / "bundle.json"
    if not master_path.is_file():
        raise BundleValidationError(f"bundle.json is required in {bundle_dir}")
    master = _load(master_path)
    if master.get("schema_version") != 1:
        raise BundleValidationError("bundle.schema_version must be 1")
    bundle_id = _string(master.get("bundle_id"), field="bundle.bundle_id")
    _reject_machine_paths(master, field="bundle")
    identity = {
        "bundle_id": bundle_id,
        "experiment": _string(master.get("experiment"), field="bundle.experiment"),
        "dataset": _string(master.get("dataset"), field="bundle.dataset"),
        "model_family": _string(master.get("model_family"), field="bundle.model_family"),
    }
    protocol = _object(master.get("protocol"), field="bundle.protocol")
    protocol_id = _string(protocol.get("id"), field="bundle.protocol.id")
    protocol_path_value = _string(protocol.get("path"), field="bundle.protocol.path")
    protocol_path = (REPO_ROOT / protocol_path_value).resolve()
    try:
        protocol_path.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise BundleValidationError("bundle.protocol.path escapes the repository") from exc
    protocol_document = _load(protocol_path)
    if protocol_document.get("protocol_id") != protocol_id:
        raise BundleValidationError("bundle protocol id does not match the referenced document")
    protocol_status = protocol_document.get("status")
    if protocol_status not in {"canonical_formal_matrix", "framework_demo"}:
        raise BundleValidationError("referenced protocol is not executable")
    protocol_freeze: dict[str, Any] = {
        "protocol_id": protocol_id,
        "protocol_path": protocol_path_value,
        "execution_authorized": True,
        "formal": protocol_status == "canonical_formal_matrix",
    }

    shared = _object(master.get("shared"), field="bundle.shared")
    missing_shared = sorted(REQUIRED_SHARED_FIELDS.difference(shared))
    extra_shared = sorted(set(shared).difference(REQUIRED_SHARED_FIELDS))
    if missing_shared or extra_shared:
        raise BundleValidationError(
            f"bundle.shared fields mismatch: missing={missing_shared} extra={extra_shared}"
        )
    flow = shared["flow"]
    if not isinstance(flow, list) or not flow:
        raise BundleValidationError("bundle.shared.flow must be a non-empty ordered stage list")
    if all(isinstance(stage, str) and stage for stage in flow):
        if len(flow) != len(set(flow)):
            raise BundleValidationError("bundle.shared.flow contains duplicate stages")
    elif all(isinstance(stage, Mapping) for stage in flow):
        stage_ids: set[str] = set()
        for index, stage in enumerate(flow):
            stage_id = _string(stage.get("id"), field=f"bundle.shared.flow[{index}].id")
            _string(stage.get("handler"), field=f"bundle.shared.flow[{index}].handler")
            _object(stage.get("inputs"), field=f"bundle.shared.flow[{index}].inputs")
            outputs = _object(
                stage.get("outputs"), field=f"bundle.shared.flow[{index}].outputs"
            )
            if not outputs:
                raise BundleValidationError(
                    f"bundle.shared.flow[{index}].outputs must not be empty"
                )
            if stage_id in stage_ids:
                raise BundleValidationError(f"duplicate flow stage id: {stage_id}")
            stage_ids.add(stage_id)
    else:
        raise BundleValidationError(
            "bundle.shared.flow must contain only stage names or only stage objects"
        )

    plans = _object(master.get("plans"), field="bundle.plans")
    method_map = _object(plans.get("method"), field="bundle.plans.method")
    position_map = _object(plans.get("position"), field="bundle.plans.position")
    methods, method_files = _plan_files(
        bundle_dir, bundle_id, method_map, plan_type="method"
    )
    positions, position_files = _plan_files(
        bundle_dir, bundle_id, position_map, plan_type="position"
    )

    matrix = master.get("matrix")
    if not isinstance(matrix, list) or not matrix:
        raise BundleValidationError("bundle.matrix must be a non-empty list")
    cells: list[dict[str, Any]] = []
    cell_ids: set[str] = set()
    for index, raw in enumerate(matrix):
        cell = _object(raw, field=f"bundle.matrix[{index}]")
        cell_id = _string(cell.get("cell_id"), field=f"bundle.matrix[{index}].cell_id")
        if cell_id in cell_ids:
            raise BundleValidationError(f"duplicate matrix cell_id: {cell_id}")
        cell_ids.add(cell_id)
        method_ref = _string(cell.get("method_ref"), field=f"matrix.{cell_id}.method_ref")
        if method_ref not in methods:
            raise BundleValidationError(f"matrix cell {cell_id} has unknown method_ref")
        position_ref = cell.get("position_ref")
        if position_ref is not None:
            position_ref = _string(position_ref, field=f"matrix.{cell_id}.position_ref")
            if position_ref not in positions:
                raise BundleValidationError(f"matrix cell {cell_id} has unknown position_ref")
        resolved = {
            "identity": identity,
            "protocol": copy.deepcopy(protocol),
            "shared": copy.deepcopy(shared),
            "cell": copy.deepcopy(dict(cell)),
            "baseline": copy.deepcopy(methods[method_ref]["baseline"]),
        }
        if position_ref is not None:
            resolved["position_control"] = copy.deepcopy(
                positions[position_ref]["position_control"]
            )
        cells.append(
            {
                "cell_id": cell_id,
                "method_ref": method_ref,
                "position_ref": position_ref,
                "resolved_config": resolved,
            }
        )

    declared_json = {"bundle.json", *method_files.values(), *position_files.values()}
    # A colocated machine-readable protocol is part of the bundle inventory;
    # repository-level protocol files remain outside the flat bundle.
    # Keep a copied/self-contained bundle valid when it carries the same
    # protocol filename next to bundle.json, even if the protocol reference
    # resolves to the repository source.
    if protocol_path.parent == bundle_dir or (bundle_dir / protocol_path.name).is_file():
        declared_json.add(protocol_path.name)
    actual_json = {path.name for path in bundle_dir.glob("*.json")}
    if actual_json != declared_json:
        raise BundleValidationError(
            f"bundle JSON inventory mismatch: undeclared={sorted(actual_json-declared_json)} "
            f"missing={sorted(declared_json-actual_json)}"
        )
    return {
        **identity,
        "bundle_directory": _portable_bundle_directory(bundle_dir),
        "files": sorted(declared_json),
        "cells": cells,
        "protocol_freeze": protocol_freeze,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate one experiment/dataset/model-family config bundle."
    )
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = validate_bundle(args.bundle_dir)
    rendered = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
