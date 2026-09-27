"""Run one registered position method without owning its ranking policy."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .component_registry import COMPONENT_REGISTRY, ComponentRegistry
from .contracts import (
    PositionSearchRequest,
    PositionSearchResult,
)
from .trajectory_controller import normalize_trajectory_config


class PositionSearchError(ValueError):
    """Raised when a position backend breaks its typed boundary."""


def _load(path: Path, *, field: str) -> dict[str, Any]:
    if not path.is_file():
        raise PositionSearchError(f"{field} does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PositionSearchError(f"{field} must contain one JSON object")
    return value


def _artifact(path: Path, root: Path, *, field: str) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise PositionSearchError(f"{field} escapes evidence root") from exc
    if not resolved.is_file():
        raise PositionSearchError(f"{field} does not exist: {resolved}")
    return {
        "relative_path": relative.as_posix(),
        "size_bytes": resolved.stat().st_size,
    }


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise PositionSearchError(f"immutable position-search artifact exists: {path}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(rendered)


def run_registered_position_backend(
    *,
    scanner_binding: Mapping[str, Any],
    backend_binding: Mapping[str, Any],
    method_plan: Mapping[str, Any],
    model_config: Mapping[str, Any],
    scan_config: Mapping[str, Any],
    execution_config: Mapping[str, Any],
    trajectory_config: Mapping[str, Any] | None = None,
    selector_data_manifest: Path,
    evidence_root: Path,
    output_dir: Path,
    manifest_path: Path,
    registry: ComponentRegistry = COMPONENT_REGISTRY,
) -> dict[str, Any]:
    """Run one model scanner and then one model-free position selector."""

    if method_plan.get("method_kind") != "position_baseline":
        raise PositionSearchError("position backend requires one position-baseline method")
    role = _load(selector_data_manifest, field="selector data role")
    if (
        role.get("manifest_type") != "data_role_freeze"
        or role.get("status") != "frozen"
        or role.get("purpose") != "selection"
    ):
        raise PositionSearchError("position backend requires a frozen selection role")
    if not isinstance(model_config, Mapping) or not isinstance(scan_config, Mapping):
        raise PositionSearchError("position scan requires model_config and scan_config")
    effective_trajectory = normalize_trajectory_config(trajectory_config)
    scanner_registration = registry.resolve(
        scanner_binding,
        expected_kind="position_scanner",
        required_operations=("scan",),
    )
    method = str(method_plan["method"])
    effective_scan = copy.deepcopy(dict(scan_config))
    if "control_objective" in method_plan:
        effective_scan["control_objective"] = copy.deepcopy(
            method_plan["control_objective"]
        )
    scan_dir = output_dir / "scan"
    if method in {"rcm_zero", "rcm_patch"}:
        from baseline.implementations.scanners.rcm import RCMScanRequest

        scan_request: Any = RCMScanRequest(
            method=method,
            data_role_manifest=selector_data_manifest.resolve(),
            model_config=copy.deepcopy(dict(model_config)),
            scan_config=effective_scan,
            trajectory_config=effective_trajectory,
            evidence_root=evidence_root.resolve(),
            output_dir=scan_dir,
        )
    elif method == "iti":
        from baseline.implementations.scanners.iti import ITIScanRequest

        scan_request = ITIScanRequest(
            data_role_manifest=selector_data_manifest.resolve(),
            model_config=copy.deepcopy(dict(model_config)),
            scan_config=effective_scan,
            trajectory_config=effective_trajectory,
            evidence_root=evidence_root.resolve(),
            output_dir=scan_dir,
        )
    elif method == "random":
        from baseline.implementations.scanners.random import RandomScanRequest

        scan_request = RandomScanRequest(
            model_config=copy.deepcopy(dict(model_config)),
            scan_config=effective_scan,
            evidence_root=evidence_root.resolve(),
            output_dir=scan_dir,
        )
    else:
        raise PositionSearchError(f"unsupported position method: {method}")
    scan_result = scanner_registration.implementation.scan(scan_request)
    from baseline.implementations.position_scanning import PositionScanResult

    if not isinstance(scan_result, PositionScanResult):
        raise PositionSearchError("position scanner must return PositionScanResult")
    registration = registry.resolve(
        backend_binding,
        expected_kind="position_backend",
        required_operations=("select_positions",),
    )
    result = registration.implementation.select_positions(
        PositionSearchRequest(
            method_plan=copy.deepcopy(method_plan),
            trajectory_config=effective_trajectory,
            execution_config=copy.deepcopy(execution_config),
            selector_data_manifest=scan_result.candidate_scores.resolve(),
            evidence_root=evidence_root.resolve(),
            output_dir=output_dir,
        )
    )
    if not isinstance(result, PositionSearchResult):
        raise PositionSearchError("position backend must return PositionSearchResult")
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "scanner": {
            "id": scanner_registration.component_id,
            "version": scanner_registration.version,
            "input_contract": scanner_registration.input_contract,
            "output_contract": scanner_registration.output_contract,
        },
        "backend": {
            "id": registration.component_id,
            "version": registration.version,
            "input_contract": registration.input_contract,
            "output_contract": registration.output_contract,
        },
        "method": method,
        "trajectory_config": effective_trajectory,
        "trace_kind": "rcm" if method.startswith("rcm_") else method,
        "selector_data_role": _artifact(
            selector_data_manifest, evidence_root, field="selector data role"
        ),
        "scan_candidates": _artifact(
            scan_result.candidate_scores, evidence_root, field="scan candidates"
        ),
        "scan_execution_manifest": _artifact(
            scan_result.execution_manifest, evidence_root, field="scan execution"
        ),
        "candidate_manifest": _artifact(
            result.candidate_manifest, evidence_root, field="candidate manifest"
        ),
        "backend_execution_manifest": _artifact(
            result.execution_manifest, evidence_root, field="backend execution manifest"
        ),
        "trajectory": (
            _artifact(scan_result.trajectory_path, evidence_root, field="trajectory")
            if scan_result.trajectory_path is not None
            else None
        ),
        "component_scores": (
            _artifact(
                scan_result.component_scores_path,
                evidence_root,
                field="component scores",
            )
            if scan_result.component_scores_path is not None
            else None
        ),
    }
    manifest["artifact_closure"] = [
        manifest["selector_data_role"],
        manifest["scan_candidates"],
        manifest["scan_execution_manifest"],
        manifest["candidate_manifest"],
        manifest["backend_execution_manifest"],
    ] + [
        manifest[key]
        for key in ("trajectory", "component_scores")
        if manifest[key] is not None
    ]
    _write_exclusive(manifest_path, manifest)
    return manifest
