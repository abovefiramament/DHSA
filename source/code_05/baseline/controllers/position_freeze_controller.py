"""Freeze one position-baseline cell for consumption by intervention cells.

The selector owns candidate scoring.  This controller owns the immutable
boundary: method identity, selected geometry, and the exact data freeze used
for selection.  Downstream cells consume this file, never selector scratch
directories.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Mapping

class PositionFreezeError(ValueError):
    """Raised when a position plan cannot form a portable frozen artifact."""


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PositionFreezeError(f"{field} must be an object")
    return value


def _string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PositionFreezeError(f"{field} must be a non-empty string")
    return value


def _load_object(path: Path, *, field: str) -> dict[str, Any]:
    if not path.is_file():
        raise PositionFreezeError(f"{field} does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PositionFreezeError(f"{field} must contain one JSON object")
    return value


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise PositionFreezeError(f"immutable position freeze exists: {path}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(rendered)


def _data_freeze_from_reference(path: Path, evidence_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    value = _load_object(path, field="data freeze")
    if value.get("manifest_type") == "data_freeze":
        direct_path = path
        closure_path = path
        manifest = value
    else:
        artifact = _mapping(value.get("dataset_manifest"), field="dataset_manifest")
        relative = Path(_string(artifact.get("relative_path"), field="dataset_manifest.relative_path"))
        if relative.is_absolute():
            raise PositionFreezeError("dataset manifest reference must be relative")
        direct_path = evidence_root.resolve() / relative
        try:
            direct_path.resolve().relative_to(evidence_root.resolve().parent)
        except ValueError as exc:
            raise PositionFreezeError("shared dataset manifest escapes experiment root") from exc
        manifest = _load_object(direct_path, field="registered data freeze")
        closure_path = path
    if manifest.get("manifest_type") != "data_freeze" or manifest.get("status") != "frozen":
        raise PositionFreezeError("position selection requires a frozen data manifest")
    try:
        relative = closure_path.resolve().relative_to(evidence_root.resolve())
    except ValueError as exc:
        raise PositionFreezeError("data freeze escapes evidence root") from exc
    return manifest, {
        "relative_path": relative.as_posix(),
        "size_bytes": direct_path.stat().st_size,
    }


def _portable_plan(position_plan: Mapping[str, Any]) -> dict[str, Any]:
    required = {"component_type", "composition", "positions", "distinct_global"}
    if not required.issubset(position_plan):
        raise PositionFreezeError("position plan is incomplete")
    plan = {
        "component_type": copy.deepcopy(position_plan["component_type"]),
        "composition": copy.deepcopy(position_plan["composition"]),
        "positions": copy.deepcopy(position_plan["positions"]),
        "distinct_global": copy.deepcopy(position_plan["distinct_global"]),
        "position_method": copy.deepcopy(position_plan.get("position_method")),
        "selection_provenance": {
            "source_kind": copy.deepcopy(position_plan.get("source_kind")),
            "source_manifest": copy.deepcopy(
                position_plan.get("source_manifest", {}).get("path")
                if isinstance(position_plan.get("source_manifest"), Mapping)
                else None
            ),
            "selection": copy.deepcopy(position_plan.get("selection")),
            "strategies": copy.deepcopy(position_plan.get("strategies", [])),
        },
    }
    return plan


def freeze_position_plan(
    position_plan: Mapping[str, Any],
    method_plan: Mapping[str, Any],
    *,
    data_freeze_path: Path,
    protocol: Mapping[str, Any],
    evidence_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Create the only position artifact an intervention cell may import."""

    if method_plan.get("method_kind") != "position_baseline":
        raise PositionFreezeError("position freeze requires one position-baseline method")
    data_freeze, data_artifact = _data_freeze_from_reference(data_freeze_path, evidence_root)
    portable = _portable_plan(position_plan)
    manifest = {
        "schema_version": 1,
        "manifest_type": "position_freeze",
        "status": "frozen",
        "selector": {
            "method": _string(method_plan.get("method"), field="method_plan.method"),
            "method_kind": "position_baseline",
        },
        "protocol": copy.deepcopy(dict(protocol)),
        "dataset": data_freeze.get("dataset"),
        "data_freeze": data_artifact,
        "position_plan": portable,
        "artifact_closure": [data_artifact],
    }
    _write_exclusive(output_path, manifest)
    return manifest


def validate_position_freeze(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the immutable identity and return a defensive copy."""

    manifest = copy.deepcopy(dict(value))
    if (
        value.get("schema_version") != 1
        or value.get("manifest_type") != "position_freeze"
        or value.get("status") != "frozen"
    ):
        raise PositionFreezeError("position freeze is incomplete")
    selector = _mapping(value.get("selector"), field="selector")
    if selector.get("method_kind") != "position_baseline":
        raise PositionFreezeError("position freeze selector kind is invalid")
    plan = _mapping(value.get("position_plan"), field="position_plan")
    positions = plan.get("positions")
    if not isinstance(positions, list) or not positions:
        raise PositionFreezeError("frozen position plan has no positions")
    return copy.deepcopy(dict(value))


def select_frozen_candidate(
    value: Mapping[str, Any], *, rank: int
) -> dict[str, Any]:
    """Resolve a one-based candidate rank without reranking a frozen position set."""

    frozen = validate_position_freeze(value)
    positions = frozen["position_plan"]["positions"]
    if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= len(positions):
        raise PositionFreezeError("candidate rank is outside the frozen position set")
    return copy.deepcopy(positions[rank - 1])
