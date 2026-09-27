"""Human-audit request and approval records bound by concrete identities."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping


class ManualAuditError(ValueError):
    pass


def _load(path: Path, *, field: str) -> dict[str, Any]:
    if not path.is_file():
        raise ManualAuditError(f"{field} does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ManualAuditError(f"{field} must contain one object")
    return value


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise ManualAuditError(f"audit artifact already exists: {path}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(rendered)


def _file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ManualAuditError(f"audit input does not exist: {path}")
    return {"name": path.name, "size_bytes": path.stat().st_size}


def create_audit_request(
    *,
    gate_id: str,
    candidate_manifest: Path,
    protocol: Mapping[str, Any],
    decision_schema: str,
    output_path: Path,
) -> dict[str, Any]:
    request = {
        "schema_version": 1,
        "manifest_type": "audit_request",
        "status": "awaiting_approval",
        "gate_id": gate_id,
        "decision_schema": decision_schema,
        "protocol": dict(protocol),
        "candidate_manifest": _file(candidate_manifest),
    }
    _write_exclusive(output_path, request)
    return request


def approve_audit(
    *,
    request_path: Path,
    candidate_manifest: Path,
    decision_manifest: Path,
    auditor_id: str,
    output_path: Path,
) -> dict[str, Any]:
    request = _load(request_path, field="audit request")
    if request.get("status") != "awaiting_approval":
        raise ManualAuditError("audit request is not awaiting approval")
    if request.get("candidate_manifest") != _file(candidate_manifest):
        raise ManualAuditError("audit candidate file does not match the request")
    approval = {
        "schema_version": 1,
        "manifest_type": "audit_approval",
        "status": "approved",
        "gate_id": request.get("gate_id"),
        "auditor_id": auditor_id,
        "decision_schema": request.get("decision_schema"),
        "candidate_manifest": _file(candidate_manifest),
        "decision_manifest": _file(decision_manifest),
    }
    _write_exclusive(output_path, approval)
    return approval


def validate_audit_approval(
    approval_path: Path,
    *,
    request_path: Path,
    candidate_manifest: Path,
    decision_manifest: Path,
) -> dict[str, Any]:
    request = _load(request_path, field="audit request")
    approval = _load(approval_path, field="audit approval")
    if approval.get("manifest_type") != "audit_approval" or approval.get("status") != "approved":
        raise ManualAuditError("audit approval is invalid")
    if approval.get("gate_id") != request.get("gate_id"):
        raise ManualAuditError("audit approval belongs to another gate")
    if approval.get("candidate_manifest") != _file(candidate_manifest):
        raise ManualAuditError("audit approval references another candidate file")
    if approval.get("decision_manifest") != _file(decision_manifest):
        raise ManualAuditError("audit approval references another decision file")
    return approval
