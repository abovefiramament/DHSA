"""Small resumable run ledger using concrete artifact values only."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence


class EvidenceRegistryError(ValueError):
    pass


def _path(root: Path) -> Path:
    return root / "run_manifest.json"


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise EvidenceRegistryError(f"artifact escapes run root: {resolved}") from exc
    if not resolved.is_file():
        raise EvidenceRegistryError(f"artifact does not exist: {resolved}")
    return {"relative_path": relative.as_posix(), "size_bytes": resolved.stat().st_size}


def load_run(cell_root: Path) -> dict[str, Any]:
    path = _path(cell_root)
    if not path.is_file():
        raise EvidenceRegistryError(f"run manifest does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise EvidenceRegistryError("run manifest is invalid")
    return value


def allocate_run(
    cell_root: Path,
    *,
    identity: Mapping[str, Any],
    stage_ids: Sequence[str],
    allowed_existing_names: Sequence[str] = (),
) -> dict[str, Any]:
    cell_root.mkdir(parents=True, exist_ok=True)
    unexpected = {
        item.name for item in cell_root.iterdir()
        if item.name not in set(allowed_existing_names)
    }
    if unexpected:
        raise EvidenceRegistryError(f"run root is not empty: {sorted(unexpected)}")
    if not stage_ids or len(stage_ids) != len(set(stage_ids)):
        raise EvidenceRegistryError("stage ids must be a non-empty unique sequence")
    run = {
        "schema_version": 1,
        "status": "allocated",
        "identity": dict(identity),
        "stages": {
            stage_id: {
                "status": "pending",
                "attempts": [],
                "current_attempt": None,
                "outputs": {},
            }
            for stage_id in stage_ids
        },
        "audit_gates": {},
    }
    _write(_path(cell_root), run)
    return run


def start_stage(cell_root: Path, *, stage_id: str, input_files: Mapping[str, Path]) -> str:
    run = load_run(cell_root)
    stage = run["stages"].get(stage_id)
    if not isinstance(stage, dict) or stage["status"] not in {"pending", "failed"}:
        raise EvidenceRegistryError(f"stage cannot start: {stage_id}")
    attempt = f"attempt-{len(stage['attempts']) + 1}"
    stage["status"] = "running"
    stage["current_attempt"] = attempt
    stage["attempts"].append(
        {
            "attempt_id": attempt,
            "status": "running",
            "inputs": {name: _artifact(path, cell_root) for name, path in input_files.items()},
        }
    )
    run["status"] = "running"
    _write(_path(cell_root), run)
    return attempt


def complete_stage(
    cell_root: Path,
    *,
    stage_id: str,
    attempt_id: str,
    output_files: Mapping[str, Path],
) -> dict[str, Any]:
    run = load_run(cell_root)
    stage = run["stages"][stage_id]
    if stage["status"] != "running" or stage["current_attempt"] != attempt_id:
        raise EvidenceRegistryError("stage attempt is not active")
    outputs = {name: _artifact(path, cell_root) for name, path in output_files.items()}
    stage["status"] = "completed"
    stage["outputs"] = outputs
    stage["attempts"][-1].update({"status": "completed", "outputs": outputs})
    run["status"] = "running"
    _write(_path(cell_root), run)
    return run


def complete_imported_stage(
    cell_root: Path,
    *,
    stage_id: str,
    attempt_id: str,
    output_files: Mapping[str, Path],
    evidence_id: str,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Complete one stage from exact external evidence and record its provenance."""

    if not isinstance(evidence_id, str) or not evidence_id:
        raise EvidenceRegistryError("imported stage evidence_id must be non-empty")
    run = complete_stage(
        cell_root,
        stage_id=stage_id,
        attempt_id=attempt_id,
        output_files=output_files,
    )
    stage = run["stages"][stage_id]
    provenance = {
        "mode": "external_exact_stage_reuse",
        "evidence_id": evidence_id,
        "source": dict(source),
    }
    stage["imported_evidence"] = provenance
    stage["attempts"][-1]["imported_evidence"] = provenance
    _write(_path(cell_root), run)
    return run


def fail_stage(
    cell_root: Path,
    *,
    stage_id: str,
    attempt_id: str,
    error_type: str,
    message: str,
) -> dict[str, Any]:
    run = load_run(cell_root)
    stage = run["stages"][stage_id]
    if stage["current_attempt"] != attempt_id:
        raise EvidenceRegistryError("stage attempt is not active")
    stage["status"] = "failed"
    stage["attempts"][-1].update(
        {"status": "failed", "error_type": error_type, "message": message}
    )
    run["status"] = "failed"
    _write(_path(cell_root), run)
    return run


def await_audit(cell_root: Path, *, gate_id: str, request_path: Path) -> dict[str, Any]:
    run = load_run(cell_root)
    run["audit_gates"][gate_id] = {
        "status": "awaiting",
        "request": _artifact(request_path, cell_root),
    }
    run["status"] = "awaiting_audit"
    _write(_path(cell_root), run)
    return run


def register_audit_approval(
    cell_root: Path, *, gate_id: str, approval_path: Path
) -> dict[str, Any]:
    run = load_run(cell_root)
    gate = run["audit_gates"].get(gate_id)
    if not isinstance(gate, dict) or gate.get("status") != "awaiting":
        raise EvidenceRegistryError(f"audit gate is not awaiting: {gate_id}")
    gate.update({"status": "approved", "approval": _artifact(approval_path, cell_root)})
    run["status"] = "approved"
    _write(_path(cell_root), run)
    return run


def freeze_run(cell_root: Path, *, required_stage_ids: Sequence[str]) -> dict[str, Any]:
    run = load_run(cell_root)
    missing = [
        stage_id for stage_id in required_stage_ids
        if run["stages"].get(stage_id, {}).get("status") != "completed"
    ]
    if missing:
        raise EvidenceRegistryError(f"cannot freeze incomplete stages: {missing}")
    run["status"] = "frozen"
    run["evidence"] = {
        stage_id: dict(run["stages"][stage_id]["outputs"])
        for stage_id in required_stage_ids
    }
    _write(_path(cell_root), run)
    return run
