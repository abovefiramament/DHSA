"""Bind formal cells to directly inspectable software environments."""

from __future__ import annotations

import fnmatch
import json
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .runtime_registry import load_runtime_registry


DEFAULT_PROFILE_REGISTRY = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "runtime"
    / "evidence_runtime_profiles_20260907_v1.json"
)


class RuntimeEnvironmentError(RuntimeError):
    """Raised when a formal cell is launched in the wrong software environment."""


def current_runtime_values() -> dict[str, Any]:
    import torch
    import transformers

    return {
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "torch": str(torch.__version__),
        "transformers": str(transformers.__version__),
        "cuda_runtime": str(torch.version.cuda),
        "cudnn": int(torch.backends.cudnn.version()) if torch.backends.cudnn.version() else None,
    }


def _load_profiles(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or value.get("schema_version") != 1:
        raise RuntimeEnvironmentError("runtime environment profile registry must use schema_version 1")
    return value


def _binding(cell_id: str, registry: Mapping[str, Any]) -> Mapping[str, Any] | None:
    bindings = registry.get("cell_bindings", [])
    if not isinstance(bindings, list):
        raise RuntimeEnvironmentError("runtime environment cell_bindings must be a list")
    for item in bindings:
        if isinstance(item, Mapping) and isinstance(item.get("pattern"), str):
            if fnmatch.fnmatchcase(cell_id, str(item["pattern"])):
                return item
    return None


def validate_evidence_cell_bindings(
    profile_registry: Path = DEFAULT_PROFILE_REGISTRY,
) -> dict[str, Any]:
    """Verify that every declared evidence cell has one usable profile binding."""
    registry = _load_profiles(profile_registry)
    profiles = registry.get("profiles", {})
    bindings = registry.get("evidence_cell_bindings", [])
    if not isinstance(profiles, Mapping) or not isinstance(bindings, list):
        raise RuntimeEnvironmentError("invalid evidence runtime binding registry")
    seen: set[str] = set()
    statuses: dict[str, int] = {}
    for item in bindings:
        if not isinstance(item, Mapping) or not isinstance(item.get("evidence_cell_id"), str):
            raise RuntimeEnvironmentError("each evidence binding requires evidence_cell_id")
        cell_id = str(item["evidence_cell_id"])
        if cell_id in seen:
            raise RuntimeEnvironmentError(f"duplicate evidence runtime binding: {cell_id}")
        seen.add(cell_id)
        profile_id = item.get("profile_id")
        profile = profiles.get(profile_id) if isinstance(profile_id, str) else None
        if not isinstance(profile, Mapping):
            raise RuntimeEnvironmentError(f"unknown runtime profile for evidence cell {cell_id}")
        status = str(profile.get("status", "missing_status"))
        statuses[status] = statuses.get(status, 0) + 1
    return {"cell_count": len(seen), "status_counts": statuses}


def validate_runtime_environment(
    *,
    cell_id: str,
    runtime_registry: Path,
    profile_registry: Path = DEFAULT_PROFILE_REGISTRY,
) -> dict[str, Any]:
    profiles = _load_profiles(profile_registry)
    binding = _binding(cell_id, profiles)
    actual = current_runtime_values()
    if binding is None:
        return {"cell_id": cell_id, "status": "not_bound", "actual": actual}
    profile_id = binding.get("profile_id")
    profile = profiles.get("profiles", {}).get(profile_id)
    if not isinstance(profile_id, str) or not isinstance(profile, Mapping):
        raise RuntimeEnvironmentError(f"invalid runtime profile binding for {cell_id}")
    if profile.get("enforce") is not True:
        return {
            "cell_id": cell_id,
            "profile_id": profile_id,
            "status": "recorded_not_enforced",
            "actual": actual,
        }

    machine = load_runtime_registry(runtime_registry)
    python_key = profile.get("python_registry_key")
    entry = machine["entries"].get(python_key) if isinstance(python_key, str) else None
    if not isinstance(entry, Mapping) or entry.get("kind") != "executable":
        raise RuntimeEnvironmentError(
            f"{cell_id} requires executable machine registration {python_key!r}"
        )
    expected_executable = str(Path(str(entry["path"])).resolve())
    if actual["python_executable"] != expected_executable:
        raise RuntimeEnvironmentError(
            f"{cell_id} must run with registered Python {entry['path']}; "
            f"current interpreter is {sys.executable}"
        )
    expected = profile.get("expected")
    if not isinstance(expected, Mapping):
        raise RuntimeEnvironmentError(f"runtime profile {profile_id} lacks expected values")
    mismatches = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise RuntimeEnvironmentError(
            f"runtime environment mismatch for {cell_id}: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )
    return {
        "cell_id": cell_id,
        "profile_id": profile_id,
        "status": "verified",
        "expected": dict(expected),
        "actual": actual,
    }


def validate_and_record_runtime_environment(
    *, cell_id: str, runtime_registry: Path, cell_root: Path
) -> dict[str, Any]:
    result = validate_runtime_environment(
        cell_id=cell_id,
        runtime_registry=runtime_registry,
    )
    destination = cell_root / "config" / "runtime_environment.local.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent,
            prefix=f".{destination.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return result
