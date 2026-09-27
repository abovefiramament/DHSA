"""Resolve machine-local absolute paths from a portable runtime registry."""

from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping


class RuntimeRegistryError(ValueError):
    """Raised when a machine registry or registry reference is invalid."""


KINDS = frozenset({"file", "directory", "output_directory", "executable"})


def _absolute(path: str) -> bool:
    return PurePosixPath(path).is_absolute() or PureWindowsPath(path).is_absolute()


def load_runtime_registry(path: Path) -> dict[str, Any]:
    """Load and validate one machine-local path registry."""

    registry_path = path.expanduser().resolve()
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeRegistryError(f"cannot load runtime registry {registry_path}: {exc}") from exc
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise RuntimeRegistryError("runtime registry schema_version must be 1")
    machine_id = payload.get("machine_id")
    if not isinstance(machine_id, str) or not machine_id:
        raise RuntimeRegistryError("runtime registry machine_id must be non-empty")
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, Mapping) or not raw_entries:
        raise RuntimeRegistryError("runtime registry entries must be a non-empty object")
    entries: dict[str, dict[str, Any]] = {}
    for key, raw in raw_entries.items():
        if not isinstance(key, str) or not key or key.startswith("/") or key.endswith("/"):
            raise RuntimeRegistryError(f"invalid runtime registry key: {key!r}")
        if not isinstance(raw, Mapping):
            raise RuntimeRegistryError(f"runtime registry entry {key!r} must be an object")
        absolute_path = raw.get("path")
        kind = raw.get("kind")
        must_exist = raw.get("must_exist")
        if not isinstance(absolute_path, str) or not _absolute(absolute_path):
            raise RuntimeRegistryError(f"runtime registry entry {key!r} needs an absolute path")
        if kind not in KINDS:
            raise RuntimeRegistryError(f"runtime registry entry {key!r} has invalid kind")
        if not isinstance(must_exist, bool):
            raise RuntimeRegistryError(
                f"runtime registry entry {key!r}.must_exist must be boolean"
            )
        local = Path(absolute_path)
        if must_exist and not local.exists():
            raise RuntimeRegistryError(
                f"registered path does not exist on this machine: {absolute_path}"
            )
        if must_exist and kind in {"file", "executable"} and not local.is_file():
            raise RuntimeRegistryError(f"registered path is not a file: {absolute_path}")
        if must_exist and kind == "executable" and not os.access(local, os.X_OK):
            raise RuntimeRegistryError(f"registered path is not executable: {absolute_path}")
        if must_exist and kind in {"directory", "output_directory"} and not local.is_dir():
            raise RuntimeRegistryError(f"registered path is not a directory: {absolute_path}")
        entries[key] = {
            "path": absolute_path,
            "kind": kind,
            "must_exist": must_exist,
        }
    return {
        "schema_version": 1,
        "machine_id": machine_id,
        "registry_path": str(registry_path),
        "entries": entries,
    }


def resolve_runtime_paths(
    value: Any,
    registry: Mapping[str, Any],
) -> tuple[Any, dict[str, dict[str, Any]]]:
    """Replace exact registry:// references and return the used entry subset."""

    entries = registry["entries"]
    used: dict[str, dict[str, Any]] = {}

    def resolve(item: Any) -> Any:
        if isinstance(item, str) and item.startswith("registry://"):
            key = item[len("registry://") :]
            if key not in entries:
                raise RuntimeRegistryError(f"unregistered runtime path: {item}")
            used[key] = copy.deepcopy(entries[key])
            return entries[key]["path"]
        if isinstance(item, Mapping):
            return {key: resolve(child) for key, child in item.items()}
        if isinstance(item, list):
            return [resolve(child) for child in item]
        return copy.deepcopy(item)

    return resolve(value), used


def reregister_runtime_entries(
    path: Path,
    updates: Mapping[str, Mapping[str, Any]],
    *,
    machine_id: str | None = None,
) -> dict[str, Any]:
    """Atomically correct machine-local bindings and return the new manifest.

    Re-registration is deliberately limited to the ignored runtime registry. It
    never edits a portable experiment bundle or an existing evidence run. The
    candidate registry is validated before replacement, so a bad path leaves the
    previous registration untouched. Callers must recompile a flow after this
    operation; callers then recompile the local execution plan.
    """

    registry_path = path.expanduser().resolve()
    if not isinstance(updates, Mapping) or not updates:
        raise RuntimeRegistryError("runtime registry updates must be a non-empty object")

    if registry_path.exists():
        try:
            current = json.loads(registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeRegistryError(
                f"cannot load runtime registry {registry_path}: {exc}"
            ) from exc
        if not isinstance(current, Mapping) or current.get("schema_version") != 1:
            raise RuntimeRegistryError("runtime registry schema_version must be 1")
        current_machine_id = current.get("machine_id")
        current_entries = current.get("entries")
        if not isinstance(current_machine_id, str) or not current_machine_id:
            raise RuntimeRegistryError("runtime registry machine_id must be non-empty")
        if not isinstance(current_entries, Mapping):
            raise RuntimeRegistryError("runtime registry entries must be an object")
        candidate_machine_id = machine_id or current_machine_id
        candidate_entries = copy.deepcopy(dict(current_entries))
    else:
        if not isinstance(machine_id, str) or not machine_id:
            raise RuntimeRegistryError(
                "machine_id is required when creating a runtime registry"
            )
        candidate_machine_id = machine_id
        candidate_entries = {}

    for key, value in updates.items():
        if not isinstance(key, str) or not key:
            raise RuntimeRegistryError("runtime registry update keys must be non-empty strings")
        if not isinstance(value, Mapping):
            raise RuntimeRegistryError(f"runtime registry update {key!r} must be an object")
        # Keep this helper strict and explicit: a correction replaces a whole
        # entry, rather than silently merging an old kind/existence policy.
        if set(value) != {"path", "kind", "must_exist"}:
            raise RuntimeRegistryError(
                f"runtime registry update {key!r} must contain path, kind and must_exist"
            )
        candidate_entries[key] = copy.deepcopy(dict(value))

    candidate = {
        "schema_version": 1,
        "machine_id": candidate_machine_id,
        "entries": candidate_entries,
    }
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=registry_path.parent,
            prefix=f".{registry_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(candidate, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        # Validation includes absolute-path and kind/existence checks before the
        # old registry can be replaced.
        load_runtime_registry(temporary_path)
        os.replace(temporary_path, registry_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
    return load_runtime_registry(registry_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create, correct, or validate machine-local path registrations."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    set_parser = subparsers.add_parser(
        "set", help="Create or replace one machine-local path entry."
    )
    set_parser.add_argument("--registry", type=Path, required=True)
    set_parser.add_argument("--machine-id")
    set_parser.add_argument("--key", required=True)
    set_parser.add_argument("--path", dest="absolute_path", required=True)
    set_parser.add_argument("--kind", choices=sorted(KINDS), required=True)
    set_parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Register a path that does not exist yet; formal machines should omit this.",
    )

    validate_parser = subparsers.add_parser(
        "validate", help="Validate one complete machine-local path registry."
    )
    validate_parser.add_argument("--registry", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "set":
        updated = reregister_runtime_entries(
            args.registry,
            {
                args.key: {
                    "path": args.absolute_path,
                    "kind": args.kind,
                    "must_exist": not args.allow_missing,
                }
            },
            machine_id=args.machine_id,
        )
        changed_key = args.key
    else:
        updated = load_runtime_registry(args.registry)
        changed_key = None
    print(
        json.dumps(
            {
                "machine_id": updated["machine_id"],
                "entry_count": len(updated["entries"]),
                "changed_key": changed_key,
                "registry_path": updated["registry_path"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
