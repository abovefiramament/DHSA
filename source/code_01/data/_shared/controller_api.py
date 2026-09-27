"""Shared materialization contract for dataset controllers.

Dataset modules own source parsing and normalization. This module implements
only protocol-supplied range selection, deterministic mixing, disjointness
checks, and manifest helpers. It contains no dataset path or sample-count
defaults.
"""

from __future__ import annotations

import csv
import json
import os
import random
import re
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


SPLIT_NAMES = ("train", "validation", "alpha_dev", "test")
MIXING_MODES = frozenset({"single", "concatenate", "round_robin", "registered_shuffle"})
ROLE_PURPOSES = frozenset(
    {
        "selection",
        "training",
        "training_validation",
        "calibration",
        "reserve",
        "final_test",
        "reference",
    }
)
ROLE_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


class ControllerError(ValueError):
    """Raised when a registered data contract cannot be satisfied."""


def load_registered_rows(source_spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Load one registered canonical source in an explicitly declared format."""

    location = source_spec.get("location")
    if not isinstance(location, str) or not location:
        raise ControllerError("data source location must be a non-empty string")
    path = Path(location).expanduser().resolve()
    if not path.is_file():
        raise ControllerError(f"registered data source does not exist: {path}")
    source_format = source_spec.get("format")
    if source_format == "jsonl":
        with path.open(encoding="utf-8") as handle:
            rows: Any = [json.loads(line) for line in handle if line.strip()]
    elif source_format == "json":
        rows = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(rows, Mapping):
            rows_key = source_spec.get("rows_key")
            if not isinstance(rows_key, str) or not rows_key:
                raise ControllerError("object-form JSON source requires rows_key")
            rows = rows.get(rows_key)
    elif source_format == "csv":
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        raise ControllerError(f"unsupported registered data format: {source_format!r}")
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise ControllerError("registered source must resolve to a list of row objects")
    return [dict(row) for row in rows]


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ControllerError(f"dataset artifact already exists: {path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        temporary.unlink(missing_ok=True)
        raise ControllerError(f"dataset artifact already exists: {path}") from exc
    os.close(descriptor)
    os.replace(temporary, path)


def _write_jsonl_exclusive(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    except FileExistsError as exc:
        raise ControllerError(f"dataset artifact already exists: {path}") from exc


def execute_dataset_controller(
    *,
    data_spec: Mapping[str, Any],
    output_dir: Path,
    load_source: Callable[[Mapping[str, Any]], Sequence[Mapping[str, Any]]],
    materialize: Callable[..., tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]],
) -> dict[str, Any]:
    """Load, validate, split, and register one dataset through a fixed adapter."""

    source_specs = data_spec.get("sources")
    if not isinstance(source_specs, Mapping) or not source_specs:
        raise ControllerError("data.sources must be a non-empty object")
    source_rows = {
        name: list(load_source(spec))
        for name, spec in source_specs.items()
        if isinstance(name, str) and isinstance(spec, Mapping)
    }
    if set(source_rows) != set(source_specs):
        raise ControllerError("every registered data source must have an object specification")
    roles, manifest = materialize(source_rows, source_specs, data_spec)
    role_specs, declared_named_roles = _named_roles(data_spec)
    formal_roles = manifest.get("role_mode") == "named_roles"
    if formal_roles != declared_named_roles:
        raise ControllerError("dataset controller role_mode must match declared data roles")
    if set(roles) != set(role_specs):
        raise ControllerError(
            "dataset controller emitted roles different from declared data.roles: "
            f"expected={sorted(role_specs)} actual={sorted(roles)}"
        )
    artifact_dir = "roles" if formal_roles else "splits"
    role_artifacts: dict[str, Any] = {}
    for role_name, rows in roles.items():
        path = output_dir / artifact_dir / f"{role_name}.jsonl"
        _write_jsonl_exclusive(path, rows)
        role_artifacts[role_name] = {
            "relative_path": path.relative_to(output_dir).as_posix(),
            "size_bytes": path.stat().st_size,
            "rows": len(rows),
        }
    registered = {
        **manifest,
        "manifest_type": "data_freeze",
        "status": "frozen",
        "role_artifacts": role_artifacts,
        "role_purposes": {
            name: str(spec["purpose"])
            for name, spec in role_specs.items()
        },
    }
    if not formal_roles:
        registered["split_artifacts"] = role_artifacts
    _write_json_atomic(output_dir / "dataset_manifest.json", registered)
    return registered


def require_keys(row: Mapping[str, Any], keys: Iterable[str], *, context: str) -> None:
    missing = [key for key in keys if key not in row]
    if missing:
        raise ControllerError(f"{context} is missing required fields: {missing}")


def _integer(value: Any, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ControllerError(f"{field} must be a non-negative integer")
    return value


def _parts(role_name: str, role_spec: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    parts = role_spec.get("parts")
    if not isinstance(parts, list) or not parts:
        raise ControllerError(f"data role {role_name!r}.parts must be non-empty")
    if not all(isinstance(part, Mapping) for part in parts):
        raise ControllerError(f"data role {role_name!r}.parts must contain objects")
    return parts


def validate_range_disjointness(splits: Mapping[str, Mapping[str, Any]]) -> None:
    """Verify that registered source-order intervals do not cross split roles."""

    intervals: dict[str, list[tuple[int, int, str]]] = {}
    for split_name, split_spec in splits.items():
        for index, part in enumerate(_parts(split_name, split_spec)):
            source = part.get("source")
            if not isinstance(source, str) or not source:
                raise ControllerError(
                    f"data.splits.{split_name}.parts[{index}].source must be explicit"
                )
            start = _integer(part.get("start"), field=f"{split_name}.parts[{index}].start")
            stop = _integer(part.get("stop"), field=f"{split_name}.parts[{index}].stop")
            if stop <= start:
                raise ControllerError(f"{split_name}.parts[{index}] has an empty range")
            for old_start, old_stop, old_split in intervals.setdefault(source, []):
                if max(start, old_start) < min(stop, old_stop):
                    raise ControllerError(
                        f"source {source!r} overlaps {old_split} and {split_name}"
                    )
            intervals[source].append((start, stop, split_name))


def _named_roles(data_spec: Mapping[str, Any]) -> tuple[Mapping[str, Any], bool]:
    roles = data_spec.get("roles")
    if roles is None:
        splits = data_spec.get("splits")
        if not isinstance(splits, Mapping) or set(splits) != set(SPLIT_NAMES):
            raise ControllerError(f"data.splits must define exactly {list(SPLIT_NAMES)}")
        return splits, False
    if "splits" in data_spec:
        raise ControllerError("use either data.roles or legacy data.splits, not both")
    if not isinstance(roles, Mapping) or not roles:
        raise ControllerError("data.roles must be a non-empty object")
    for role_name, role_spec in roles.items():
        if not isinstance(role_name, str) or ROLE_NAME.fullmatch(role_name) is None:
            raise ControllerError(f"invalid data role name: {role_name!r}")
        if not isinstance(role_spec, Mapping):
            raise ControllerError(f"data.roles.{role_name} must be an object")
        purpose = role_spec.get("purpose")
        if purpose not in ROLE_PURPOSES:
            raise ControllerError(
                f"data.roles.{role_name}.purpose must be one of {sorted(ROLE_PURPOSES)}"
            )
    return roles, True


def _overlap_policy(
    role_specs: Mapping[str, Mapping[str, Any]],
    data_spec: Mapping[str, Any],
    *,
    formal_roles: bool,
) -> tuple[set[frozenset[str]], dict[str, Any]]:
    raw = data_spec.get("overlap_policy")
    if not formal_roles:
        if raw is not None:
            raise ControllerError("legacy data.splits cannot declare overlap_policy")
        return set(), {"default": "forbid", "allow": []}
    if not isinstance(raw, Mapping):
        raise ControllerError("named data roles require data.overlap_policy")
    if set(raw) != {"default", "allow"}:
        raise ControllerError("overlap_policy must contain exactly default and allow")
    if raw.get("default") != "forbid":
        raise ControllerError("overlap_policy.default must be forbid")
    allow = raw.get("allow")
    if not isinstance(allow, list):
        raise ControllerError("overlap_policy.allow must be a list")
    allowed: set[frozenset[str]] = set()
    normalized: list[list[str]] = []
    for index, item in enumerate(allow):
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not all(isinstance(role, str) for role in item)
        ):
            raise ControllerError(
                f"overlap_policy.allow[{index}] must contain two role names"
            )
        left, right = item
        if left == right or left not in role_specs or right not in role_specs:
            raise ControllerError(f"invalid allowed overlap pair: {item}")
        pair = frozenset(item)
        if pair in allowed:
            raise ControllerError(f"duplicate allowed overlap pair: {item}")
        allowed.add(pair)
        normalized.append(sorted(item))
    return allowed, {"default": "forbid", "allow": sorted(normalized)}


def validate_role_overlaps(
    role_specs: Mapping[str, Mapping[str, Any]],
    data_spec: Mapping[str, Any],
    *,
    formal_roles: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Reject every source-range overlap not explicitly admitted by protocol."""

    allowed, normalized = _overlap_policy(
        role_specs, data_spec, formal_roles=formal_roles
    )
    intervals: dict[str, list[tuple[int, int, str]]] = {}
    admitted: list[dict[str, Any]] = []
    for role_name, role_spec in role_specs.items():
        for index, part in enumerate(_parts(role_name, role_spec)):
            source = part.get("source")
            if not isinstance(source, str) or not source:
                raise ControllerError(
                    f"data role {role_name!r}.parts[{index}].source must be explicit"
                )
            start = _integer(part.get("start"), field=f"{role_name}.parts[{index}].start")
            stop = _integer(part.get("stop"), field=f"{role_name}.parts[{index}].stop")
            if stop <= start:
                raise ControllerError(f"{role_name}.parts[{index}] has an empty range")
            for old_start, old_stop, old_role in intervals.setdefault(source, []):
                overlap_start = max(start, old_start)
                overlap_stop = min(stop, old_stop)
                if overlap_start >= overlap_stop:
                    continue
                if old_role == role_name:
                    raise ControllerError(
                        f"data role {role_name!r} duplicates source range on {source!r}"
                    )
                pair = frozenset((old_role, role_name))
                if pair not in allowed:
                    raise ControllerError(
                        f"source {source!r} overlaps roles {old_role!r} and {role_name!r}"
                    )
                admitted.append(
                    {
                        "source": source,
                        "roles": sorted((old_role, role_name)),
                        "start": overlap_start,
                        "stop": overlap_stop,
                    }
                )
            intervals[source].append((start, stop, role_name))
    return normalized, admitted


def _mix(
    blocks: Sequence[list[dict[str, Any]]],
    mixing: Mapping[str, Any],
    *,
    split_name: str,
) -> list[dict[str, Any]]:
    mode = mixing.get("mode")
    if mode not in MIXING_MODES:
        raise ControllerError(f"unsupported mixing mode for {split_name}: {mode!r}")
    if mode == "single":
        if len(blocks) != 1:
            raise ControllerError(f"{split_name} mixing.mode=single requires one source part")
        return list(blocks[0])
    if mode == "concatenate":
        return [row for block in blocks for row in block]
    if mode == "round_robin":
        output: list[dict[str, Any]] = []
        width = max((len(block) for block in blocks), default=0)
        for offset in range(width):
            output.extend(block[offset] for block in blocks if offset < len(block))
        return output
    shuffle = mixing.get("shuffle")
    if not isinstance(shuffle, Mapping):
        raise ControllerError(f"{split_name} registered_shuffle requires shuffle settings")
    if shuffle.get("algorithm") != "python_random_v1":
        raise ControllerError(f"{split_name} has an unregistered shuffle algorithm")
    seed = _integer(shuffle.get("seed"), field=f"{split_name}.mixing.shuffle.seed")
    output = [row for block in blocks for row in block]
    random.Random(seed).shuffle(output)
    return output


def materialize_registered_roles(
    source_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    data_spec: Mapping[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Materialize protocol-named roles with explicit overlap semantics."""

    roles, formal_roles = _named_roles(data_spec)
    overlap_policy, admitted_overlaps = validate_role_overlaps(
        roles, data_spec, formal_roles=formal_roles
    )
    output: dict[str, list[dict[str, Any]]] = {}
    scaled_counts = data_spec.get("scaled_counts")
    if scaled_counts is not None:
        if data_spec.get("execution_profile") != "formal_scaled_gpu":
            raise ControllerError("scaled_counts requires execution_profile=formal_scaled_gpu")
        if not isinstance(scaled_counts, Mapping):
            raise ControllerError("data.scaled_counts must be an object")
        if not all(isinstance(name, str) and isinstance(count, int) and count > 0 for name, count in scaled_counts.items()):
            raise ControllerError("data.scaled_counts values must be positive integers")
    part_counts: dict[str, list[dict[str, Any]]] = {}
    for role_name, role_spec in roles.items():
        parts = _parts(role_name, role_spec)
        blocks: list[list[dict[str, Any]]] = []
        counts: list[dict[str, Any]] = []
        for index, part in enumerate(parts):
            source = str(part["source"])
            if source not in source_rows:
                raise ControllerError(f"unloaded source {source!r} in {role_name}")
            start = int(part["start"])
            stop = int(part["stop"])
            rows = source_rows[source]
            if stop > len(rows):
                raise ControllerError(
                    f"{role_name} range [{start},{stop}) exceeds source {source!r}"
                )
            block = [
                {
                    "source_name": source,
                    "data_role": role_name,
                    **({"split": role_name} if not formal_roles else {}),
                    **dict(row),
                }
                for row in rows[start:stop]
            ]
            requested = _integer(
                part.get("requested_rows"),
                field=f"{role_name}.parts[{index}].requested_rows",
            )
            if len(block) != requested:
                raise ControllerError(
                    f"{role_name} part {source!r} expected {requested} rows, got {len(block)}"
                )
            blocks.append(block)
            counts.append({"source": source, "rows": len(block), "start": start, "stop": stop})
        mixing = role_spec.get("mixing")
        if not isinstance(mixing, Mapping):
            raise ControllerError(f"data role {role_name!r}.mixing must be explicit")
        output[role_name] = _mix(blocks, mixing, split_name=role_name)
        if scaled_counts is not None and role_name in scaled_counts:
            limit = int(scaled_counts[role_name])
            if limit > len(output[role_name]):
                raise ControllerError(
                    f"scaled count for {role_name!r} exceeds formal role size"
                )
            output[role_name] = output[role_name][:limit]
        part_counts[role_name] = counts

    manifest = {
        "role_mode": "named_roles" if formal_roles else "legacy_splits",
        "role_counts": {name: len(rows) for name, rows in output.items()},
        "role_purposes": {
            name: role["purpose"] for name, role in roles.items()
        }
        if formal_roles
        else {
            "train": "training",
            "validation": "training_validation",
            "alpha_dev": "calibration",
            "test": "final_test",
        },
        "source_parts": part_counts,
        "mixing": {name: dict(role["mixing"]) for name, role in roles.items()},
        "overlap_policy": overlap_policy,
        "admitted_overlaps": admitted_overlaps,
        "construction_rule": "registered_source_order_ranges_then_registered_mixing",
    }
    if scaled_counts is not None:
        manifest["scaled_counts"] = {name: len(rows) for name, rows in output.items()}
        manifest["scaled_from"] = "formal_protocol_role_construction"
        manifest["scaled_gpu_required"] = True
    if not formal_roles:
        manifest["split_counts"] = dict(manifest["role_counts"])
    return output, manifest


def materialize_registered_splits(
    source_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    data_spec: Mapping[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Backward-compatible name for the generalized data-role controller."""

    return materialize_registered_roles(source_rows, data_spec)


def write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
