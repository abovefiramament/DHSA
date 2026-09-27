"""Onboard, register, and adopt external experiment evidence by real values.

This module never executes legacy code.  It packages already produced files,
records inspectable scientific stage conditions, and lets the current typed
flow import only a contiguous compatible stage prefix.  Machine-local paths
remain in the runtime registry; the portable catalog contains no such paths.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ELIGIBILITIES = frozenset(
    {"exact_cell", "exact_stage", "comparable_external", "attested_legacy"}
)
REUSE_LEVELS = ("position", "candidate_bank", "bank", "alpha", "test")
MACHINE_KEYS = frozenset(
    {
        "absolute_path",
        "local_path",
        "base_local_path",
        "evidence_root",
        "registry_path",
        "source_directory",
    }
)
IDENTITY_KEYS = frozenset(
    {"protocol_job_id", "job_id", "cell_id", "bundle_id"}
)
EXECUTION_ONLY_KEYS = frozenset(
    {"external_position_source", "position_import", "evidence_reuse", "shared_scan"}
)


class ExternalEvidenceError(ValueError):
    """Raised when external evidence is incomplete, ambiguous, or incompatible."""


def _load(path: Path, *, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalEvidenceError(f"cannot read {field}: {path}") from exc
    if not isinstance(value, dict):
        raise ExternalEvidenceError(f"{field} must contain one JSON object")
    return value


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise ExternalEvidenceError(f"immutable evidence artifact exists: {path}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(rendered)


def _is_machine_path(value: str) -> bool:
    return (
        value.startswith("/")
        or (len(value) > 2 and value[1] == ":" and value[2] in {"/", "\\"})
    ) and not value.startswith(("http://", "https://"))


def normalize_real_value(value: Any) -> Any:
    """Remove machine/identity plumbing while retaining scientific values."""

    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for raw_key, child in value.items():
            key = str(raw_key)
            if key in MACHINE_KEYS | IDENTITY_KEYS | EXECUTION_ONLY_KEYS:
                continue
            if isinstance(child, str) and _is_machine_path(child):
                continue
            output[key] = normalize_real_value(child)
        return output
    if isinstance(value, list):
        return [normalize_real_value(child) for child in value]
    return copy.deepcopy(value)


def stage_conditions(stage: Mapping[str, Any]) -> dict[str, Any]:
    """Return the portable scientific inputs for one compiled stage."""

    inputs = stage.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ExternalEvidenceError("compiled stage inputs are missing")
    normalized: dict[str, Any] = {}
    for name, raw in inputs.items():
        if not isinstance(raw, Mapping):
            raise ExternalEvidenceError(f"compiled input {name!r} is invalid")
        kind = raw.get("kind")
        if kind == "config":
            normalized[str(name)] = {
                "kind": "config",
                "reference": raw.get("reference"),
                "value": normalize_real_value(raw.get("value")),
            }
        elif kind == "artifact":
            normalized[str(name)] = {
                "kind": "artifact",
                "reference": raw.get("reference"),
                "producer_handler_id": raw.get("producer_handler_id"),
                "relative_path": raw.get("relative_path"),
            }
        else:
            raise ExternalEvidenceError(f"compiled input {name!r} has unknown kind")
    return {
        "handler_id": stage.get("handler_id"),
        "inputs": normalized,
    }


def _comparison_conditions(value: Mapping[str, Any]) -> dict[str, Any]:
    """Canonicalize backward-compatible values used only for evidence reuse."""

    output = normalize_real_value(value)

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            scanner = node.get("scanner_binding")
            if isinstance(scanner, dict) and isinstance(scanner.get("value"), dict):
                scanner = scanner["value"]
            scan = node.get("scan_config")
            if isinstance(scan, dict) and isinstance(scan.get("value"), dict):
                scan = scan["value"]
            if (
                isinstance(scanner, dict)
                and scanner.get("id") in {"rcm_zero_scanner", "rcm_patch_scanner"}
                and isinstance(scan, dict)
            ):
                # Earlier framework cells executed this exact default without serializing it.
                scan.setdefault("score_mode", "answer_rest_margin")
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(output)
    return output


def _source_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    required = ("repository_url", "revision", "repository_path", "contributor")
    if not all(isinstance(value.get(name), str) and value.get(name) for name in required):
        raise ExternalEvidenceError(
            "source metadata requires repository_url, revision, repository_path and contributor"
        )
    if not str(value["repository_url"]).startswith(("https://", "http://")):
        raise ExternalEvidenceError("source.repository_url must be public HTTP(S)")
    return {name: str(value[name]) for name in required}


def _relative_inside(path: Path, root: Path, *, field: str) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ExternalEvidenceError(f"{field} escapes its source root: {path}") from exc


def _referenced_paths(path: Path, root: Path) -> set[Path]:
    """Collect concrete relative_path references recursively from JSON manifests."""

    found: set[Path] = set()
    if path.suffix.lower() != ".json":
        return found
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return found

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            relative = item.get("relative_path")
            if isinstance(relative, str) and relative:
                candidate = (root / relative).resolve()
                try:
                    candidate.relative_to(root.resolve())
                except ValueError:
                    pass
                else:
                    if candidate.is_file():
                        found.add(candidate)
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return found


def _artifact_closure(
    paths: Iterable[Path], root: Path, *, stop_paths: Iterable[Path] = ()
) -> list[Path]:
    pending = list(paths)
    seen: set[Path] = set()
    stopped = {path.resolve() for path in stop_paths}
    while pending:
        path = pending.pop().resolve()
        if path in stopped:
            continue
        relative = _relative_inside(path, root, field="stage artifact")
        if relative == "data" or relative.startswith("data/"):
            continue
        if path in seen:
            continue
        if not path.is_file():
            raise ExternalEvidenceError(f"stage artifact is missing: {path}")
        seen.add(path)
        pending.extend(_referenced_paths(path, root) - seen)
    return sorted(seen, key=lambda item: _relative_inside(item, root, field="artifact"))


def _input_artifact_paths(inputs: Any, root: Path) -> set[Path]:
    if not isinstance(inputs, Mapping):
        return set()
    paths: set[Path] = set()
    for raw in inputs.values():
        if not isinstance(raw, Mapping) or raw.get("kind") != "artifact":
            continue
        relative = raw.get("relative_path")
        if isinstance(relative, str) and relative:
            candidate = (root / relative).resolve()
            try:
                candidate.relative_to(root.resolve())
            except ValueError:
                continue
            if candidate.is_file():
                paths.add(candidate)
    return paths


def _copy_payload(source_root: Path, package_root: Path, paths: Sequence[Path]) -> list[str]:
    relatives: list[str] = []
    for source in paths:
        relative = _relative_inside(source, source_root, field="payload")
        destination = package_root / "payload" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        relatives.append(relative)
    return relatives


def _copy_payload_mapping(
    source_root: Path,
    package_root: Path,
    mappings: Sequence[tuple[Path, str]],
) -> list[str]:
    relatives: list[str] = []
    payload_root = (package_root / "payload").resolve()
    for source, target_relative in mappings:
        _relative_inside(source, source_root, field="mapped payload source")
        destination = (payload_root / target_relative).resolve()
        try:
            destination.relative_to(payload_root)
        except ValueError as exc:
            raise ExternalEvidenceError(
                f"mapped payload target escapes its package: {target_relative}"
            ) from exc
        if target_relative == "data" or target_relative.startswith("data/"):
            raise ExternalEvidenceError("external evidence cannot package the data stage")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if not destination.is_file() or destination.read_bytes() != source.read_bytes():
                raise ExternalEvidenceError(
                    f"two external files map to different bytes at {target_relative}"
                )
        else:
            shutil.copy2(source, destination)
        relatives.append(target_relative)
    return sorted(set(relatives))


def onboard_framework_cell(
    *,
    source_cell_root: Path,
    source: Mapping[str, Any],
    evidence_id: str,
    output_dir: Path,
    target_cell_id: str | None = None,
    through_stage: str | None = None,
) -> dict[str, Any]:
    """Package completed stages from one new-framework cell without rerunning it."""

    cell_root = source_cell_root.expanduser().resolve()
    run = _load(cell_root / "run_manifest.json", field="source run manifest")
    flow = _load(
        cell_root / "config" / "runtime_flow.local.json", field="source runtime flow"
    )
    if not isinstance(evidence_id, str) or not evidence_id:
        raise ExternalEvidenceError("evidence_id must be non-empty")
    package = output_dir.expanduser().resolve()
    package.mkdir(parents=True, exist_ok=False)
    run_stages = run.get("stages")
    flow_stages = flow.get("stages")
    if not isinstance(run_stages, Mapping) or not isinstance(flow_stages, list):
        raise ExternalEvidenceError("source run/flow stages are invalid")
    stages: dict[str, Any] = {}
    completed_non_data: list[str] = []
    selected_stages = flow_stages
    if through_stage is not None:
        matches = [i for i, stage in enumerate(flow_stages) if stage.get("stage_id") == through_stage]
        if len(matches) != 1:
            raise ExternalEvidenceError(f"unknown or ambiguous prefix stage: {through_stage}")
        selected_stages = flow_stages[:matches[0] + 1]
        if any(run_stages.get(stage["stage_id"], {}).get("status") != "completed" for stage in selected_stages):
            raise ExternalEvidenceError("requested evidence prefix is not fully completed")
    for stage in selected_stages:
        stage_id = stage.get("stage_id")
        if not isinstance(stage_id, str):
            raise ExternalEvidenceError("source stage_id is invalid")
        state = run_stages.get(stage_id)
        if not isinstance(state, Mapping) or state.get("status") != "completed":
            continue
        if stage_id == "data":
            continue
        raw_outputs = state.get("outputs")
        if not isinstance(raw_outputs, Mapping) or not raw_outputs:
            raise ExternalEvidenceError(f"completed stage lacks outputs: {stage_id}")
        output_paths: dict[str, Path] = {}
        for name, artifact in raw_outputs.items():
            relative = artifact.get("relative_path") if isinstance(artifact, Mapping) else None
            if not isinstance(relative, str) or not relative:
                raise ExternalEvidenceError(f"stage output is invalid: {stage_id}.{name}")
            output_paths[str(name)] = (cell_root / relative).resolve()
        closure = _artifact_closure(
            output_paths.values(),
            cell_root,
            stop_paths=_input_artifact_paths(stage.get("inputs"), cell_root),
        )
        payload_files = _copy_payload(cell_root, package, closure)
        stages[stage_id] = {
            "conditions": stage_conditions(stage),
            "outputs": {
                name: _relative_inside(path, cell_root, field="stage output")
                for name, path in output_paths.items()
            },
            "payload_files": payload_files,
        }
        completed_non_data.append(stage_id)
    target_non_data = [
        str(stage["stage_id"]) for stage in flow_stages if stage.get("stage_id") != "data"
    ]
    if completed_non_data != target_non_data[: len(completed_non_data)]:
        raise ExternalEvidenceError(
            "completed reusable stages must form one contiguous prefix after data"
        )
    exact_cell = completed_non_data == target_non_data and run.get("status") == "frozen"
    source_cell_id = run.get("identity", {}).get("cell_id")
    resolved_target_cell_id = target_cell_id or source_cell_id
    if not isinstance(resolved_target_cell_id, str) or not resolved_target_cell_id:
        raise ExternalEvidenceError("target_cell_id is required")
    manifest = {
        "schema_version": 1,
        "manifest_type": "external_experiment_evidence",
        "evidence_id": evidence_id,
        "eligibility": "exact_cell" if exact_cell else "exact_stage",
        "source": _source_metadata(source),
        "source_identity": normalize_real_value(run.get("identity", {})),
        "source_cell_id": source_cell_id,
        "target_cell_id": resolved_target_cell_id,
        "stage_order": completed_non_data,
        "stages": stages,
        "primary_eligible": exact_cell,
    }
    _write_exclusive(package / "evidence_import_manifest.json", manifest)
    return manifest


def onboard_declared_evidence(
    *, declaration_path: Path, source_root: Path, output_dir: Path
) -> dict[str, Any]:
    """Package externally produced stages described by inspectable evidence."""

    declaration = _load(declaration_path, field="external evidence declaration")
    root = source_root.expanduser().resolve()
    package = output_dir.expanduser().resolve()
    package.mkdir(parents=True, exist_ok=False)
    missing: list[str] = []
    evidence_id = declaration.get("evidence_id")
    if not isinstance(evidence_id, str) or not evidence_id:
        missing.append("evidence_id")
    target_cell_id = declaration.get("target_cell_id")
    if not isinstance(target_cell_id, str) or not target_cell_id:
        missing.append("target_cell_id")
    try:
        source = _source_metadata(declaration.get("source", {}))
    except ExternalEvidenceError:
        source = dict(declaration.get("source", {})) if isinstance(declaration.get("source"), Mapping) else {}
        missing.append("source")
    stages_raw = declaration.get("stages")
    stages: dict[str, Any] = {}
    stage_order = declaration.get("stage_order")
    if not isinstance(stages_raw, Mapping) or not isinstance(stage_order, list):
        missing.append("stages_or_stage_order")
        stages_raw = {}
        stage_order = []
    for stage_id in stage_order:
        raw = stages_raw.get(stage_id) if isinstance(stage_id, str) else None
        if not isinstance(raw, Mapping):
            missing.append(f"stages.{stage_id}")
            continue
        conditions = raw.get("conditions")
        outputs = raw.get("outputs")
        condition_evidence = raw.get("condition_evidence")
        if not isinstance(conditions, Mapping) or not isinstance(outputs, Mapping):
            missing.append(f"stages.{stage_id}.conditions_or_outputs")
            continue
        if not isinstance(condition_evidence, list) or not condition_evidence:
            missing.append(f"stages.{stage_id}.condition_evidence")
        else:
            for relative in condition_evidence:
                candidate = (root / str(relative)).resolve()
                try:
                    candidate.relative_to(root)
                except ValueError:
                    missing.append(f"stages.{stage_id}.condition_evidence")
                else:
                    if not candidate.is_file():
                        missing.append(f"stages.{stage_id}.condition_evidence")
        output_paths: dict[str, Path] = {}
        output_targets: dict[str, str] = {}
        for name, declared_output in outputs.items():
            if isinstance(declared_output, str):
                source_relative = declared_output
                target_relative = declared_output
            elif isinstance(declared_output, Mapping):
                source_relative = declared_output.get("source_path")
                target_relative = declared_output.get("target_relative_path")
            else:
                source_relative = None
                target_relative = None
            if (
                not isinstance(source_relative, str)
                or not source_relative
                or not isinstance(target_relative, str)
                or not target_relative
            ):
                missing.append(f"stages.{stage_id}.outputs.{name}")
                continue
            candidate = (root / source_relative).resolve()
            try:
                candidate.relative_to(root)
            except ValueError:
                missing.append(f"stages.{stage_id}.outputs.{name}")
            else:
                if not candidate.is_file():
                    missing.append(f"stages.{stage_id}.outputs.{name}")
                else:
                    output_paths[str(name)] = candidate
                    output_targets[str(name)] = target_relative
        if output_paths:
            closure = _artifact_closure(
                output_paths.values(),
                root,
                stop_paths=_input_artifact_paths(conditions.get("inputs"), root),
            )
            direct_targets = {
                path.resolve(): output_targets[name]
                for name, path in output_paths.items()
            }
            mappings = [
                (
                    path,
                    direct_targets.get(
                        path.resolve(),
                        _relative_inside(path, root, field="declared payload"),
                    ),
                )
                for path in closure
            ]
            extra_payload = raw.get("payload_files", [])
            if not isinstance(extra_payload, list):
                missing.append(f"stages.{stage_id}.payload_files")
                extra_payload = []
            for item in extra_payload:
                if isinstance(item, str):
                    source_relative = item
                    target_relative = item
                elif isinstance(item, Mapping):
                    source_relative = item.get("source_path")
                    target_relative = item.get("target_relative_path")
                else:
                    source_relative = None
                    target_relative = None
                if (
                    not isinstance(source_relative, str)
                    or not source_relative
                    or not isinstance(target_relative, str)
                    or not target_relative
                ):
                    missing.append(f"stages.{stage_id}.payload_files")
                    continue
                candidate = (root / source_relative).resolve()
                try:
                    candidate.relative_to(root)
                except ValueError:
                    missing.append(f"stages.{stage_id}.payload_files")
                    continue
                if not candidate.is_file():
                    missing.append(f"stages.{stage_id}.payload_files")
                    continue
                mappings.append((candidate, target_relative))
            stages[str(stage_id)] = {
                "conditions": normalize_real_value(conditions),
                "outputs": output_targets,
                "payload_files": _copy_payload_mapping(root, package, mappings),
            }
    eligibility = declaration.get("eligibility")
    if eligibility not in ELIGIBILITIES:
        missing.append("eligibility")
        eligibility = "attested_legacy"
    review = declaration.get("review")
    if eligibility in {"exact_cell", "exact_stage"}:
        if (
            not isinstance(review, Mapping)
            or not isinstance(review.get("reviewer"), str)
            or not review.get("reviewer")
            or review.get("decision") != "exact_real_value_match"
            or review.get("reviewed_stages") != stage_order
        ):
            missing.append("review.exact_real_value_match")
        if eligibility == "exact_cell":
            completion_evidence = declaration.get("cell_completion_evidence")
            if not isinstance(completion_evidence, list) or not completion_evidence:
                missing.append("cell_completion_evidence")
            else:
                for relative in completion_evidence:
                    candidate = (root / str(relative)).resolve()
                    try:
                        candidate.relative_to(root)
                    except ValueError:
                        missing.append("cell_completion_evidence")
                    else:
                        if not candidate.is_file():
                            missing.append("cell_completion_evidence")
    if missing:
        eligibility = "attested_legacy"
    manifest = {
        "schema_version": 1,
        "manifest_type": "external_experiment_evidence",
        "evidence_id": evidence_id,
        "eligibility": eligibility,
        "source": source,
        "source_identity": normalize_real_value(declaration.get("source_identity", {})),
        "source_cell_id": declaration.get("source_cell_id"),
        "target_cell_id": target_cell_id,
        "stage_order": [str(value) for value in stage_order if str(value) in stages],
        "stages": stages,
        "primary_eligible": eligibility == "exact_cell" and not missing,
        "review": normalize_real_value(review) if isinstance(review, Mapping) else None,
        "onboarding": {
            "missing_or_unverifiable_fields": sorted(set(missing)),
            "manual_values_are_not_self_validating": True,
        },
    }
    _write_exclusive(package / "evidence_import_manifest.json", manifest)
    return manifest


def _stage_group(stage_id: str) -> str | None:
    if stage_id.startswith("position_"):
        return "position"
    if stage_id == "bank_train" or stage_id.startswith(("loreft_train_", "bipo_train_")):
        return "candidate_bank"
    if stage_id == "bank_finalize" or "audit" in stage_id:
        return "bank"
    if stage_id.startswith("alpha_"):
        return "alpha"
    if stage_id.startswith("test_"):
        return "test"
    return None


def required_reuse_stages(flow_stages: Sequence[Mapping[str, Any]], through: str) -> list[str]:
    if through not in REUSE_LEVELS:
        raise ExternalEvidenceError(f"reuse_through must be one of {list(REUSE_LEVELS)}")
    limit = REUSE_LEVELS.index(through)
    required: list[str] = []
    for stage in flow_stages:
        stage_id = str(stage.get("stage_id"))
        group = _stage_group(stage_id)
        if group is not None and REUSE_LEVELS.index(group) <= limit:
            required.append(stage_id)
    if not required:
        raise ExternalEvidenceError(f"current flow has no stages through {through}")
    return required


def load_reuse_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path.expanduser().resolve()
    value = _load(manifest_path, field="external evidence manifest")
    if (
        value.get("schema_version") != 1
        or value.get("manifest_type") != "external_experiment_evidence"
        or value.get("eligibility") not in {"exact_cell", "exact_stage"}
    ):
        raise ExternalEvidenceError(
            "only exact_cell/exact_stage external evidence may be reused"
        )
    value["manifest_path"] = str(manifest_path)
    return value


def validate_reuse_plan(
    *, manifest: Mapping[str, Any], flow_stages: Sequence[Mapping[str, Any]], through: str
) -> list[str]:
    required = required_reuse_stages(flow_stages, through)
    available = manifest.get("stages")
    order = manifest.get("stage_order")
    if not isinstance(available, Mapping) or not isinstance(order, list):
        raise ExternalEvidenceError("external evidence stage catalog is invalid")
    if order[: len(required)] != required or any(stage not in available for stage in required):
        raise ExternalEvidenceError(
            "external evidence does not provide the required contiguous stage prefix"
        )
    return required


def import_stage_payload(
    *,
    manifest: Mapping[str, Any],
    source_stage: Mapping[str, Any],
    target_stage: Mapping[str, Any],
    cell_root: Path,
) -> dict[str, Path]:
    source_conditions = source_stage.get("conditions")
    target_conditions = stage_conditions(target_stage)
    if not isinstance(source_conditions, Mapping) or _comparison_conditions(source_conditions) != _comparison_conditions(target_conditions):
        raise ExternalEvidenceError(
            f"scientific conditions differ for stage {target_stage.get('stage_id')}"
        )
    manifest_path = Path(str(manifest["manifest_path"]))
    package_root = manifest_path.parent
    payload_files = source_stage.get("payload_files")
    if not isinstance(payload_files, list):
        raise ExternalEvidenceError("external stage payload_files are missing")
    for relative in payload_files:
        if not isinstance(relative, str) or not relative:
            raise ExternalEvidenceError("external payload path is invalid")
        if relative == "data" or relative.startswith("data/"):
            raise ExternalEvidenceError(
                "external evidence cannot replace the current data stage"
            )
        source = (package_root / "payload" / relative).resolve()
        try:
            source.relative_to((package_root / "payload").resolve())
        except ValueError as exc:
            raise ExternalEvidenceError("external payload escapes its package") from exc
        destination = (cell_root / relative).resolve()
        try:
            destination.relative_to(cell_root.resolve())
        except ValueError as exc:
            raise ExternalEvidenceError("external payload target escapes current cell") from exc
        if not source.is_file():
            raise ExternalEvidenceError(f"external payload is missing: {relative}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if not destination.is_file() or destination.read_bytes() != source.read_bytes():
                raise ExternalEvidenceError(
                    f"existing target differs from imported real bytes: {relative}"
                )
        else:
            shutil.copy2(source, destination)
    declared_outputs = source_stage.get("outputs")
    target_outputs = target_stage.get("outputs")
    if not isinstance(declared_outputs, Mapping) or not isinstance(target_outputs, Mapping):
        raise ExternalEvidenceError("stage outputs are invalid")
    if set(declared_outputs) != set(target_outputs):
        raise ExternalEvidenceError("external/current stage output names differ")
    output_paths: dict[str, Path] = {}
    for name, relative in declared_outputs.items():
        current_relative = target_outputs[name].get("relative_path")
        if relative != current_relative:
            raise ExternalEvidenceError(
                f"external/current output path differs for {target_stage.get('stage_id')}.{name}"
            )
        path = (cell_root / str(relative)).resolve()
        if not path.is_file():
            raise ExternalEvidenceError(f"imported stage output is missing: {path}")
        output_paths[str(name)] = path
    return output_paths


def register_portable_evidence(
    *, catalog_path: Path, manifest: Mapping[str, Any], primary: bool
) -> dict[str, Any]:
    """Register provenance without recording a machine-local package path."""

    if (
        manifest.get("schema_version") != 1
        or manifest.get("manifest_type") != "external_experiment_evidence"
        or manifest.get("eligibility") not in ELIGIBILITIES
    ):
        raise ExternalEvidenceError("evidence eligibility is invalid")
    evidence_id = manifest.get("evidence_id")
    target_cell_id = manifest.get("target_cell_id")
    if not isinstance(evidence_id, str) or not evidence_id:
        raise ExternalEvidenceError("evidence_id is invalid")
    if primary and (
        manifest.get("eligibility") != "exact_cell"
        or not isinstance(target_cell_id, str)
        or not target_cell_id
    ):
        raise ExternalEvidenceError("only exact_cell evidence can be primary")
    path = catalog_path.expanduser().resolve()
    if path.exists():
        catalog = _load(path, field="portable evidence catalog")
    else:
        catalog = {"schema_version": 1, "entries": {}}
    if catalog.get("schema_version") != 1 or not isinstance(catalog.get("entries"), Mapping):
        raise ExternalEvidenceError("portable evidence catalog is invalid")
    entries = copy.deepcopy(dict(catalog["entries"]))
    if evidence_id in entries:
        raise ExternalEvidenceError(f"evidence_id is already registered: {evidence_id}")
    if primary:
        conflicts = [
            key
            for key, value in entries.items()
            if isinstance(value, Mapping)
            and value.get("cell_id") == target_cell_id
            and value.get("primary") is True
        ]
        if conflicts:
            raise ExternalEvidenceError(
                f"cell already has primary evidence: {conflicts[0]}"
            )
    entries[evidence_id] = {
        "cell_id": target_cell_id,
        "eligibility": manifest.get("eligibility"),
        "primary": bool(primary),
        "source": copy.deepcopy(manifest.get("source")),
        "reusable_stages": list(manifest.get("stage_order", [])),
        "machine_registry_key": f"evidence/{evidence_id}",
    }
    candidate = copy.deepcopy(catalog)
    candidate["entries"] = entries
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(candidate, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage reusable external experiment evidence")
    subparsers = parser.add_subparsers(dest="command", required=True)
    framework = subparsers.add_parser("onboard-cell")
    framework.add_argument("--source-cell-root", type=Path, required=True)
    framework.add_argument("--source", type=Path, required=True)
    framework.add_argument("--evidence-id", required=True)
    framework.add_argument("--output-dir", type=Path, required=True)
    framework.add_argument("--target-cell-id")
    legacy = subparsers.add_parser("onboard-external")
    legacy.add_argument("--source-root", type=Path, required=True)
    legacy.add_argument("--declaration", type=Path, required=True)
    legacy.add_argument("--output-dir", type=Path, required=True)
    register = subparsers.add_parser("register")
    register.add_argument("--catalog", type=Path, required=True)
    register.add_argument("--manifest", type=Path, required=True)
    register.add_argument("--primary", action="store_true")
    args = parser.parse_args()
    if args.command == "onboard-cell":
        result = onboard_framework_cell(
            source_cell_root=args.source_cell_root,
            source=_load(args.source, field="source metadata"),
            evidence_id=args.evidence_id,
            output_dir=args.output_dir,
            target_cell_id=args.target_cell_id,
        )
    elif args.command == "onboard-external":
        result = onboard_declared_evidence(
            declaration_path=args.declaration,
            source_root=args.source_root,
            output_dir=args.output_dir,
        )
    else:
        manifest = load_reuse_manifest(args.manifest) if args.primary else _load(
            args.manifest, field="external evidence manifest"
        )
        result = register_portable_evidence(
            catalog_path=args.catalog, manifest=manifest, primary=args.primary
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
