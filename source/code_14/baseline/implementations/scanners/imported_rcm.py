"""Import previously measured RCM evidence without running the model again.

The machine registry supplies only the absolute path of ``import_manifest.json``.
The manifest supplies inspectable scientific conditions and relative evidence
files.  Only packages marked ``verified_import`` are executable; incomplete
legacy packages remain preservable as ``attested_legacy``.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..loaders import load_rows, resolve_model_id
from ..position_scanning import PositionScanResult, summarize_measurements


class ImportedRCMError(ValueError):
    """Raised when external RCM evidence is not equivalent to the current role."""


def _object(path: Path, *, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ImportedRCMError(f"cannot read {field}: {path}") from exc
    if not isinstance(value, dict):
        raise ImportedRCMError(f"{field} must contain one JSON object")
    return value


def _rows(path: Path, *, field: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    try:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ImportedRCMError(f"{field} row {number} is not an object")
            output.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise ImportedRCMError(f"cannot read {field}: {path}") from exc
    if not output:
        raise ImportedRCMError(f"{field} is empty")
    return output


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _relative_file(manifest_path: Path, files: Mapping[str, Any], name: str) -> Path:
    value = files.get(name)
    if not isinstance(value, str) or not value:
        raise ImportedRCMError(f"import manifest files.{name} is missing")
    path = (manifest_path.parent / value).resolve()
    try:
        path.relative_to(manifest_path.parent.resolve())
    except ValueError as exc:
        raise ImportedRCMError(f"import file escapes its package: {name}") from exc
    if not path.is_file():
        raise ImportedRCMError(f"import file does not exist: {path}")
    return path


def _optional_file(
    manifest_path: Path, files: Mapping[str, Any], name: str
) -> Path | None:
    if name not in files or files.get(name) is None:
        return None
    return _relative_file(manifest_path, files, name)


def _sample_id(row: Mapping[str, Any], index: int) -> str:
    value = row.get("sample_id", row.get("id", index))
    if not isinstance(value, str) or not value:
        value = str(value)
    return value


def _ordered_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def _scientific_scan_config(value: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only the machine import pointer from the current scan conditions."""

    output = copy.deepcopy(dict(value))
    output.pop("external_position_source", None)
    return output


def _validate_evidence_links(manifest_path: Path, manifest: Mapping[str, Any]) -> None:
    evidence = manifest.get("evidence")
    required = {"model", "data", "scorer", "rcm"}
    if not isinstance(evidence, Mapping) or set(evidence) != required:
        raise ImportedRCMError(
            "verified legacy import requires model/data/scorer/rcm evidence links"
        )
    for name, relative in evidence.items():
        if not isinstance(relative, str) or not relative:
            raise ImportedRCMError(f"legacy evidence link {name} is invalid")
        path = (manifest_path.parent / relative).resolve()
        try:
            path.relative_to(manifest_path.parent.resolve())
        except ValueError as exc:
            raise ImportedRCMError(f"legacy evidence link escapes package: {name}") from exc
        if not path.is_file():
            raise ImportedRCMError(f"legacy evidence file is missing: {name}")


def _validate_conditions(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    *,
    request: Any,
    current_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[str], dict[str, Mapping[str, Any]]]:
    if manifest.get("schema_version") != 1 or manifest.get("manifest_type") != "rcm_external_import":
        raise ImportedRCMError("external RCM import manifest identity is invalid")
    if manifest.get("eligibility") != "verified_import":
        raise ImportedRCMError(
            "only verified_import packages may execute; attested legacy evidence is archival"
        )
    _validate_evidence_links(manifest_path, manifest)
    conditions = manifest.get("scientific_conditions")
    if not isinstance(conditions, Mapping):
        raise ImportedRCMError("scientific_conditions is missing")
    if conditions.get("method") != request.method:
        raise ImportedRCMError("imported RCM method differs from the current method")
    model_id = resolve_model_id(request.model_config, mode="inference")
    expected_model = request.model_config.get(
        "model_registry_id", request.model_config.get("model_id")
    )
    if conditions.get("model_registry_id") not in {expected_model, model_id}:
        raise ImportedRCMError("imported RCM model differs from the current model")
    current_dataset = None
    role_manifest = _object(request.data_role_manifest, field="current selector role")
    current_dataset = role_manifest.get("dataset")
    if conditions.get("dataset") != current_dataset:
        raise ImportedRCMError("imported RCM dataset differs from the current selector role")
    declared_scan = conditions.get("scan_config")
    if not isinstance(declared_scan, Mapping):
        raise ImportedRCMError("scientific_conditions.scan_config is missing")
    if dict(declared_scan) != _scientific_scan_config(request.scan_config):
        raise ImportedRCMError("imported RCM scan conditions differ from the current protocol")
    current_ids = [_sample_id(row, index) for index, row in enumerate(current_rows)]
    if len(current_ids) != len(set(current_ids)):
        raise ImportedRCMError("current selector role has duplicate sample IDs")
    metadata: dict[str, Mapping[str, Any]] = {}
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ImportedRCMError("import manifest files are missing")
    metadata_path = _optional_file(manifest_path, files, "sample_metadata")
    if metadata_path is not None:
        for index, row in enumerate(_rows(metadata_path, field="sample metadata")):
            sample_id = _sample_id(row, index)
            if sample_id in metadata:
                raise ImportedRCMError("sample metadata contains duplicate sample IDs")
            metadata[sample_id] = row
    return current_ids, metadata


def _filter_ids(
    source_ids: Sequence[str],
    metadata: Mapping[str, Mapping[str, Any]],
    source_config: Mapping[str, Any],
) -> list[str]:
    subset_filter = source_config.get("subset_filter")
    if subset_filter is None:
        return list(source_ids)
    if not isinstance(subset_filter, Mapping):
        raise ImportedRCMError("subset_filter must be an object")
    field = subset_filter.get("field")
    values = subset_filter.get("values")
    if not isinstance(field, str) or not field or not isinstance(values, list) or not values:
        raise ImportedRCMError("subset_filter requires field and non-empty values")
    allowed = {str(value) for value in values}
    if not metadata:
        raise ImportedRCMError("mixed-source splitting requires sample_metadata")
    selected: list[str] = []
    for sample_id in source_ids:
        row = metadata.get(sample_id)
        if row is None or field not in row:
            raise ImportedRCMError(
                f"sample metadata cannot resolve {field!r} for {sample_id!r}"
            )
        if str(row[field]) in allowed:
            selected.append(sample_id)
    return selected


def _validate_component_coverage(rows: Sequence[Mapping[str, Any]], sample_ids: Sequence[str]) -> None:
    expected: set[str] | None = None
    for sample_id in sample_ids:
        components = {
            str(row.get("component_id"))
            for row in rows
            if str(row.get("sample_id")) == sample_id
        }
        if not components or "None" in components:
            raise ImportedRCMError(f"sample {sample_id!r} has no component coverage")
        if expected is None:
            expected = components
        elif components != expected:
            raise ImportedRCMError("imported samples do not share one candidate component pool")


class ExternalRCMImportScanner:
    """Model-free scanner that normalizes verified legacy RCM or fixed K8 evidence."""

    def scan(self, request: Any) -> PositionScanResult:
        source_config = request.scan_config.get("external_position_source")
        if not isinstance(source_config, Mapping):
            raise ImportedRCMError("external_position_source is missing")
        manifest_value = source_config.get("manifest")
        if not isinstance(manifest_value, str) or not Path(manifest_value).is_absolute():
            raise ImportedRCMError("external import manifest must be a registered absolute path")
        manifest_path = Path(manifest_value).resolve()
        manifest = _object(manifest_path, field="external RCM import manifest")
        current_rows = load_rows(request.data_role_manifest, root=request.evidence_root)
        current_ids, metadata = _validate_conditions(
            manifest_path, manifest, request=request, current_rows=current_rows
        )
        files = manifest["files"]
        artifact_kind = manifest.get("artifact_kind")
        request.output_dir.mkdir(parents=True, exist_ok=True)
        candidate_path = request.output_dir / "candidate_scores.jsonl"
        component_path = request.output_dir / "component_scores.jsonl"
        trajectory_path: Path | None = None

        if artifact_kind == "rcm_scan":
            sample_path = _relative_file(manifest_path, files, "sample_scores")
            source_rows = _rows(sample_path, field="RCM sample scores")
            source_ids = _ordered_unique(
                str(row.get("sample_id")) for row in source_rows
            )
            selected_ids = _filter_ids(source_ids, metadata, source_config)
            if selected_ids != current_ids:
                raise ImportedRCMError(
                    "imported/split sample order differs from the current selector role"
                )
            selected = set(selected_ids)
            filtered_rows = [
                row for row in source_rows if str(row.get("sample_id")) in selected
            ]
            _validate_component_coverage(filtered_rows, selected_ids)
            subset_filter = source_config.get("subset_filter")
            scope = manifest.get("candidate_scope")
            reuse = source_config.get("reuse_semantics", "exact_role")
            if subset_filter is not None:
                allowed = {
                    "all_model_heads": "exact_role",
                    "shared_source_head_pool": "shared_candidate_pool",
                }
                if allowed.get(scope) != reuse:
                    raise ImportedRCMError(
                        "mixed split must declare exact all-head reuse or shared candidate-pool reuse"
                    )
            summaries = summarize_measurements(
                filtered_rows,
                method=request.method,
                scan_config=_scientific_scan_config(request.scan_config),
            )
            _write_jsonl(candidate_path, summaries)
            _write_jsonl(
                component_path,
                (
                    {
                        "sample_id": row["sample_id"],
                        "component_id": row["component_id"],
                        "score": row.get("effect", row.get("mean_score_delta")),
                    }
                    for row in filtered_rows
                ),
            )
            source_trajectory = _optional_file(manifest_path, files, "trajectory")
            if source_trajectory is not None and request.trajectory_config.get("enabled", True):
                trajectory_rows = [
                    row
                    for row in _rows(source_trajectory, field="RCM trajectory")
                    if str(row.get("sample_id")) in selected
                ]
                trajectory_path = request.output_dir / "trajectory.jsonl"
                _write_jsonl(trajectory_path, trajectory_rows)
        elif artifact_kind == "fixed_k8":
            position_path = _relative_file(manifest_path, files, "positions")
            positions = _rows(position_path, field="fixed K8 positions")
            declared_ids = manifest.get("selector_sample_ids")
            if declared_ids != current_ids:
                raise ImportedRCMError(
                    "fixed K8 selector samples differ from the current selector role"
                )
            _write_jsonl(candidate_path, positions)
            source_components = _optional_file(manifest_path, files, "component_scores")
            if source_components is not None:
                _write_jsonl(
                    component_path,
                    _rows(source_components, field="fixed K8 component scores"),
                )
            else:
                _write_jsonl(
                    component_path,
                    (
                        {
                            "sample_id": sample_id,
                            "component_id": row.get("component_id"),
                            "score": row.get("selection_score", row.get("effect")),
                        }
                        for sample_id in current_ids
                        for row in positions
                    ),
                )
        else:
            raise ImportedRCMError("artifact_kind must be rcm_scan or fixed_k8")

        execution_path = request.output_dir / "scan_execution.json"
        execution_path.write_text(
            json.dumps(
                {
                    "method": request.method,
                    "model_id": resolve_model_id(request.model_config, mode="inference"),
                    "rows": len(current_ids),
                    "execution": "external_verified_import",
                    "artifact_kind": artifact_kind,
                    "candidate_scope": manifest.get("candidate_scope"),
                    "reuse_semantics": source_config.get("reuse_semantics", "exact_role"),
                    "source_manifest": str(manifest_path),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return PositionScanResult(
            candidate_scores=candidate_path,
            execution_manifest=execution_path,
            trajectory_path=trajectory_path,
            component_scores_path=component_path,
        )
