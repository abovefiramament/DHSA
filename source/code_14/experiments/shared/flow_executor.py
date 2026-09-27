"""Execute a compiled experiment flow through fixed component adapters.

The executor owns ordering, stage lifecycle, artifact resolution, and failure
provenance. Scientific behavior stays in dataset, method, audit, and evaluator
implementations registered by the compiler.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from baseline.controllers import (
    build_bank_plan_from_position_freeze,
    freeze_position_plan,
    resolve_method_parameters,
    resolve_positions,
    validate_position_freeze,
)
from baseline.controllers.bank_execution_controller import (
    execute_registered_bank_plan,
    finalize_audited_bank_plan,
)
from experiments.shared.component_runtime import (
    finalize_post_training_audit,
    materialize_dataset,
    prepare_post_training_audit,
)
from experiments.shared.post_training_audit_execution import (
    execute_post_training_audit,
    resolve_blinded_audit_decisions,
)
from experiments.shared.evidence_registry import (
    allocate_run,
    await_audit,
    complete_stage,
    complete_imported_stage,
    fail_stage,
    freeze_run,
    load_run,
    register_audit_approval,
    start_stage,
)
from experiments.shared.external_evidence import (
    ExternalEvidenceError,
    import_stage_payload,
    load_reuse_manifest,
    validate_reuse_plan,
)
from experiments.shared.manual_audit import (
    approve_audit,
    create_audit_request,
    validate_audit_approval,
)
from experiments.shared.evaluation_controller import (
    authorize_final_test,
    freeze_alpha,
    run_registered_evaluator,
)
from experiments.shared.component_catalog import load_component_catalog
from experiments.shared.generation_controller import run_registered_generator
from experiments.shared.position_search_controller import run_registered_position_backend


class FlowExecutionError(RuntimeError):
    """Raised when a compiled stage cannot be executed through its adapter."""


class FlowAwaitingAudit(FlowExecutionError):
    """Signals a deliberate pause at an immutable human-audit gate."""


@dataclass(frozen=True)
class StageContext:
    cell_root: Path
    runtime_config_path: Path
    runtime_config: Mapping[str, Any]
    stage: Mapping[str, Any]
    inputs: Mapping[str, Any]
    outputs: Mapping[str, Path]


StageHandler = Callable[[StageContext], Mapping[str, Path]]


def _protocol_reference(runtime_config: Mapping[str, Any]) -> dict[str, Any]:
    protocol = runtime_config.get("protocol")
    if not isinstance(protocol, Mapping):
        raise FlowExecutionError("runtime protocol is missing")
    if not isinstance(protocol.get("id"), str) or not isinstance(protocol.get("path"), str):
        raise FlowExecutionError("runtime protocol id/path is missing")
    return {"id": protocol["id"], "path": protocol["path"]}


def _load_object(path: Path, *, field: str) -> dict[str, Any]:
    if not path.is_file():
        raise FlowExecutionError(f"{field} does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FlowExecutionError(f"{field} must contain one JSON object")
    return value


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(dict(value), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as exc:
        raise FlowExecutionError(f"stage output already exists: {path}") from exc


def _bind_shared_request(path: Path, expected: Mapping[str, Any]) -> None:
    """Atomically create a shared request, or validate the winning request."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(expected), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            pass
    finally:
        temporary_path.unlink(missing_ok=True)
    registered = _load_object(path, field="shared dataset materialization request")
    if registered != dict(expected):
        raise FlowExecutionError(
            "shared dataset key is already bound to a different materialization request"
        )


def _input_values(stage: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Path]]:
    values: dict[str, Any] = {}
    files: dict[str, Path] = {}
    for name, raw in stage["inputs"].items():
        if raw["kind"] == "config":
            values[name] = raw["value"]
        elif raw["kind"] == "artifact":
            path = Path(raw["absolute_path"])
            values[name] = path
            files[name] = path
        else:
            raise FlowExecutionError(f"unsupported input kind: {raw['kind']}")
    return values, files


def _bind_artifacts(value: Any, stage: Mapping[str, Any]) -> Any:
    by_reference = {
        raw["reference"]: Path(raw["absolute_path"])
        for raw in stage["inputs"].values()
        if raw["kind"] == "artifact"
    }
    if isinstance(value, Mapping):
        if set(value) == {"artifact"}:
            reference = value["artifact"]
            path = by_reference.get(reference)
            if path is None or not path.is_file():
                raise FlowExecutionError(
                    f"stage does not provide materialized artifact {reference}"
                )
            return {"path": str(path)}
        return {key: _bind_artifacts(item, stage) for key, item in value.items()}
    if isinstance(value, list):
        return [_bind_artifacts(item, stage) for item in value]
    return value


def _dataset_handler(context: StageContext) -> Mapping[str, Path]:
    data_spec = context.inputs.get("data_config")
    if not isinstance(data_spec, Mapping):
        raise FlowExecutionError("dataset stage requires config input data_config")
    dataset = context.runtime_config["identity"]["dataset"]
    shared_key = data_spec.get("shared_artifact_key")
    if not isinstance(shared_key, str) or not shared_key or any(token in shared_key for token in ("/", "\\", "..")):
        raise FlowExecutionError("formal dataset stage requires a safe shared_artifact_key")
    materialized_dir = context.cell_root.parent / "_site_shared_data" / shared_key
    if context.runtime_config["identity"]["experiment"] == "performance":
        materialized_dir = materialized_dir / context.cell_root.name
    request_path = materialized_dir / "data_request.local.json"
    manifest_path = materialized_dir / "dataset_manifest.json"
    expected_request = {
        "schema_version": 1,
        "request_type": "dataset_materialization",
        "dataset": dataset,
        "protocol": _protocol_reference(context.runtime_config),
        "data_spec": dict(data_spec),
    }
    _bind_shared_request(request_path, expected_request)
    if manifest_path.is_file():
        manifest = _load_object(manifest_path, field="shared dataset manifest")
        if (
            manifest.get("manifest_type") != "data_freeze"
            or manifest.get("status") != "frozen"
            or manifest.get("dataset") != dataset
        ):
            raise FlowExecutionError("shared dataset artifact does not match the formal data stage")
    else:
        manifest = materialize_dataset(
            dataset,
            data_spec,
            output_dir=materialized_dir,
            execution_config=context.runtime_config["shared"]["execution"],
            evidence_root=context.cell_root,
        )
    role_artifacts = manifest.get("role_artifacts")
    role_purposes = manifest.get("role_purposes")
    if not isinstance(role_artifacts, Mapping) or not isinstance(role_purposes, Mapping):
        raise FlowExecutionError("dataset controller did not return a data-role contract")
    raw_bindings = data_spec.get("role_outputs")
    if raw_bindings is None:
        role_name_by_output = {
            "training_manifest": "train",
            "validation_manifest": "validation",
            "alpha_manifest": "alpha_dev",
            "test_manifest": "test",
        }
    elif isinstance(raw_bindings, Mapping):
        role_name_by_output = dict(raw_bindings)
        if not all(
            isinstance(name, str)
            and name
            and isinstance(role, str)
            and role in role_artifacts
            for name, role in role_name_by_output.items()
        ):
            raise FlowExecutionError("data.role_outputs contains an invalid role binding")
    else:
        raise FlowExecutionError("data.role_outputs must be an object")
    declared_role_outputs = set(context.outputs) - {"dataset_manifest"}
    if declared_role_outputs != set(role_name_by_output):
        raise FlowExecutionError(
            "dataset stage outputs must exactly match data.role_outputs: "
            f"expected={sorted(role_name_by_output)} got={sorted(declared_role_outputs)}"
        )
    produced: dict[str, Path] = {}
    request_artifact = {
        "relative_path": os.path.relpath(request_path, start=context.cell_root),
        "size_bytes": request_path.stat().st_size,
    }
    source_artifact = {
        "relative_path": os.path.relpath(manifest_path, start=context.cell_root),
        "size_bytes": manifest_path.stat().st_size,
    }
    for name, path in context.outputs.items():
        if name == "dataset_manifest":
            _write_exclusive(
                path,
                {
                    "manifest_type": "data_freeze_reference",
                    "status": "frozen",
                    "data_request": request_artifact,
                    "dataset_manifest": source_artifact,
                    "artifact_closure": [request_artifact, source_artifact],
                },
            )
        else:
            role = role_name_by_output[name]
            artifact = role_artifacts.get(role)
            if artifact is None:
                raise FlowExecutionError(f"dataset output {name!r} has no role mapping")
            role_file = materialized_dir / artifact["relative_path"]
            role_artifact = {
                "relative_path": os.path.relpath(role_file, start=context.cell_root),
                "size_bytes": role_file.stat().st_size,
                "rows": artifact["rows"],
            }
            role_manifest = {
                "schema_version": 1,
                "manifest_type": "data_role_freeze",
                "status": "frozen",
                "dataset": dataset,
                "data_role": role,
                "purpose": role_purposes[role],
                "data_request": request_artifact,
                "dataset_manifest": source_artifact,
                "role_artifact": role_artifact,
                "artifact_closure": [request_artifact, source_artifact, role_artifact],
            }
            if manifest.get("role_mode") == "legacy_splits":
                role_manifest.update({"split": role, "split_artifact": role_artifact})
            _write_exclusive(path, role_manifest)
        produced[name] = path
    return produced

def _position_handler(context: StageContext) -> Mapping[str, Path]:
    position_config = context.inputs.get("position_config")
    if not isinstance(position_config, Mapping):
        raise FlowExecutionError("position stage requires config input position_config")
    bound = _bind_artifacts(
        {"position_control": position_config}, context.stage
    )
    plan = resolve_positions(bound)
    if len(context.outputs) != 1:
        raise FlowExecutionError("position stage must declare exactly one output")
    name, path = next(iter(context.outputs.items()))
    _write_exclusive(path, plan)
    return {name: path}


def _position_search_handler(context: StageContext) -> Mapping[str, Path]:
    scanner_binding = context.inputs.get("scanner_binding")
    binding = context.inputs.get("backend_binding")
    method_config = context.inputs.get("method_config")
    model_config = context.inputs.get("model_config")
    scan_config = context.inputs.get("scan_config")
    selector_role = context.inputs.get("selector_data_manifest")
    if (
        not isinstance(scanner_binding, Mapping)
        or not isinstance(binding, Mapping)
        or not isinstance(method_config, Mapping)
        or not isinstance(model_config, Mapping)
        or not isinstance(scan_config, Mapping)
        or not isinstance(selector_role, Path)
    ):
        raise FlowExecutionError(
            "position search requires scanner/backend bindings, model/scan configs, "
            "method_config, and selector_data_manifest"
        )
    required = {
        "candidate_manifest",
        "backend_execution_manifest",
        "position_search_manifest",
    }
    if set(context.outputs) != required:
        raise FlowExecutionError(f"position search outputs must be {sorted(required)}")
    trajectory_config = context.inputs.get("trajectory_config")
    if trajectory_config is not None and not isinstance(trajectory_config, Mapping):
        raise FlowExecutionError("position search trajectory_config must be a config object")
    manifest = run_registered_position_backend(
        scanner_binding=scanner_binding,
        backend_binding=binding,
        method_plan=resolve_method_parameters(method_config),
        model_config=model_config,
        scan_config=scan_config,
        execution_config=context.runtime_config["shared"]["execution"],
        trajectory_config=trajectory_config,
        selector_data_manifest=selector_role,
        evidence_root=context.cell_root,
        output_dir=context.outputs["position_search_manifest"].parent,
        manifest_path=context.outputs["position_search_manifest"],
    )
    expected = {
        "candidate_manifest": manifest["candidate_manifest"]["relative_path"],
        "backend_execution_manifest": manifest["backend_execution_manifest"]["relative_path"],
    }
    for name, relative in expected.items():
        if context.outputs[name].resolve() != (context.cell_root / relative).resolve():
            raise FlowExecutionError(f"position search output path drift for {name}")
    return dict(context.outputs)


def _position_freeze_handler(context: StageContext) -> Mapping[str, Path]:
    position_path = context.inputs.get("position_plan")
    data_freeze_path = context.inputs.get("data_freeze")
    method_config = context.inputs.get("method_config")
    if (
        not isinstance(position_path, Path)
        or not isinstance(data_freeze_path, Path)
        or not isinstance(method_config, Mapping)
    ):
        raise FlowExecutionError(
            "position freeze requires position_plan, data_freeze, and method_config"
        )
    if set(context.outputs) != {"position_freeze"}:
        raise FlowExecutionError("position freeze must output position_freeze")
    freeze_position_plan(
        _load_object(position_path, field="position plan"),
        resolve_method_parameters(method_config),
        data_freeze_path=data_freeze_path,
        protocol=_protocol_reference(context.runtime_config),
        evidence_root=context.cell_root,
        output_path=context.outputs["position_freeze"],
    )
    return dict(context.outputs)


def _component_manifest_paths(index_path: Path, *, evidence_root: Path) -> list[Path]:
    index = _load_object(index_path, field="component payload index")
    if (
        index.get("manifest_type") != "component_payload_index"
        or index.get("status") != "complete"
    ):
        raise FlowExecutionError("component payload index is incomplete")
    records = index.get("components")
    if not isinstance(records, list) or not records:
        raise FlowExecutionError("component payload index has no components")
    paths: list[Path] = []
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise FlowExecutionError("component payload index row is invalid")
        component_id = record.get("component_id")
        relative_value = record.get("relative_path")
        if (
            not isinstance(component_id, str)
            or not component_id
            or component_id in seen
            or not isinstance(relative_value, str)
            or not relative_value
        ):
            raise FlowExecutionError("component payload index identity is invalid")
        relative = Path(relative_value)
        if relative.is_absolute() or ".." in relative.parts:
            raise FlowExecutionError("component payload index escapes evidence root")
        path = evidence_root.resolve() / relative
        if (
            not path.is_file()
            or path.stat().st_size != record.get("size_bytes")
        ):
            raise FlowExecutionError("component payload index file is missing or has the wrong size")
        manifest = _load_object(path, field=f"component payload {component_id}")
        if manifest.get("component_id") != component_id:
            raise FlowExecutionError("component payload index points to another component")
        seen.add(component_id)
        paths.append(path)
    return paths


def _bank_execute_handler(context: StageContext) -> Mapping[str, Path]:
    position_path = context.inputs.get("position_freeze")
    training = context.inputs.get("training_manifest")
    validation = context.inputs.get("validation_manifest")
    backend_binding = context.inputs.get("backend_binding")
    bank_options = context.inputs.get("bank_options", {})
    if not isinstance(position_path, Path) or not isinstance(training, Path):
        raise FlowExecutionError(
            "bank execution requires position_freeze and training_manifest artifacts"
        )
    if validation is not None and not isinstance(validation, Path):
        raise FlowExecutionError("validation_manifest must be an artifact")
    if not isinstance(backend_binding, Mapping) or not isinstance(bank_options, Mapping):
        raise FlowExecutionError("bank backend binding/options must be config objects")
    required_outputs = {
        "bank_plan",
        "bank_execution_manifest",
        "controller_manifest",
    }
    require_components = bank_options.get("require_component_payloads", False)
    if not isinstance(require_components, bool):
        raise FlowExecutionError("bank_options.require_component_payloads must be boolean")
    expected_outputs = required_outputs | ({"component_index"} if require_components else set())
    if set(context.outputs) != expected_outputs:
        raise FlowExecutionError(
            f"bank execution outputs must be {sorted(expected_outputs)}"
        )
    execution_path = context.outputs["bank_execution_manifest"]
    controller_path = context.outputs["controller_manifest"]
    if (
        execution_path.name != "bank_execution_manifest.json"
        or controller_path.name != "controller_manifest.json"
        or execution_path.parent != controller_path.parent
    ):
        raise FlowExecutionError(
            "bank execution/controller outputs must use fixed names in one directory"
        )
    position_freeze = validate_position_freeze(
        _load_object(position_path, field="position freeze")
    )
    method_plan = resolve_method_parameters(context.runtime_config)
    bank_plan = build_bank_plan_from_position_freeze(position_freeze, method_plan)
    result = execute_registered_bank_plan(
        bank_plan,
        method_plan,
        backend_binding=backend_binding,
        training_data_manifest=training,
        validation_data_manifest=validation,
        evidence_root=context.cell_root,
        output_dir=execution_path.parent,
        execution_config=context.runtime_config["shared"]["execution"],
        require_component_payloads=require_components,
    )
    _write_exclusive(context.outputs["bank_plan"], bank_plan)
    if require_components:
        records: list[dict[str, Any]] = []
        component_dir = execution_path.parent / "banks"
        manifest_paths = sorted(component_dir.glob("*/components/component_*.json"))
        if len(manifest_paths) != len(result["component_payload_manifests"]):
            raise FlowExecutionError("component payload manifest inventory drift")
        for path in manifest_paths:
            manifest = _load_object(path, field="component payload manifest")
            if not isinstance(manifest, Mapping):
                raise FlowExecutionError("component payload manifest is invalid")
            records.append(
                {
                    "component_id": manifest.get("component_id"),
                    "relative_path": path.resolve()
                    .relative_to(context.cell_root.resolve())
                    .as_posix(),
                    "size_bytes": path.stat().st_size,
                }
            )
        index = {
            "schema_version": 1,
            "manifest_type": "component_payload_index",
            "status": "complete",
            "components": records,
            "artifact_closure": [
                {
                    "relative_path": row["relative_path"],
                    "size_bytes": row["size_bytes"],
                }
                for row in records
            ],
        }
        _write_exclusive(context.outputs["component_index"], index)
    if (
        result["bank_execution_manifest_path"] != execution_path
        or result["controller_manifest_path"] != controller_path
    ):
        raise FlowExecutionError("bank controller output path drift")
    return dict(context.outputs)


def _bank_finalize_audit_handler(context: StageContext) -> Mapping[str, Path]:
    audit_freeze_path = context.inputs.get("audit_freeze")
    backend_binding = context.inputs.get("backend_binding")
    training = context.inputs.get("training_manifest")
    validation = context.inputs.get("validation_manifest")
    if not isinstance(audit_freeze_path, Path) or not isinstance(
        backend_binding, Mapping
    ):
        raise FlowExecutionError(
            "audited bank finalization requires audit_freeze and backend_binding"
        )
    if training is not None and not isinstance(training, Path):
        raise FlowExecutionError("training_manifest must be an artifact")
    if validation is not None and not isinstance(validation, Path):
        raise FlowExecutionError("validation_manifest must be an artifact")
    if set(context.outputs) != {"bank_execution_manifest", "controller_manifest"}:
        raise FlowExecutionError(
            "audited bank finalization outputs must be bank_execution_manifest and controller_manifest"
        )
    execution_path = context.outputs["bank_execution_manifest"]
    controller_path = context.outputs["controller_manifest"]
    if (
        execution_path.name != "bank_execution_manifest.json"
        or controller_path.name != "controller_manifest.json"
        or execution_path.parent != controller_path.parent
    ):
        raise FlowExecutionError("audited bank output paths violate the fixed contract")
    component_paths = [
        path
        for name, path in context.inputs.items()
        if name != "component_index"
        and name.startswith("component_")
        and isinstance(path, Path)
    ]
    component_index = context.inputs.get("component_index")
    if component_index is not None:
        if component_paths or not isinstance(component_index, Path):
            raise FlowExecutionError(
                "audited bank finalization uses either component_index or component_* inputs"
            )
        component_paths = _component_manifest_paths(
            component_index, evidence_root=context.cell_root
        )
    components = [
        _load_object(path, field=f"component manifest {index}")
        for index, path in enumerate(component_paths)
    ]
    result = finalize_audited_bank_plan(
        _load_object(audit_freeze_path, field="audit freeze"),
        resolve_method_parameters(context.runtime_config),
        backend_binding=backend_binding,
        evidence_root=context.cell_root,
        output_dir=execution_path.parent,
        execution_config=context.runtime_config["shared"]["execution"],
        component_payload_manifests=components,
        training_data_manifest=training,
        validation_data_manifest=validation,
    )
    if (
        result["bank_execution_manifest_path"] != execution_path
        or result["controller_manifest_path"] != controller_path
    ):
        raise FlowExecutionError("audited bank controller output path drift")
    return dict(context.outputs)


def _evaluator_handler(context: StageContext) -> Mapping[str, Path]:
    predictions = context.inputs.get("predictions")
    references = context.inputs.get("references")
    authorization = context.inputs.get("test_authorization")
    binding = context.inputs.get("evaluator_binding")
    evaluation = context.inputs.get("evaluation_config")
    if not isinstance(predictions, Path):
        raise FlowExecutionError("evaluator requires a predictions artifact")
    if references is not None and not isinstance(references, Path):
        raise FlowExecutionError("evaluator references must be an artifact")
    if authorization is not None and not isinstance(authorization, Path):
        raise FlowExecutionError("evaluator test_authorization must be an artifact")
    if (references is None) == (authorization is None):
        raise FlowExecutionError(
            "evaluator requires exactly one references artifact or test authorization"
        )
    if not isinstance(binding, Mapping) or not isinstance(evaluation, Mapping):
        raise FlowExecutionError("evaluator binding/config must be objects")
    required = {"per_sample_scores", "summary_metrics", "evaluation_manifest"}
    if set(context.outputs) != required:
        raise FlowExecutionError(f"evaluator outputs must be {sorted(required)}")
    output_dir = context.outputs["evaluation_manifest"].parent
    manifest = run_registered_evaluator(
        evaluator_binding=binding,
        predictions=predictions,
        references=references,
        test_authorization_path=authorization,
        evaluation_config=evaluation,
        evidence_root=context.cell_root,
        output_dir=output_dir,
        manifest_path=context.outputs["evaluation_manifest"],
    )
    expected = {
        "per_sample_scores": manifest["per_sample_scores"]["relative_path"],
        "summary_metrics": manifest["summary_metrics"]["relative_path"],
    }
    for name, relative in expected.items():
        if context.outputs[name].resolve() != (context.cell_root / relative).resolve():
            raise FlowExecutionError(f"evaluator output path drift for {name}")
    return dict(context.outputs)


def _generation_handler(context: StageContext) -> Mapping[str, Path]:
    binding = context.inputs.get("backend_binding")
    model_config = context.inputs.get("model_config")
    generation_config = context.inputs.get("generation_config")
    trajectory_config = context.inputs.get("trajectory_config")
    if not all(
        isinstance(item, Mapping)
        for item in (binding, model_config, generation_config)
    ):
        raise FlowExecutionError(
            "generation backend/model/generation inputs must be config objects"
        )
    if trajectory_config is not None and not isinstance(trajectory_config, Mapping):
        raise FlowExecutionError("generation trajectory_config must be a config object")
    required = {"predictions", "backend_execution_manifest", "generation_manifest"}
    if set(context.outputs) != required:
        raise FlowExecutionError(f"generation outputs must be {sorted(required)}")
    controller = context.inputs.get("controller_manifest")
    data_role = context.inputs.get("data_role_manifest")
    authorization = context.inputs.get("test_authorization")
    for name, value in (
        ("controller_manifest", controller),
        ("data_role_manifest", data_role),
        ("test_authorization", authorization),
    ):
        if value is not None and not isinstance(value, Path):
            raise FlowExecutionError(f"generation {name} must be an artifact")
    output_dir = context.outputs["generation_manifest"].parent
    manifest = run_registered_generator(
        backend_binding=binding,
        model_config=model_config,
        generation_config=generation_config,
        trajectory_config=trajectory_config,
        trace_kind=(
            context.stage.get("trace_kind")
            or (
                "audit"
                if "audit" in str(context.stage.get("stage_id", "")).lower()
                else "generation_test"
            )
        ),
        execution_config=context.runtime_config["shared"]["execution"],
        evidence_root=context.cell_root,
        output_dir=output_dir,
        manifest_path=context.outputs["generation_manifest"],
        controller_manifest_path=controller,
        data_role_manifest_path=data_role,
        test_authorization_path=authorization,
    )
    expected = {
        "predictions": manifest["predictions"]["relative_path"],
        "backend_execution_manifest": manifest["backend_execution_manifest"]["relative_path"],
    }
    for name, relative in expected.items():
        if context.outputs[name].resolve() != (context.cell_root / relative).resolve():
            raise FlowExecutionError(f"generation output path drift for {name}")
    return dict(context.outputs)


def _alpha_freeze_handler(context: StageContext) -> Mapping[str, Path]:
    required_inputs = {
        "curve",
        "controller_manifest",
        "calibration_manifest",
        "evaluator_manifest",
        "selector_binding",
        "selection_config",
    }
    if not required_inputs.issubset(context.inputs):
        raise FlowExecutionError("alpha freeze inputs are incomplete")
    if len(context.outputs) != 1 or "alpha_freeze" not in context.outputs:
        raise FlowExecutionError("alpha freeze must output alpha_freeze")
    paths = {
        name: context.inputs[name]
        for name in (
            "curve",
            "controller_manifest",
            "calibration_manifest",
            "evaluator_manifest",
        )
    }
    if not all(isinstance(path, Path) for path in paths.values()):
        raise FlowExecutionError("alpha freeze artifact inputs are invalid")
    freeze_alpha(
        curve_path=paths["curve"],
        controller_manifest_path=paths["controller_manifest"],
        calibration_role_manifest_path=paths["calibration_manifest"],
        evaluator_manifest_path=paths["evaluator_manifest"],
        selector_binding=context.inputs["selector_binding"],
        selection_config=context.inputs["selection_config"],
        evidence_root=context.cell_root,
        output_path=context.outputs["alpha_freeze"],
    )
    return dict(context.outputs)


def _test_gate_handler(context: StageContext) -> Mapping[str, Path]:
    alpha = context.inputs.get("alpha_freeze")
    controller = context.inputs.get("controller_manifest")
    test_role = context.inputs.get("test_role_manifest")
    test_policy = context.inputs.get("test_policy")
    if controller is not None and not isinstance(controller, Path):
        raise FlowExecutionError("test gate controller_manifest must be an artifact")
    if not isinstance(test_role, Path):
        raise FlowExecutionError("test gate requires test_role_manifest")
    if alpha is not None and not isinstance(alpha, Path):
        raise FlowExecutionError("test gate alpha_freeze must be an artifact")
    if test_policy is not None and not isinstance(test_policy, Mapping):
        raise FlowExecutionError("test gate test_policy must be a config object")
    if (alpha is None) == (test_policy is None):
        raise FlowExecutionError(
            "test gate requires exactly one of alpha_freeze or test_policy"
        )
    if len(context.outputs) != 1 or "test_authorization" not in context.outputs:
        raise FlowExecutionError("test gate must output test_authorization")
    authorize_final_test(
        alpha_freeze_path=alpha,
        controller_manifest_path=controller,
        test_role_manifest_path=test_role,
        test_policy=test_policy,
        evidence_root=context.cell_root,
        output_path=context.outputs["test_authorization"],
    )
    return dict(context.outputs)


def _audit_prepare_handler(context: StageContext) -> Mapping[str, Path]:
    bank_plan = context.inputs.get("bank_plan")
    audit_config = context.inputs.get("audit_config")
    components = [
        path
        for name, path in context.inputs.items()
        if name != "component_index"
        and name.startswith("component_")
        and isinstance(path, Path)
    ]
    component_index = context.inputs.get("component_index")
    if component_index is not None:
        if components or not isinstance(component_index, Path):
            raise FlowExecutionError(
                "post-training audit uses either component_index or component_* inputs"
            )
        components = _component_manifest_paths(
            component_index, evidence_root=context.cell_root
        )
    if not isinstance(bank_plan, Path) or not isinstance(audit_config, Mapping):
        raise FlowExecutionError(
            "post-training audit prepare requires bank_plan and audit_config"
        )
    if not components or len(context.outputs) != 1:
        raise FlowExecutionError(
            "post-training audit prepare requires component_* inputs and one output"
        )
    name, output = next(iter(context.outputs.items()))
    prepare_post_training_audit(
        bank_plan_path=bank_plan,
        component_manifest_paths=components,
        audit_config=_bind_artifacts(audit_config, context.stage),
        output_path=output,
    )
    return {name: output}


def _audit_execute_handler(context: StageContext) -> Mapping[str, Path]:
    audit_plan = context.inputs.get("audit_plan")
    source_bank_plan = context.inputs.get("source_bank_plan")
    audit_role = context.inputs.get("audit_role_manifest")
    bindings = {
        name: context.inputs.get(name)
        for name in (
            "bank_backend_binding",
            "generation_backend_binding",
            "evaluator_binding",
        )
    }
    values = {
        name: context.inputs.get(name)
        for name in (
            "model_config",
            "generation_config",
            "evaluation_config",
            "trajectory_config",
        )
    }
    if not all(
        isinstance(path, Path) for path in (audit_plan, source_bank_plan, audit_role)
    ):
        raise FlowExecutionError(
            "post-training audit execution requires audit, bank, and role artifacts"
        )
    if not all(isinstance(item, Mapping) for item in bindings.values()):
        raise FlowExecutionError("post-training audit bindings must be objects")
    if not all(isinstance(item, Mapping) for item in values.values()):
        raise FlowExecutionError("post-training audit config must be objects")
    if set(context.outputs) != {"audit_packet", "decision_key"}:
        raise FlowExecutionError("post-training audit outputs must be packet and key")
    baseline = context.runtime_config.get("baseline")
    if not isinstance(baseline, Mapping) or not isinstance(
        baseline.get("parameters"), Mapping
    ):
        raise FlowExecutionError("post-training audit requires a baseline plan")
    method_plan = {"method": baseline.get("method"), **dict(baseline["parameters"])}
    stage_id = context.stage.get("stage_id")
    run = load_run(context.cell_root)
    stage_state = run.get("stages", {}).get(stage_id)
    attempt_id = (
        stage_state.get("current_attempt") if isinstance(stage_state, Mapping) else None
    )
    if not isinstance(attempt_id, str) or not attempt_id:
        raise FlowExecutionError("audit execution has no active stage attempt")
    attempt_dir = context.outputs["audit_packet"].parent / "_attempts" / attempt_id
    attempt_dir.mkdir(parents=True, exist_ok=False)
    packet_path = attempt_dir / "audit_packet.json"
    decision_key_path = attempt_dir / "decision_key.json"
    execute_post_training_audit(
        audit_plan_path=audit_plan,
        source_bank_plan_path=source_bank_plan,
        audit_role_manifest_path=audit_role,
        method_plan=method_plan,
        bank_backend_binding=bindings["bank_backend_binding"],
        generation_backend_binding=bindings["generation_backend_binding"],
        evaluator_binding=bindings["evaluator_binding"],
        model_config=values["model_config"],
        generation_config=values["generation_config"],
        evaluation_config=values["evaluation_config"],
        trajectory_config=values["trajectory_config"],
        execution_config=context.runtime_config["shared"]["execution"],
        evidence_root=context.cell_root,
        output_dir=attempt_dir,
        packet_path=packet_path,
        decision_key_path=decision_key_path,
    )
    published = (
        (packet_path, context.outputs["audit_packet"]),
        (decision_key_path, context.outputs["decision_key"]),
        (
            attempt_dir / "decision_template.csv",
            context.outputs["audit_packet"].parent / "decision_template.csv",
        ),
    )
    for source, output in published:
        if not source.is_file():
            raise FlowExecutionError(f"audit execution did not create: {source}")
        if output.exists():
            raise FlowExecutionError(f"immutable audit output already exists: {output}")
        os.link(source, output)
    return dict(context.outputs)


def _audit_freeze_handler(context: StageContext) -> Mapping[str, Path]:
    audit_plan = context.inputs.get("audit_plan")
    bank_plan = context.inputs.get("source_bank_plan")
    decisions = context.inputs.get("decision_manifest")
    decision_key = context.inputs.get("decision_key")
    audit_packet = context.inputs.get("audit_packet")
    if not all(
        isinstance(path, Path)
        for path in (audit_plan, bank_plan, decisions, decision_key, audit_packet)
    ):
        raise FlowExecutionError(
            "post-training audit freeze requires plan, bank, packet, key, and decisions"
        )
    if len(context.outputs) != 1:
        raise FlowExecutionError("post-training audit freeze must declare one output")
    name, output = next(iter(context.outputs.items()))
    resolved_decisions = output.parent / "resolved_decisions.json"
    resolve_blinded_audit_decisions(
        audit_plan_path=audit_plan,
        decision_key_path=decision_key,
        decision_manifest_path=decisions,
        output_path=resolved_decisions,
    )
    finalize_post_training_audit(
        audit_plan_path=audit_plan,
        source_bank_plan_path=bank_plan,
        decision_manifest_path=resolved_decisions,
        output_path=output,
    )
    return {name: output}


def _manual_audit_handler(context: StageContext) -> Mapping[str, Path]:
    candidate = context.inputs.get("candidate_manifest")
    gate = context.inputs.get("gate_config")
    if not isinstance(candidate, Path) or not isinstance(gate, Mapping):
        raise FlowExecutionError(
            "manual audit requires candidate_manifest and gate_config"
        )
    required_outputs = {"request", "decision", "approval"}
    if set(context.outputs) != required_outputs:
        raise FlowExecutionError(
            "manual audit outputs must be request, decision, and approval"
        )
    gate_id = gate.get("gate_id")
    decision_schema = gate.get("decision_schema")
    if not isinstance(gate_id, str) or not isinstance(decision_schema, str):
        raise FlowExecutionError("manual audit gate_id and decision_schema are required")
    request = context.outputs["request"]
    decision = context.outputs["decision"]
    approval = context.outputs["approval"]
    if not request.exists():
        create_audit_request(
            gate_id=gate_id,
            candidate_manifest=candidate,
            protocol=_protocol_reference(context.runtime_config),
            decision_schema=decision_schema,
            output_path=request,
        )
        await_audit(context.cell_root, gate_id=gate_id, request_path=request)
        raise FlowAwaitingAudit(
            f"run is awaiting audit gate {gate_id}; submit a bound decision to resume"
        )
    if not decision.is_file() or not approval.is_file():
        raise FlowAwaitingAudit(f"audit gate {gate_id} has not been approved")
    validate_audit_approval(
        approval,
        request_path=request,
        candidate_manifest=candidate,
        decision_manifest=decision,
    )
    return {name: context.outputs[name] for name in sorted(required_outputs)}


def _method_handler(context: StageContext) -> Mapping[str, Path]:
    handler = context.stage["handler"]
    entrypoint = context.cell_root.parents[1] / handler["entrypoint"]
    if not entrypoint.is_file():
        # The cell root may be outside the repository; use this module's repository.
        entrypoint = Path(__file__).resolve().parents[2] / handler["entrypoint"]
    command = [
        sys.executable,
        str(entrypoint),
        "--config",
        str(context.runtime_config_path),
        "--job-id",
        context.runtime_config["cell"]["cell_id"],
        "--artifact-root",
        str(context.cell_root),
        "--device",
        str(context.runtime_config["shared"]["execution"]["device"]),
    ]
    operation = handler["operation"]
    if context.runtime_config["baseline"]["method"] == "cast":
        command.extend(["--stage", operation])
        training_group = context.inputs.get("training_group")
        if training_group is not None:
            if not isinstance(training_group, str) or not training_group:
                raise FlowExecutionError("training_group input must be a non-empty string")
            command.extend(["--training-group", training_group])
    environment = os.environ.copy()
    repository = Path(__file__).resolve().parents[2]
    environment["PYTHONPATH"] = str(repository / "src") + os.pathsep + environment.get(
        "PYTHONPATH", ""
    )
    subprocess.run(command, cwd=repository, env=environment, check=True)
    return context.outputs


DEFAULT_HANDLERS: dict[str, StageHandler] = {
    "dataset_controller": _dataset_handler,
    "position_search": _position_search_handler,
    "position_controller": _position_handler,
    "position_freeze": _position_freeze_handler,
    "post_training_audit_prepare": _audit_prepare_handler,
    "post_training_audit_execute": _audit_execute_handler,
    "post_training_audit_freeze": _audit_freeze_handler,
    "manual_audit": _manual_audit_handler,
    "bank_execute": _bank_execute_handler,
    "bank_finalize_audit": _bank_finalize_audit_handler,
    "generation": _generation_handler,
    "evaluation": _evaluator_handler,
    "alpha_freeze": _alpha_freeze_handler,
    "test_gate": _test_gate_handler,
    "method_train": _method_handler,
    "method_extract": _method_handler,
    "method_audit": _method_handler,
    "method_finalize": _method_handler,
    "method_calibrate": _method_handler,
    "method_test": _method_handler,
    "method_run": _method_handler,
}


def execute_flow(
    cell_root: Path,
    *,
    handler_overrides: Mapping[str, StageHandler] | None = None,
    freeze: bool = False,
) -> dict[str, Any]:
    """Execute or resume every compiled stage in registered order."""

    load_component_catalog()

    cell_root = cell_root.expanduser().resolve()
    config_dir = cell_root / "config"
    runtime_config_path = config_dir / "runtime_config.local.json"
    runtime_flow_path = config_dir / "runtime_flow.local.json"
    flow_manifest = _load_object(config_dir / "flow_manifest.json", field="flow manifest")
    runtime_config = _load_object(runtime_config_path, field="runtime config")
    runtime_flow = _load_object(runtime_flow_path, field="runtime flow")
    stages = runtime_flow.get("stages")
    if not isinstance(stages, list) or not stages:
        raise FlowExecutionError("runtime flow has no stages")
    stage_ids = [stage["stage_id"] for stage in stages]
    run_path = cell_root / "run_manifest.json"
    execution = runtime_config.get("shared", {}).get("execution", {})
    expected_identity = {
        "run_id": flow_manifest["cell_id"],
        "cell_id": flow_manifest["cell_id"],
        "bundle_id": flow_manifest["bundle_id"],
        "experiment": flow_manifest["experiment"],
        "execution_profile": flow_manifest["execution_profile"],
        "protocol": _protocol_reference(runtime_config),
    }
    if execution.get("profile") != expected_identity["execution_profile"]:
        raise FlowExecutionError("runtime config and flow manifest execution profiles differ")
    if not run_path.exists():
        repository = Path(__file__).resolve().parents[2]
        code_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        allocate_run(
            cell_root,
            identity={**expected_identity, "code_commit": code_commit},
            stage_ids=stage_ids,
            allowed_existing_names=("config",),
        )
    else:
        identity = load_run(cell_root).get("identity", {})
        if any(identity.get(key) != value for key, value in expected_identity.items()):
            raise FlowExecutionError(
                "existing run identity differs from the compiled experiment/profile/cell"
            )
    handlers = {**DEFAULT_HANDLERS, **dict(handler_overrides or {})}
    reuse_config = execution.get("evidence_reuse")
    reuse_manifest: Mapping[str, Any] | None = None
    reusable_stage_ids: set[str] = set()
    if reuse_config is not None:
        if not isinstance(reuse_config, Mapping):
            raise FlowExecutionError("execution.evidence_reuse must be an object")
        manifest_path = reuse_config.get("manifest")
        through = reuse_config.get("reuse_through")
        evidence_id = reuse_config.get("evidence_id")
        if (
            not isinstance(manifest_path, str)
            or not Path(manifest_path).is_absolute()
            or not isinstance(through, str)
            or not isinstance(evidence_id, str)
            or not evidence_id
        ):
            raise FlowExecutionError(
                "evidence_reuse requires evidence_id, reuse_through and a registered absolute manifest"
            )
        try:
            loaded = load_reuse_manifest(Path(manifest_path))
            if loaded.get("evidence_id") != evidence_id:
                raise ExternalEvidenceError("evidence_id differs from the imported manifest")
            if loaded.get("target_cell_id") != flow_manifest.get("cell_id"):
                raise ExternalEvidenceError(
                    "external evidence is not registered for the current matrix cell"
                )
            reusable_stage_ids = set(
                validate_reuse_plan(
                    manifest=loaded, flow_stages=stages, through=through
                )
            )
        except ExternalEvidenceError as exc:
            raise FlowExecutionError(str(exc)) from exc
        reuse_manifest = loaded
    for stage in stages:
        run = load_run(cell_root)
        stage_id = stage["stage_id"]
        if run["stages"][stage_id]["status"] == "completed":
            continue
        if run["status"] == "awaiting_audit":
            return run
        handler_id = stage["handler_id"]
        handler = handlers.get(handler_id)
        if handler is None:
            raise FlowExecutionError(f"no execution adapter for handler {handler_id!r}")
        inputs, input_files = _input_values(stage)
        outputs = {
            name: Path(value["absolute_path"])
            for name, value in stage["outputs"].items()
        }
        stage_state = run["stages"][stage_id]
        if stage_id in reusable_stage_ids:
            if reuse_manifest is None:
                raise FlowExecutionError("external evidence reuse manifest is unavailable")
            if stage_state["status"] == "running":
                attempt = stage_state["current_attempt"]
            else:
                attempt = start_stage(
                    cell_root, stage_id=stage_id, input_files=input_files
                )
            try:
                imported = import_stage_payload(
                    manifest=reuse_manifest,
                    source_stage=reuse_manifest["stages"][stage_id],
                    target_stage=stage,
                    cell_root=cell_root,
                )
                if set(imported) != set(outputs):
                    raise FlowExecutionError(
                        f"imported stage {stage_id} output names drift"
                    )
                complete_imported_stage(
                    cell_root,
                    stage_id=stage_id,
                    attempt_id=attempt,
                    output_files=imported,
                    evidence_id=str(reuse_manifest["evidence_id"]),
                    source=reuse_manifest.get("source", {}),
                )
            except Exception as exc:
                fail_stage(
                    cell_root,
                    stage_id=stage_id,
                    attempt_id=attempt,
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
                raise
            continue
        if stage_state["status"] == "running" and run["status"] == "approved":
            attempt = stage_state["current_attempt"]
        else:
            attempt = start_stage(
                cell_root, stage_id=stage_id, input_files=input_files
            )
        context = StageContext(
            cell_root=cell_root,
            runtime_config_path=runtime_config_path,
            runtime_config=runtime_config,
            stage=stage,
            inputs=inputs,
            outputs=outputs,
        )
        try:
            produced = dict(handler(context))
            if set(produced) != set(outputs):
                raise FlowExecutionError(
                    f"stage {stage_id} output names drift: expected {sorted(outputs)}, got {sorted(produced)}"
                )
            missing = [str(path) for path in produced.values() if not path.is_file()]
            if missing:
                raise FlowExecutionError(f"stage {stage_id} did not create outputs: {missing}")
            complete_stage(
                cell_root,
                stage_id=stage_id,
                attempt_id=attempt,
                output_files=produced,
            )
        except FlowAwaitingAudit:
            return load_run(cell_root)
        except Exception as exc:
            fail_stage(
                cell_root,
                stage_id=stage_id,
                attempt_id=attempt,
                error_type=type(exc).__name__,
                message=str(exc),
            )
            raise
    if freeze:
        freeze_run(cell_root, required_stage_ids=stage_ids)
    return load_run(cell_root)



def _normalize_audit_decision_source(decision_source: Path, output_path: Path) -> None:
    """Convert an operator CSV or JSON decision into the gate JSON artifact."""

    suffix = decision_source.suffix.lower()
    if suffix == ".csv":
        with decision_source.open(encoding="utf-8", newline="") as handle:
            rows: Any = list(csv.DictReader(handle))
    elif suffix == ".json":
        rows = json.loads(decision_source.read_text(encoding="utf-8"))
        if isinstance(rows, Mapping):
            rows = rows.get("decisions")
    else:
        raise FlowExecutionError("audit decision source must be CSV or JSON")
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise FlowExecutionError("audit decision source must contain object rows")
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump({"decisions": [dict(row) for row in rows]}, handle, ensure_ascii=False, indent=2)
        handle.write(chr(10))

def approve_flow_audit(
    cell_root: Path,
    *,
    gate_id: str,
    decision_source: Path,
    auditor_id: str,
) -> dict[str, Any]:
    """Seal one external audit decision into its paused flow and approve it."""

    cell_root = cell_root.expanduser().resolve()
    run = load_run(cell_root)
    if run.get("status") != "awaiting_audit" or gate_id not in run.get(
        "audit_gates", {}
    ):
        raise FlowExecutionError(f"flow is not awaiting gate {gate_id}")
    runtime_flow = _load_object(
        cell_root / "config" / "runtime_flow.local.json", field="runtime flow"
    )
    matches = []
    for stage in runtime_flow["stages"]:
        gate_input = stage["inputs"].get("gate_config")
        if (
            stage["handler_id"] == "manual_audit"
            and gate_input
            and gate_input["kind"] == "config"
            and gate_input["value"].get("gate_id") == gate_id
        ):
            matches.append(stage)
    if len(matches) != 1:
        raise FlowExecutionError(f"manual audit stage is ambiguous for gate {gate_id}")
    stage = matches[0]
    outputs = {
        name: Path(value["absolute_path"]) for name, value in stage["outputs"].items()
    }
    candidate_input = stage["inputs"].get("candidate_manifest")
    if not candidate_input or candidate_input["kind"] != "artifact":
        raise FlowExecutionError("manual audit candidate input is not an artifact")
    candidate = Path(candidate_input["absolute_path"])
    decision = outputs["decision"]
    decision.parent.mkdir(parents=True, exist_ok=True)
    if decision.exists():
        raise FlowExecutionError(f"immutable decision already exists: {decision}")
    if not decision_source.is_file():
        raise FlowExecutionError(f"decision source does not exist: {decision_source}")
    _normalize_audit_decision_source(decision_source, decision)
    approval = outputs["approval"]
    approve_audit(
        request_path=outputs["request"],
        candidate_manifest=candidate,
        decision_manifest=decision,
        auditor_id=auditor_id,
        output_path=approval,
    )
    validate_audit_approval(
        approval,
        request_path=outputs["request"],
        candidate_manifest=candidate,
        decision_manifest=decision,
    )
    return register_audit_approval(
        cell_root, gate_id=gate_id, approval_path=approval
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute one compiled experiment cell.")
    parser.add_argument("--cell-root", type=Path, required=True)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--approve-gate")
    parser.add_argument("--decision", type=Path)
    parser.add_argument("--auditor-id")
    args = parser.parse_args()
    if args.approve_gate:
        if args.decision is None or not args.auditor_id:
            parser.error("--approve-gate requires --decision and --auditor-id")
        result = approve_flow_audit(
            args.cell_root,
            gate_id=args.approve_gate,
            decision_source=args.decision,
            auditor_id=args.auditor_id,
        )
    else:
        result = execute_flow(args.cell_root, freeze=args.freeze)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
