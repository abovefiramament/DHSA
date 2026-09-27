"""Compile one validated experiment cell into a closed, traceable flow plan.

This module never chooses scientific values. It binds a validated configuration
bundle to fixed repository handlers and records every config/artifact edge.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping

from baseline.controllers.bank_controller import build_bank_plan
from baseline.controllers.parameter_controller import resolve_method_parameters
from baseline.controllers.position_controller import resolve_positions
from experiments.shared.config_bundle import validate_bundle
from experiments.shared.runtime_registry import load_runtime_registry, resolve_runtime_paths


class ExperimentFlowError(ValueError):
    """Raised when a configured flow is not closed or uses an unknown handler."""


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_HANDLERS = {
    "confiqa": "data/confiqa/controller.py",
    "imdb": "data/imdb/controller.py",
    "tldr": "data/tldr/controller.py",
    "tulu": "data/tulu/controller.py",
}
METHOD_HANDLERS = {
    "direct_policy": "experiments/shared/generation_controller.py",
    "cast": "baseline/implementations/cast.py",
    "rcm_zero": "baseline/implementations/rcm_zero.py",
    "rcm_patch": "baseline/implementations/rcm_patch.py",
    "iti": "baseline/implementations/iti.py",
    "random": "baseline/implementations/random.py",
}
STATIC_HANDLERS = {
    "position_search": {
        "entrypoint": "experiments/shared/position_search_controller.py",
        "operation": "run_registered_position_backend",
    },
    "position_controller": {
        "entrypoint": "baseline/controllers/position_controller.py",
        "operation": "resolve",
    },
    "position_freeze": {
        "entrypoint": "baseline/controllers/position_freeze_controller.py",
        "operation": "freeze_position_plan",
    },
    "post_training_audit_prepare": {
        "entrypoint": "experiments/shared/component_runtime.py",
        "operation": "prepare_post_training_audit",
    },
    "post_training_audit_execute": {
        "entrypoint": "experiments/shared/post_training_audit_execution.py",
        "operation": "execute_post_training_audit",
    },
    "post_training_audit_freeze": {
        "entrypoint": "experiments/shared/component_runtime.py",
        "operation": "finalize_post_training_audit",
    },
    "manual_audit": {
        "entrypoint": "experiments/shared/manual_audit.py",
        "operation": "pause_and_approve",
    },
    "bank_execute": {
        "entrypoint": "baseline/controllers/bank_execution_controller.py",
        "operation": "execute_registered_bank_plan",
    },
    "bank_finalize_audit": {
        "entrypoint": "baseline/controllers/bank_execution_controller.py",
        "operation": "finalize_audited_bank_plan",
    },
    "evaluation": {
        "entrypoint": "experiments/shared/evaluation_controller.py",
        "operation": "run_registered_evaluator",
    },
    "generation": {
        "entrypoint": "experiments/shared/generation_controller.py",
        "operation": "run_registered_generator",
    },
    "alpha_freeze": {
        "entrypoint": "experiments/shared/evaluation_controller.py",
        "operation": "freeze_alpha",
    },
    "test_gate": {
        "entrypoint": "experiments/shared/evaluation_controller.py",
        "operation": "authorize_final_test",
    },
}
METHOD_STAGE_OPERATIONS = {
    "direct_policy": {},
    "cast": {
        "method_train": "train",
        "method_extract": "extract",
        "method_audit": "audit",
        "method_finalize": "finalize",
        "method_calibrate": "dev",
        "method_test": "test",
    },
    "rcm_zero": {"method_run": "select"},
    "rcm_patch": {"method_run": "select"},
    "iti": {"method_run": "select"},
    "random": {"method_run": "select"},
}
METHOD_STAGE_IDS = frozenset().union(*METHOD_STAGE_OPERATIONS.values())


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _config_value(config: Mapping[str, Any], reference: str) -> Any:
    if not reference.startswith("config://"):
        raise ExperimentFlowError(f"not a config reference: {reference}")
    path = reference[len("config://") :]
    if not path:
        raise ExperimentFlowError("config reference cannot be empty")
    value: Any = config
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ExperimentFlowError(f"unknown config reference: {reference}")
        value = value[part]
    return copy.deepcopy(value)


def _expand_config_refs(value: Any, root: Mapping[str, Any]) -> Any:
    if isinstance(value, Mapping):
        if set(value) == {"$ref"}:
            reference = value["$ref"]
            if not isinstance(reference, str) or not reference.startswith("config://"):
                raise ExperimentFlowError("$ref must contain one config:// reference")
            return _config_value(root, reference)
        return {key: _expand_config_refs(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_config_refs(item, root) for item in value]
    return copy.deepcopy(value)


def _portable_runtime_value(
    value: Any,
    used_registry: Mapping[str, Mapping[str, Any]],
) -> Any:
    reverse = {
        entry["path"]: f"registry://{key}"
        for key, entry in used_registry.items()
    }

    def convert(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {
                key: convert(child)
                for key, child in item.items()
                if key != "absolute_path"
            }
        if isinstance(item, list):
            return [convert(child) for child in item]
        if isinstance(item, str):
            if item in reverse:
                return reverse[item]
            if PurePosixPath(item).is_absolute() or PureWindowsPath(item).is_absolute():
                raise ExperimentFlowError(f"unregistered absolute path in public manifest: {item}")
        return copy.deepcopy(item)

    return convert(value)

def _relative_output(value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ExperimentFlowError(f"{field} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path == Path("."):
        raise ExperimentFlowError(f"{field} must remain under the cell evidence root")
    return path


def _handler(handler_id: str, *, dataset: str, method: str) -> dict[str, str]:
    if handler_id == "dataset_controller":
        entrypoint = DATASET_HANDLERS.get(dataset)
        if entrypoint is None:
            raise ExperimentFlowError(f"no fixed dataset controller for {dataset!r}")
        result = {"entrypoint": entrypoint, "operation": "materialize"}
    elif handler_id in METHOD_STAGE_IDS:
        entrypoint = METHOD_HANDLERS.get(method)
        if entrypoint is None:
            raise ExperimentFlowError(f"no fixed method handler for {method!r}")
        operation = METHOD_STAGE_OPERATIONS[method].get(handler_id)
        if operation is None:
            raise ExperimentFlowError(
                f"handler {handler_id!r} is not valid for method {method!r}"
            )
        result = {
            "entrypoint": entrypoint,
            "operation": operation,
        }
    elif handler_id in STATIC_HANDLERS:
        result = dict(STATIC_HANDLERS[handler_id])
    else:
        raise ExperimentFlowError(f"unknown fixed handler: {handler_id}")
    if not (REPO_ROOT / result["entrypoint"]).is_file():
        raise ExperimentFlowError(f"registered handler does not exist: {result['entrypoint']}")
    return result


def _artifact_key(reference: str) -> tuple[str, str]:
    if not reference.startswith("artifact://"):
        raise ExperimentFlowError(f"not an artifact reference: {reference}")
    parts = reference[len("artifact://") :].split("/")
    if len(parts) != 2 or not all(parts):
        raise ExperimentFlowError(
            f"artifact reference must be artifact://<stage>/<output>: {reference}"
        )
    return parts[0], parts[1]


def _output_segment(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or value in {".", ".."}:
        raise ExperimentFlowError(f"{field} must be a non-empty path segment")
    if PurePosixPath(value).name != value or PureWindowsPath(value).name != value:
        raise ExperimentFlowError(f"{field} must not contain path separators")
    return value


def _symbolic_artifacts(value: Any):
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == "artifact" and isinstance(item, str):
                yield item
            else:
                yield from _symbolic_artifacts(item)
    elif isinstance(value, list):
        for item in value:
            yield from _symbolic_artifacts(item)


def compile_experiment_cell(
    bundle_dir: Path,
    *,
    cell_id: str,
    runtime_registry: Path,
) -> dict[str, Any]:
    """Validate one cell and write its immutable derived planning evidence."""

    bundle_manifest = validate_bundle(bundle_dir)
    protocol_freeze = bundle_manifest.get("protocol_freeze")
    if protocol_freeze is not None and not protocol_freeze.get(
        "execution_authorized", False
    ):
        raise ExperimentFlowError("formal protocol does not authorize execution")
    strict_responsibility_contracts = bool(
        isinstance(protocol_freeze, Mapping) and protocol_freeze.get("formal", False)
    )
    registry_manifest = load_runtime_registry(runtime_registry)
    matches = [cell for cell in bundle_manifest["cells"] if cell["cell_id"] == cell_id]
    if len(matches) != 1:
        raise ExperimentFlowError(f"unknown or ambiguous cell_id: {cell_id}")
    cell = matches[0]
    resolved = copy.deepcopy(cell["resolved_config"])
    bound_config = _expand_config_refs(resolved, resolved)
    portable_method_plan = resolve_method_parameters(bound_config)
    runtime_config, used_registry = resolve_runtime_paths(bound_config, registry_manifest)
    flow = runtime_config["shared"]["flow"]
    if not all(isinstance(stage, Mapping) for stage in flow):
        raise ExperimentFlowError(
            "execution planning requires object-form flow stages with explicit edges"
        )
    dataset = runtime_config["identity"]["dataset"]
    method = runtime_config["baseline"]["method"]
    method_plan = resolve_method_parameters(runtime_config)
    method_kind = method_plan["method_kind"]
    position_control = runtime_config.get("position_control")
    position_is_runtime = bool(
        isinstance(position_control, Mapping)
        and list(_symbolic_artifacts(position_control))
    )
    position_plan = (
        resolve_positions(runtime_config)
        if position_control is not None and not position_is_runtime
        else None
    )
    bank_plan = (
        build_bank_plan(position_plan, method_plan)
        if position_plan is not None and method == "cast"
        else None
    )
    evidence_root = runtime_config["shared"]["execution"].get("evidence_root")
    if not isinstance(evidence_root, str) or not Path(evidence_root).is_absolute():
        raise ExperimentFlowError(
            "shared.execution.evidence_root must resolve through the runtime registry"
        )
    experiment = _output_segment(
        runtime_config.get("identity", {}).get("experiment"),
        field="identity.experiment",
    )
    execution_profile = _output_segment(
        runtime_config.get("shared", {}).get("execution", {}).get("profile"),
        field="shared.execution.profile",
    )
    cell_root = Path(evidence_root) / experiment / execution_profile / cell_id
    declared: dict[tuple[str, str], dict[str, str]] = {}
    used_paths: set[Path] = set()
    compiled_stages: list[dict[str, Any]] = []
    formal_roles = runtime_config["shared"]["data"].get("roles")
    role_outputs = runtime_config["shared"]["data"].get("role_outputs", {})
    final_test_roles = (
        {
            name
            for name, spec in formal_roles.items()
            if isinstance(spec, Mapping) and spec.get("purpose") == "final_test"
        }
        if isinstance(formal_roles, Mapping)
        else set()
    )
    final_test_references: set[str] = set()
    test_gate_references: set[str] = set()

    for index, raw_stage in enumerate(flow):
        stage = dict(raw_stage)
        stage_id = stage["id"]
        handler_id = stage["handler"]
        handler = _handler(handler_id, dataset=dataset, method=method)
        if (
            strict_responsibility_contracts
            and handler_id in {"position_search", "position_controller", "position_freeze"}
            and not isinstance(position_control, Mapping)
        ):
            raise ExperimentFlowError(
                f"{handler_id} requires an explicit position_control"
            )
        if (
            strict_responsibility_contracts
            and handler_id in {"bank_execute", "bank_finalize_audit"}
            and method_kind != "intervention_baseline"
        ):
            raise ExperimentFlowError(
                f"{handler_id} requires an intervention-baseline cell"
            )
        if strict_responsibility_contracts and handler_id in METHOD_STAGE_IDS:
            raise ExperimentFlowError(
                "formal protocols must use typed shared controllers, not direct method stages"
            )
        inputs: dict[str, Any] = {}
        for name, reference in stage["inputs"].items():
            if not isinstance(reference, str):
                raise ExperimentFlowError(
                    f"flow stage {stage_id} input {name} must be a reference string"
                )
            if reference.startswith("config://"):
                inputs[name] = {
                    "reference": reference,
                    "kind": "config",
                    "value": _config_value(runtime_config, reference),
                }
            elif reference.startswith("artifact://"):
                key = _artifact_key(reference)
                if key not in declared:
                    raise ExperimentFlowError(
                        f"stage {stage_id} reads unavailable prior artifact {reference}"
                    )
                inputs[name] = {
                    "reference": reference,
                    "kind": "artifact",
                    **declared[key],
                }
                if reference in final_test_references and handler_id != "test_gate":
                    raise ExperimentFlowError(
                        f"final-test role {reference} may be consumed only by test_gate"
                    )
            else:
                raise ExperimentFlowError(
                    f"stage {stage_id} input {name} must use config:// or artifact://"
                )
        outputs: dict[str, dict[str, str]] = {}
        for name, raw_path in stage["outputs"].items():
            relative = _relative_output(
                raw_path, field=f"flow[{index}].outputs.{name}"
            )
            if relative in used_paths:
                raise ExperimentFlowError(f"duplicate flow output path: {relative}")
            used_paths.add(relative)
            value = {
                "reference": f"artifact://{stage_id}/{name}",
                "relative_path": relative.as_posix(),
                "absolute_path": str(cell_root / relative),
                "producer_handler_id": handler_id,
            }
            declared[(stage_id, name)] = value
            outputs[name] = value
            if (
                handler_id == "dataset_controller"
                and isinstance(role_outputs, Mapping)
                and role_outputs.get(name) in final_test_roles
            ):
                final_test_references.add(value["reference"])
            if handler_id == "test_gate" and name == "test_authorization":
                test_gate_references.add(value["reference"])
        compiled_stages.append(
            {
                "index": index,
                "stage_id": stage_id,
                "handler_id": handler_id,
                "handler": handler,
                "inputs": inputs,
                "outputs": outputs,
            }
        )

    if final_test_roles:
        gates = [
            stage for stage in compiled_stages if stage["handler_id"] == "test_gate"
        ]
        if len(gates) != 1:
            raise ExperimentFlowError(
                "named final_test data requires exactly one test_gate stage"
            )
        gate_inputs = {
            item["reference"]
            for item in gates[0]["inputs"].values()
            if item["kind"] == "artifact"
        }
        if not final_test_references.issubset(gate_inputs):
            raise ExperimentFlowError(
                "test_gate must consume every declared final-test role artifact"
            )
        authorized_generation = 0
        for stage in compiled_stages:
            if stage["handler_id"] not in {"method_test", "generation"}:
                continue
            stage_inputs = {
                item["reference"]
                for item in stage["inputs"].values()
                if item["kind"] == "artifact"
            }
            consumes_authorization = bool(
                stage_inputs.intersection(test_gate_references)
            )
            if stage["handler_id"] == "method_test" and not consumes_authorization:
                raise ExperimentFlowError(
                    "method_test must consume the registered test_authorization"
                )
            if stage["handler_id"] == "generation" and consumes_authorization:
                authorized_generation += 1
        if authorized_generation != 1:
            raise ExperimentFlowError(
                "named final_test data requires exactly one generation stage bound to test_authorization"
            )

    portable_stages = _portable_runtime_value(compiled_stages, used_registry)
    portable_position_plan = (
        _portable_runtime_value(position_plan, used_registry)
        if position_plan is not None else None
    )
    portable_bank_plan = (
        _portable_runtime_value(bank_plan, used_registry)
        if bank_plan is not None else None
    )
    method_input_refs = {
        item["reference"]
        for stage in compiled_stages
        if stage["handler_id"]
        in METHOD_STAGE_IDS
        | {
            "bank_execute",
            "bank_finalize_audit",
            "generation",
            "test_gate",
        }
        for item in stage["inputs"].values()
        if item["kind"] == "artifact"
    }
    position_input_refs = {
        item["reference"]
        for stage in compiled_stages
        if stage["handler_id"] == "position_controller"
        for item in stage["inputs"].values()
        if item["kind"] == "artifact"
    }

    for reference in _symbolic_artifacts(method_plan):
        if _artifact_key(reference) not in declared:
            raise ExperimentFlowError(
                f"method parameter references undeclared artifact: {reference}"
            )
        if reference not in method_input_refs:
            raise ExperimentFlowError(
                f"method parameter artifact is not an explicit method-stage input: {reference}"
            )
    if position_control is not None:
        for reference in _symbolic_artifacts(position_control):
            if _artifact_key(reference) not in declared:
                raise ExperimentFlowError(
                    f"position parameter references undeclared artifact: {reference}"
                )
            if reference not in position_input_refs:
                raise ExperimentFlowError(
                    "position parameter artifact is not an explicit position-stage input: "
                    f"{reference}"
                )

    planning_dir = cell_root / "config"
    flow_manifest = {
        "schema_version": 1,
        "bundle_id": bundle_manifest["bundle_id"],
        "cell_id": cell_id,
        "experiment": experiment,
        "execution_profile": execution_profile,
        "runtime_bindings": sorted(used_registry),
        "status": "planned",
        "protocol": copy.deepcopy(runtime_config["protocol"]),
        "position_resolution": "runtime" if position_is_runtime else "compile_time",
        "stages": portable_stages,
    }
    _write_json(planning_dir / "config_bundle_manifest.json", bundle_manifest)
    if protocol_freeze is not None:
        _write_json(planning_dir / "protocol_freeze.json", protocol_freeze)
    _write_json(planning_dir / "resolved_config.json", resolved)
    _write_json(planning_dir / "runtime_config.local.json", runtime_config)
    _write_json(planning_dir / "method_plan.json", portable_method_plan)
    _write_json(planning_dir / "method_plan.local.json", method_plan)
    _write_json(
        planning_dir / "runtime_registry_manifest.local.json",
        {
            "machine_id": registry_manifest["machine_id"],
            "used_entries": used_registry,
        },
    )
    _write_json(
        planning_dir / "runtime_flow.local.json",
        {
            "machine_id": registry_manifest["machine_id"],
            "stages": compiled_stages,
        },
    )
    if position_plan is not None:
        _write_json(planning_dir / "position_plan.json", portable_position_plan)
        _write_json(planning_dir / "position_plan.local.json", position_plan)
        if bank_plan is not None:
            _write_json(planning_dir / "bank_plan.json", portable_bank_plan)
    _write_json(planning_dir / "flow_manifest.json", flow_manifest)
    # This return-only field lets adapters locate the compiled private
    # evidence root without assuming a single outputs/* registry entry.  It
    # is intentionally excluded from the portable on-disk manifest.
    return {**flow_manifest, "runtime_cell_root": str(cell_root)}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile a configuration bundle cell into a closed flow plan."
    )
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--runtime-registry", type=Path, required=True)
    args = parser.parse_args()
    manifest = compile_experiment_cell(
        args.bundle_dir,
        cell_id=args.cell_id,
        runtime_registry=args.runtime_registry,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
