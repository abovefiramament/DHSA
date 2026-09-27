"""Site cell compilation for the new typed experiment framework.

The entrypoint owns cell identity and dispatch only. Dataset materialization,
position selection, bank lifecycle, audit, generation, evaluation and evidence
freezing are shared registered functions; this module never reimplements them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from experiments.shared.component_catalog import load_component_catalog
from experiments.shared.component_registry import COMPONENT_REGISTRY, ComponentRegistryError
from experiments.shared.experiment_controller import compile_experiment_cell
from experiments.shared.model_registry import resolve_registered_model
from experiments.site.site_protocol_binding import (
    POSITION_METHOD_BY_SELECTOR,
    iter_bound_site_jobs,
    load_formal_site_matrix,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATRIX = REPOSITORY_ROOT / "configs" / "site" / "site_cell_matrix_20260829_v1.json"
ALLOWED_POSITION_METHODS = tuple(POSITION_METHOD_BY_SELECTOR.values())
FIXED_HANDLER_SEQUENCES = {
    "confiqa": ("dataset_controller", "position_search", "position_controller", "position_freeze", "bank_execute", "generation", "evaluation", "alpha_freeze", "test_gate", "generation", "evaluation"),
    "imdb": ("dataset_controller", "position_search", "position_controller", "position_freeze", "bank_execute", "test_gate", "generation", "evaluation"),
    "tldr": ("dataset_controller", "position_search", "position_controller", "position_freeze", "bank_execute", "post_training_audit_prepare", "post_training_audit_execute", "manual_audit", "post_training_audit_freeze", "bank_finalize_audit", "generation", "evaluation", "alpha_freeze", "test_gate", "generation", "evaluation"),
}


class SiteCellError(ValueError):
    """Raised when a formal Site cell cannot be compiled safely."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SiteCellError(f"cannot load JSON file {path}") from exc
    if not isinstance(value, dict):
        raise SiteCellError(f"JSON root must be an object: {path}")
    return value


def _cell_root_from_resolved(resolved: Mapping[str, Any], cell_id: str) -> Path:
    shared = resolved.get("shared", {})
    execution = shared.get("execution", {}) if isinstance(shared, Mapping) else {}
    evidence_root = execution.get("evidence_root") if isinstance(execution, Mapping) else None
    if not isinstance(evidence_root, str) or not Path(evidence_root).is_absolute():
        raise SiteCellError("Site evidence_root must resolve through the runtime registry")
    identity = resolved.get("identity", {})
    experiment = identity.get("experiment") if isinstance(identity, Mapping) else None
    profile = execution.get("profile") if isinstance(execution, Mapping) else None
    for field, value in (("identity.experiment", experiment), ("shared.execution.profile", profile)):
        if (
            not isinstance(value, str)
            or not value
            or value in {".", ".."}
            or Path(value).name != value
        ):
            raise SiteCellError(f"{field} must be one safe path segment")
    return Path(evidence_root).resolve() / experiment / profile / cell_id


def load_site_cell_matrix(path: Path = DEFAULT_MATRIX) -> dict[str, Any]:
    try:
        return load_formal_site_matrix(path)
    except ValueError as exc:
        raise SiteCellError(str(exc)) from exc


def iter_site_cells(matrix_path: Path = DEFAULT_MATRIX) -> tuple[dict[str, Any], ...]:
    """Expand the single formal matrix into its 60 condition jobs."""

    rows = tuple(iter_bound_site_jobs(matrix_path))
    if len(rows) != 60:
        raise SiteCellError(f"formal Site matrix must expand to 60 jobs, got {len(rows)}")
    if len({row["job_id"] for row in rows}) != len(rows):
        raise SiteCellError("formal Site job IDs must be unique")
    return rows


def iter_site_experiment_cells(matrix_path: Path = DEFAULT_MATRIX) -> tuple[dict[str, Any], ...]:
    matrix = load_site_cell_matrix(matrix_path)
    contract = matrix["baseline_family_contract"]
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for cell in matrix["experiment_cells"]:
        key = (cell["dataset"], cell["model_family"])
        if key not in grouped:
            model = resolve_registered_model(
                cell["model_registry_id"], registry_path=matrix["model_registry"]
            )
            grouped[key] = {
                "cell_id": f"{cell['dataset']}__{cell['model_family']}",
                "dataset": cell["dataset"],
                "model_family": cell["model_family"],
                "model": model,
                "strata": [],
                "intervention_family": cell["intervention_family"],
                "scorer": cell["scorer"],
                "baseline_families": list(contract["families"]),
                "condition_variants": dict(contract["variants"]),
            }
        grouped[key]["strata"].append(cell["subset"])
    return tuple(grouped.values())


def resolve_site_cell(cell_id: str, *, matrix_path: Path = DEFAULT_MATRIX) -> dict[str, Any]:
    matrix = load_site_cell_matrix(matrix_path)
    matches = [row for row in iter_site_cells(matrix_path) if row["job_id"] == cell_id]
    if len(matches) != 1:
        raise SiteCellError(f"unknown or ambiguous formal Site job: {cell_id}")
    row = matches[0]
    expected = matrix["slot_contract"]["selector_to_position_method"].get(row["selector"])
    if row["position_method"] != expected:
        raise SiteCellError("selector-to-position binding drift")
    return {"cell": row, "flow_id": matrix["flow_id"], "position_method": row["position_method"], "fixed_slots": list(matrix["slot_contract"]["fixed"])}


def _validate_fixed_flow(bundle_dir: Path, *, dataset: str) -> None:
    master = _load_json(bundle_dir.resolve() / "bundle.json")
    flow = master.get("shared", {}).get("flow")
    if not isinstance(flow, list) or not all(isinstance(stage, Mapping) for stage in flow):
        raise SiteCellError("Site bundle must declare object-form flow stages")
    handlers = tuple(stage.get("handler") for stage in flow)
    if handlers != FIXED_HANDLER_SEQUENCES[dataset]:
        raise SiteCellError(f"fixed Site handler sequence drift for {dataset}")


def _expected_component_kind(handler_id: str, input_name: str) -> str | None:
    if handler_id == "dataset_controller" and input_name == "data_config":
        return "data_adapter"
    if input_name == "evaluator_binding":
        return "evaluator"
    if input_name == "selector_binding":
        return "alpha_selector"
    if handler_id == "position_search" and input_name == "scanner_binding":
        return "position_scanner"
    if input_name != "backend_binding":
        if handler_id == "post_training_audit_execute":
            return {
                "bank_backend_binding": "bank_backend",
                "generation_backend_binding": "generation_backend",
                "evaluator_binding": "evaluator",
            }.get(input_name)
        return None
    return {
        "position_search": "position_backend",
        "bank_execute": "bank_backend",
        "bank_finalize_audit": "bank_backend",
        "native_baseline_train": "native_baseline_backend",
        "generation": "generation_backend",
    }.get(handler_id)


def component_registration_status(cell_root: Path) -> dict[str, Any]:
    runtime_flow = _load_json(cell_root / "config" / "runtime_flow.local.json")
    load_component_catalog()
    requirements: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for stage in runtime_flow.get("stages", []):
        if not isinstance(stage, Mapping):
            continue
        handler_id = str(stage.get("handler_id"))
        for input_name, raw in stage.get("inputs", {}).items():
            if not isinstance(raw, Mapping) or raw.get("kind") != "config":
                continue
            expected_kind = _expected_component_kind(handler_id, str(input_name))
            binding = raw.get("value")
            if expected_kind == "data_adapter" and isinstance(binding, Mapping):
                binding = binding.get("backend_binding")
            if expected_kind is None or not isinstance(binding, Mapping):
                continue
            item = {"stage_id": stage.get("stage_id"), "handler_id": handler_id, "input": input_name, "kind": expected_kind, "binding": dict(binding)}
            try:
                registration = COMPONENT_REGISTRY.resolve(binding, expected_kind=expected_kind)
            except (ComponentRegistryError, TypeError, ValueError) as exc:
                item.update({"status": "pending_external_registration", "reason": str(exc)})
                missing.append(item)
            else:
                item.update({"status": "registered", "registered_version": registration.version})
            requirements.append(item)
    return {"schema_version": 1, "status": "ready" if not missing else "pending_external_registration", "requirements": requirements, "missing": missing}


def compile_site_cell(bundle_dir: Path, *, cell_id: str, runtime_registry: Path, matrix_path: Path = DEFAULT_MATRIX) -> dict[str, Any]:
    slot = resolve_site_cell(cell_id, matrix_path=matrix_path)
    _validate_fixed_flow(bundle_dir, dataset=slot["cell"]["dataset"])
    manifest = compile_experiment_cell(bundle_dir, cell_id=cell_id, runtime_registry=runtime_registry)
    cell_root = Path(manifest["runtime_cell_root"])
    resolved = _load_json(cell_root / "config" / "runtime_config.local.json")
    cell_root = _cell_root_from_resolved(resolved, cell_id)
    expected = slot["cell"]
    identity = resolved.get("identity", {})
    if identity.get("dataset") != expected["dataset"] or identity.get("model_family") != expected["model_family"]:
        raise SiteCellError("compiled Site identity disagrees with formal matrix")
    if resolved.get("baseline", {}).get("method") != "cast":
        raise SiteCellError("Site cells must use baseline.method=cast")
    parameters = resolved.get("baseline", {}).get("parameters", {})
    controller = parameters.get("controller")
    family = controller.get("family") if isinstance(controller, Mapping) else None
    if family != expected["intervention_family"]:
        raise SiteCellError("compiled intervention family disagrees with formal matrix")
    evaluation = resolved.get("shared", {}).get("evaluation", {})
    if evaluation.get("scorer") != expected["scorer"]:
        raise SiteCellError("compiled scorer disagrees with formal matrix")
    position_control = resolved.get("position_control", {})
    declared = position_control.get("method") if isinstance(position_control, Mapping) else None
    if declared is None and isinstance(position_control, Mapping) and isinstance(position_control.get("source"), Mapping):
        declared = position_control["source"].get("method")
    if declared != slot["position_method"]:
        raise SiteCellError("compiled position method disagrees with formal matrix")
    status = component_registration_status(cell_root)
    status.update({"cell_id": cell_id, "position_method": slot["position_method"]})
    status_path = cell_root / "config" / "component_registration_status.json"
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"slot": slot, "flow": manifest, "component_status": status}
