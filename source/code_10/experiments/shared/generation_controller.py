"""Method-neutral inference controller with a mandatory final-test gate."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .component_registry import COMPONENT_REGISTRY, ComponentRegistry
from .contracts import (
    GenerationRequest,
    GenerationResult,
)
from .trajectory_controller import (
    normalize_trajectory_config,
    register_inference_trajectory,
)


class GenerationControllerError(ValueError):
    """Raised when generation inputs are not frozen or mutually consistent."""


def _load(path: Path, *, field: str) -> dict[str, Any]:
    if not path.is_file():
        raise GenerationControllerError(f"{field} does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GenerationControllerError(f"{field} must contain one JSON object")
    return value


def _artifact(path: Path, root: Path, *, field: str) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise GenerationControllerError(f"{field} escapes evidence root") from exc
    if not resolved.is_file():
        raise GenerationControllerError(f"{field} does not exist: {resolved}")
    return {
        "relative_path": relative.as_posix(),
        "size_bytes": resolved.stat().st_size,
    }


def _resolve_artifact(record: Mapping[str, Any], root: Path, *, field: str) -> Path:
    relative = record.get("relative_path")
    if not isinstance(relative, str) or not relative:
        raise GenerationControllerError(f"{field}.relative_path is invalid")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise GenerationControllerError(f"{field} escapes evidence root")
    resolved = root.resolve() / path
    if not resolved.is_file() or resolved.stat().st_size != record.get("size_bytes"):
        raise GenerationControllerError(f"{field} is missing or has the wrong size")
    return resolved


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise GenerationControllerError(f"immutable generation artifact exists: {path}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(rendered)


def run_registered_generator(
    *,
    backend_binding: Mapping[str, Any],
    model_config: Mapping[str, Any],
    generation_config: Mapping[str, Any],
    execution_config: Mapping[str, Any],
    trajectory_config: Mapping[str, Any] | None = None,
    trace_kind: str = "generation_test",
    evidence_root: Path,
    output_dir: Path,
    manifest_path: Path,
    controller_manifest_path: Path | None = None,
    data_role_manifest_path: Path | None = None,
    test_authorization_path: Path | None = None,
    registry: ComponentRegistry = COMPONENT_REGISTRY,
) -> dict[str, Any]:
    """Generate on a non-test role or through one frozen final-test grant."""

    normalized_trajectory = normalize_trajectory_config(trajectory_config)
    authorization_artifact = None
    selected_alpha = generation_config.get("alpha")
    declared_grid = generation_config.get("alpha_grid")
    if declared_grid is not None and (
        not isinstance(declared_grid, list) or not declared_grid
    ):
        raise GenerationControllerError("generation alpha_grid must be a non-empty list")
    selected_alpha_grid = copy.deepcopy(declared_grid)
    if selected_alpha_grid is not None:
        selected_alpha = None
    if test_authorization_path is not None:
        if controller_manifest_path is not None or data_role_manifest_path is not None:
            raise GenerationControllerError(
                "final-test generation receives only test_authorization, not raw test inputs"
            )
        authorization = _load(test_authorization_path, field="test authorization")
        if (
            authorization.get("manifest_type") != "test_authorization"
            or authorization.get("status") != "authorized"
        ):
            raise GenerationControllerError("test authorization is not active")
        controller_record = authorization.get("controller")
        controller_manifest_path = (
            _resolve_artifact(
                controller_record, evidence_root, field="authorized controller"
            )
            if isinstance(controller_record, Mapping)
            else None
        )
        data_role_manifest_path = _resolve_artifact(
            authorization.get("test_role", {}), evidence_root, field="authorized test role"
        )
        mode = authorization.get("mode")
        if mode == "selected_alpha":
            alpha_path = _resolve_artifact(
                authorization.get("alpha_freeze", {}),
                evidence_root,
                field="authorized alpha freeze",
            )
            alpha_freeze = _load(alpha_path, field="alpha freeze")
            if (
                alpha_freeze.get("manifest_type") != "alpha_freeze"
                or alpha_freeze.get("status") != "frozen"
            ):
                raise GenerationControllerError("authorized alpha is not frozen")
            selected_alpha = alpha_freeze.get("decision", {}).get("selected_alpha")
        elif mode == "full_curve":
            policy = authorization.get("generation_policy")
            if not isinstance(policy, Mapping) or policy.get("mode") != "report_full_curve":
                raise GenerationControllerError("authorized full-curve policy is invalid")
            grid = policy.get("alpha_grid")
            if not isinstance(grid, list) or not grid:
                raise GenerationControllerError("authorized full-curve grid is empty")
            selected_alpha = None
            selected_alpha_grid = copy.deepcopy(grid)
        elif mode == "direct_policy":
            policy = authorization.get("generation_policy")
            if policy != {"mode": "direct_policy"}:
                raise GenerationControllerError(
                    "authorized direct-policy generation policy is invalid"
                )
            if controller_manifest_path is not None:
                raise GenerationControllerError(
                    "direct-policy authorization cannot carry a controller"
                )
            selected_alpha = None
            selected_alpha_grid = None
        else:
            raise GenerationControllerError("test authorization mode is unsupported")
        authorization_artifact = _artifact(
            test_authorization_path, evidence_root, field="test_authorization"
        )
    if data_role_manifest_path is None:
        raise GenerationControllerError(
            "generation requires a data role or one test authorization"
        )
    if controller_manifest_path is not None:
        controller = _load(controller_manifest_path, field="controller manifest")
        if (
            controller.get("manifest_type") != "controller_freeze"
            or controller.get("status") != "frozen"
        ):
            raise GenerationControllerError("generation requires a frozen controller")
    role = _load(data_role_manifest_path, field="data role manifest")
    if role.get("status") != "frozen" or role.get("manifest_type") != "data_role_freeze":
        raise GenerationControllerError("generation requires a frozen data role")
    purpose = role.get("purpose")
    if purpose == "final_test" and test_authorization_path is None:
        raise GenerationControllerError("raw final-test role access is forbidden")
    if purpose != "final_test" and test_authorization_path is not None:
        raise GenerationControllerError("test authorization points to a non-test role")

    registration = registry.resolve(
        backend_binding,
        expected_kind="generation_backend",
        required_operations=("generate",),
    )
    result = registration.implementation.generate(
        GenerationRequest(
            controller_manifest=(
                controller_manifest_path.resolve()
                if controller_manifest_path is not None
                else None
            ),
            data_role_manifest=data_role_manifest_path.resolve(),
            model_config=copy.deepcopy(model_config),
            generation_config=copy.deepcopy(generation_config),
            trajectory_config=copy.deepcopy(normalized_trajectory),
            execution_config=copy.deepcopy(execution_config),
            selected_alpha=copy.deepcopy(selected_alpha),
            selected_alpha_grid=(
                tuple(copy.deepcopy(selected_alpha_grid))
                if selected_alpha_grid is not None
                else None
            ),
            evidence_root=evidence_root.resolve(),
            output_dir=output_dir,
        )
    )
    if not isinstance(result, GenerationResult):
        raise GenerationControllerError("generation backend must return GenerationResult")
    trajectory_path = result.trajectory_path
    component_scores_path = result.component_scores_path
    if normalized_trajectory["enabled"]:
        trajectory_path = trajectory_path or output_dir / "trajectory.jsonl"
        component_scores_path = (
            component_scores_path or output_dir / "component_scores.jsonl"
        )
    trajectory = register_inference_trajectory(
        config=normalized_trajectory,
        trajectory_path=trajectory_path,
        component_scores_path=component_scores_path,
        evidence_root=evidence_root,
        trace_kind=trace_kind,
    )
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "backend": {
            "id": registration.component_id,
            "version": registration.version,
            "input_contract": registration.input_contract,
            "output_contract": registration.output_contract,
        },
        "data_role": role["data_role"],
        "data_purpose": purpose,
        "selected_alpha": copy.deepcopy(selected_alpha),
        "selected_alpha_grid": copy.deepcopy(selected_alpha_grid),
        "model_config": copy.deepcopy(dict(model_config)),
        "generation_config": copy.deepcopy(dict(generation_config)),
        "trajectory_config": trajectory["config"],
        "trace_kind": trajectory["trace_kind"],
        "execution_profile": execution_config.get("profile"),
        "controller": (
            _artifact(controller_manifest_path, evidence_root, field="controller")
            if controller_manifest_path is not None
            else None
        ),
        "data_role_manifest": _artifact(data_role_manifest_path, evidence_root, field="data_role"),
        "predictions": _artifact(result.predictions, evidence_root, field="predictions"),
        "backend_execution_manifest": _artifact(
            result.execution_manifest, evidence_root, field="backend execution manifest"
        ),
        "trajectory": trajectory["trajectory"],
        "component_scores": trajectory["component_scores"],
        "trajectory_rows": trajectory["trajectory_rows"],
        "component_score_rows": trajectory["component_score_rows"],
        "test_authorization": authorization_artifact,
    }
    manifest["artifact_closure"] = [
        manifest[key]
        for key in (
            "controller",
            "data_role_manifest",
            "predictions",
            "backend_execution_manifest",
            "trajectory",
            "component_scores",
        )
        if manifest[key] is not None
    ] + ([authorization_artifact] if authorization_artifact is not None else [])
    _write_exclusive(manifest_path, manifest)
    return manifest
