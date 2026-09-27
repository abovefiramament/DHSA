"""Evaluation, alpha-freeze, and final-test authorization controllers."""

from __future__ import annotations

import copy
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .component_registry import COMPONENT_REGISTRY, ComponentRegistry
from .contracts import (
    AlphaSelectionRequest,
    EvaluationRequest,
    EvaluationResult,
)


class EvaluationControllerError(ValueError):
    """Raised when calibration or final evaluation breaks its contract."""


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluationControllerError(f"{field} must be an object")
    return value


def _string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationControllerError(f"{field} must be a non-empty string")
    return value


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise EvaluationControllerError(f"immutable evaluation artifact exists: {path}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(rendered)


def _load_object(path: Path, *, field: str) -> dict[str, Any]:
    if not path.is_file():
        raise EvaluationControllerError(f"{field} does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EvaluationControllerError(f"{field} must contain one JSON object")
    return value


def _validate_data_role(value: Mapping[str, Any], *, purpose: str) -> None:
    if (
        value.get("manifest_type") != "data_role_freeze"
        or value.get("status") != "frozen"
        or value.get("purpose") != purpose
    ):
        raise EvaluationControllerError(
            f"expected one frozen data role with purpose {purpose!r}"
        )


def _validate_controller(value: Mapping[str, Any]) -> None:
    if (
        value.get("manifest_type") != "controller_freeze"
        or value.get("status") != "frozen"
    ):
        raise EvaluationControllerError("controller is not frozen")


def _artifact(path: Path, evidence_root: Path, *, field: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    root = evidence_root.expanduser().resolve()
    if not resolved.is_file():
        raise EvaluationControllerError(f"{field} does not exist: {resolved}")
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise EvaluationControllerError(f"{field} escapes evidence root") from exc
    return {
        "relative_path": relative.as_posix(),
        "size_bytes": resolved.stat().st_size,
    }


def _resolve_artifact(
    record: Mapping[str, Any], evidence_root: Path, *, field: str
) -> Path:
    relative = Path(_string(record.get("relative_path"), field=f"{field}.relative_path"))
    if relative.is_absolute() or ".." in relative.parts:
        raise EvaluationControllerError(f"{field} escapes evidence root")
    path = evidence_root.resolve() / relative
    if (
        not path.is_file()
        or path.stat().st_size != record.get("size_bytes")
    ):
        raise EvaluationControllerError(f"{field} is missing or has the wrong size")
    return path


def _load_rows(path: Path) -> list[Mapping[str, Any]]:
    if not path.is_file():
        raise EvaluationControllerError(f"curve file does not exist: {path}")
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            rows: Any = list(csv.DictReader(handle))
    elif path.suffix.lower() == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(rows, Mapping):
            rows = rows.get("rows")
    elif path.suffix.lower() == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        raise EvaluationControllerError("curve file must be CSV, JSON, or JSONL")
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise EvaluationControllerError("curve file must contain object rows")
    return rows


class DeterministicMetricAlphaSelector:
    """Generic preregistered metric rule with explicit constraints and tie-break."""

    def select(self, request: AlphaSelectionRequest) -> Mapping[str, Any]:
        config = _mapping(request.selection_config, field="selection_config")
        required = {
            "alpha_field",
            "alpha_grid",
            "primary_metric",
            "direction",
            "constraints",
            "near_best_tolerance",
            "tie_break",
        }
        if set(config) != required:
            raise EvaluationControllerError(
                "alpha selection fields mismatch: "
                f"missing={sorted(required-set(config))} extra={sorted(set(config)-required)}"
            )
        alpha_field = _string(config["alpha_field"], field="alpha_field")
        metric = _string(config["primary_metric"], field="primary_metric")
        direction = config["direction"]
        if direction not in {"maximize", "minimize"}:
            raise EvaluationControllerError("direction must be maximize or minimize")
        tie_break = config["tie_break"]
        if tie_break not in {"smallest_alpha", "largest_alpha"}:
            raise EvaluationControllerError(
                "tie_break must be smallest_alpha or largest_alpha"
            )
        tolerance = config["near_best_tolerance"]
        if not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool) or tolerance < 0:
            raise EvaluationControllerError("near_best_tolerance must be non-negative")
        grid = config["alpha_grid"]
        if not isinstance(grid, list) or not grid:
            raise EvaluationControllerError("alpha_grid must be non-empty")
        expected = [float(value) for value in grid]
        if len(expected) != len(set(expected)):
            raise EvaluationControllerError("alpha_grid contains duplicates")
        by_alpha: dict[float, Mapping[str, Any]] = {}
        for index, row in enumerate(request.curve_rows):
            try:
                alpha = float(row[alpha_field])
                score = float(row[metric])
            except (KeyError, TypeError, ValueError) as exc:
                raise EvaluationControllerError(f"invalid curve row {index}") from exc
            if not math.isfinite(alpha) or not math.isfinite(score):
                raise EvaluationControllerError(f"non-finite curve row {index}")
            if alpha in by_alpha:
                raise EvaluationControllerError(f"duplicate alpha row: {alpha}")
            by_alpha[alpha] = row
        if set(by_alpha) != set(expected):
            raise EvaluationControllerError(
                f"alpha curve coverage mismatch: expected={expected} got={sorted(by_alpha)}"
            )
        constraints = config["constraints"]
        if not isinstance(constraints, list):
            raise EvaluationControllerError("constraints must be a list")
        eligible: list[tuple[float, float]] = []
        trace: list[dict[str, Any]] = []
        for alpha in expected:
            row = by_alpha[alpha]
            reasons: list[str] = []
            for constraint in constraints:
                item = _mapping(constraint, field="constraints[]")
                if set(item) != {"metric", "operator", "value"}:
                    raise EvaluationControllerError(
                        "constraint must contain metric, operator, and value"
                    )
                name = _string(item["metric"], field="constraint.metric")
                operator = item["operator"]
                try:
                    actual = float(row[name])
                    threshold = float(item["value"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise EvaluationControllerError(
                        f"invalid constraint metric {name!r}"
                    ) from exc
                comparisons = {
                    ">=": actual >= threshold,
                    "<=": actual <= threshold,
                    ">": actual > threshold,
                    "<": actual < threshold,
                    "==": actual == threshold,
                }
                if operator not in comparisons:
                    raise EvaluationControllerError(
                        f"unsupported constraint operator: {operator!r}"
                    )
                if not comparisons[operator]:
                    reasons.append(f"{name}{operator}{threshold}")
            score = float(row[metric])
            if not reasons:
                eligible.append((alpha, score))
            trace.append(
                {
                    "alpha": alpha,
                    "primary_metric": score,
                    "eligible": not reasons,
                    "failed_constraints": reasons,
                }
            )
        if not eligible:
            raise EvaluationControllerError("no alpha satisfies registered constraints")
        best = (
            max(score for _alpha, score in eligible)
            if direction == "maximize"
            else min(score for _alpha, score in eligible)
        )
        near_best = [
            alpha
            for alpha, score in eligible
            if (best - score <= tolerance if direction == "maximize" else score - best <= tolerance)
        ]
        selected = min(near_best) if tie_break == "smallest_alpha" else max(near_best)
        return {
            "selected_alpha": selected,
            "best_primary_metric": best,
            "near_best_alphas": sorted(near_best),
            "selection_trace": trace,
        }


def run_registered_evaluator(
    *,
    evaluator_binding: Mapping[str, Any],
    predictions: Path,
    references: Path | None,
    test_authorization_path: Path | None = None,
    evaluation_config: Mapping[str, Any],
    evidence_root: Path,
    output_dir: Path,
    manifest_path: Path,
    registry: ComponentRegistry = COMPONENT_REGISTRY,
) -> dict[str, Any]:
    if (references is None) == (test_authorization_path is None):
        raise EvaluationControllerError(
            "evaluation requires exactly one references artifact or test authorization"
        )
    authorization_artifact = None
    if test_authorization_path is not None:
        authorization = _load_object(
            test_authorization_path, field="test authorization"
        )
        if (
            authorization.get("manifest_type") != "test_authorization"
            or authorization.get("status") != "authorized"
        ):
            raise EvaluationControllerError("test authorization is not active")
        references = _resolve_artifact(
            _mapping(authorization.get("test_role"), field="test_authorization.test_role"),
            evidence_root,
            field="authorized test role",
        )
        authorization_artifact = _artifact(
            test_authorization_path, evidence_root, field="test authorization"
        )
    assert references is not None
    registration = registry.resolve(
        evaluator_binding,
        expected_kind="evaluator",
        required_operations=("evaluate",),
    )
    result = registration.implementation.evaluate(
        EvaluationRequest(
            predictions=predictions.resolve(),
            references=references.resolve(),
            evaluation_config=copy.deepcopy(evaluation_config),
            evidence_root=evidence_root.resolve(),
            output_dir=output_dir,
        )
    )
    if not isinstance(result, EvaluationResult):
        raise EvaluationControllerError("evaluator must return EvaluationResult")
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "evaluator": {
            "id": registration.component_id,
            "version": registration.version,
            "input_contract": registration.input_contract,
            "output_contract": registration.output_contract,
        },
        "predictions": _artifact(predictions, evidence_root, field="predictions"),
        "references": _artifact(references, evidence_root, field="references"),
        "test_authorization": authorization_artifact,
        "evaluation_config": copy.deepcopy(dict(evaluation_config)),
        "per_sample_scores": _artifact(
            result.per_sample_scores, evidence_root, field="per_sample_scores"
        ),
        "summary_metrics": _artifact(
            result.summary_metrics, evidence_root, field="summary_metrics"
        ),
    }
    evaluator_artifacts = [
        _artifact(path, evidence_root, field="evaluator artifact")
        for path in result.auxiliary_artifacts
    ]
    if evaluator_artifacts:
        manifest["evaluator_artifacts"] = evaluator_artifacts
    manifest["artifact_closure"] = [
        manifest["predictions"],
        manifest["references"],
        manifest["per_sample_scores"],
        manifest["summary_metrics"],
    ] + ([authorization_artifact] if authorization_artifact is not None else [])
    manifest["artifact_closure"].extend(evaluator_artifacts)
    _write_exclusive(manifest_path, manifest)
    return manifest


def freeze_alpha(
    *,
    curve_path: Path,
    controller_manifest_path: Path,
    calibration_role_manifest_path: Path,
    evaluator_manifest_path: Path,
    selector_binding: Mapping[str, Any],
    selection_config: Mapping[str, Any],
    evidence_root: Path,
    output_path: Path,
    registry: ComponentRegistry = COMPONENT_REGISTRY,
) -> dict[str, Any]:
    from .builtin_components import register_builtin_components

    register_builtin_components(registry)
    registration = registry.resolve(
        selector_binding,
        expected_kind="alpha_selector",
        required_operations=("select",),
    )
    rows = _load_rows(curve_path)
    selected = registration.implementation.select(
        AlphaSelectionRequest(
            curve_rows=tuple(copy.deepcopy(rows)),
            selection_config=copy.deepcopy(selection_config),
        )
    )
    if not isinstance(selected, Mapping) or "selected_alpha" not in selected:
        raise EvaluationControllerError("alpha selector returned an invalid decision")
    calibration_role = _load_object(
        calibration_role_manifest_path, field="calibration role manifest"
    )
    _validate_data_role(calibration_role, purpose="calibration")
    _validate_controller(
        _load_object(controller_manifest_path, field="controller manifest")
    )
    evaluator_manifest = _load_object(
        evaluator_manifest_path, field="evaluator manifest"
    )
    if evaluator_manifest.get("status") != "complete":
        raise EvaluationControllerError("evaluator manifest is incomplete")
    manifest = {
        "schema_version": 1,
        "manifest_type": "alpha_freeze",
        "status": "frozen",
        "selector": {
            "id": registration.component_id,
            "version": registration.version,
        },
        "selection_config": copy.deepcopy(selection_config),
        "decision": copy.deepcopy(dict(selected)),
        "curve": _artifact(curve_path, evidence_root, field="curve"),
        "controller": _artifact(
            controller_manifest_path, evidence_root, field="controller_manifest"
        ),
        "calibration_role": _artifact(
            calibration_role_manifest_path,
            evidence_root,
            field="calibration_role_manifest",
        ),
        "evaluator_manifest": _artifact(
            evaluator_manifest_path, evidence_root, field="evaluator_manifest"
        ),
    }
    manifest["artifact_closure"] = [
        manifest["curve"],
        manifest["controller"],
        manifest["calibration_role"],
        manifest["evaluator_manifest"],
    ]
    _write_exclusive(output_path, manifest)
    return manifest


def authorize_final_test(
    *,
    alpha_freeze_path: Path | None,
    controller_manifest_path: Path | None,
    test_role_manifest_path: Path,
    test_policy: Mapping[str, Any] | None = None,
    evidence_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    if (alpha_freeze_path is None) == (test_policy is None):
        raise EvaluationControllerError(
            "test authorization requires exactly one of alpha_freeze or test_policy"
        )
    test_role = _load_object(test_role_manifest_path, field="test role manifest")
    _validate_data_role(test_role, purpose="final_test")
    controller_artifact = None
    if controller_manifest_path is not None:
        controller_artifact = _artifact(
            controller_manifest_path, evidence_root, field="controller_manifest"
        )
        _validate_controller(
            _load_object(controller_manifest_path, field="controller manifest")
        )
    authorization: dict[str, Any] = {
        "schema_version": 1,
        "manifest_type": "test_authorization",
        "status": "authorized",
        "test_role": _artifact(
            test_role_manifest_path, evidence_root, field="test_role_manifest"
        ),
    }
    closure = [authorization["test_role"]]
    if controller_artifact is not None:
        authorization["controller"] = controller_artifact
        closure.insert(0, controller_artifact)
    if alpha_freeze_path is not None:
        if controller_artifact is None:
            raise EvaluationControllerError(
                "selected-alpha authorization requires a controller"
            )
        alpha = _load_object(alpha_freeze_path, field="alpha freeze")
        if (
            alpha.get("manifest_type") != "alpha_freeze"
            or alpha.get("status") != "frozen"
        ):
            raise EvaluationControllerError("alpha freeze is incomplete")
        if controller_artifact["relative_path"] != alpha.get("controller", {}).get("relative_path"):
            raise EvaluationControllerError("alpha freeze belongs to another controller")
        authorization["mode"] = "selected_alpha"
        authorization["alpha_freeze"] = _artifact(
            alpha_freeze_path, evidence_root, field="alpha_freeze"
        )
        closure.append(authorization["alpha_freeze"])
    else:
        policy = dict(_mapping(test_policy, field="test_policy"))
        if policy == {"mode": "direct_policy"}:
            if controller_artifact is not None:
                raise EvaluationControllerError(
                    "direct-policy authorization cannot include a controller"
                )
            authorization["mode"] = "direct_policy"
            authorization["generation_policy"] = {"mode": "direct_policy"}
        elif set(policy) == {"mode", "alpha_grid"} and policy.get("mode") == "report_full_curve":
            if controller_artifact is None:
                raise EvaluationControllerError(
                    "full-curve authorization requires a controller"
                )
            grid = policy.get("alpha_grid")
            if not isinstance(grid, list) or not grid:
                raise EvaluationControllerError("full-curve alpha_grid must be non-empty")
            normalized: list[float] = []
            for value in grid:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise EvaluationControllerError("full-curve alpha_grid must be numeric")
                number = float(value)
                if not math.isfinite(number):
                    raise EvaluationControllerError("full-curve alpha_grid must be finite")
                normalized.append(number)
            if len(normalized) != len(set(normalized)):
                raise EvaluationControllerError("full-curve alpha_grid contains duplicates")
            authorization["mode"] = "full_curve"
            authorization["generation_policy"] = {
                "mode": "report_full_curve",
                "alpha_grid": normalized,
            }
        else:
            raise EvaluationControllerError(
                "test_policy must be direct_policy or report_full_curve with alpha_grid"
            )
    authorization["artifact_closure"] = closure
    _write_exclusive(output_path, authorization)
    return authorization
