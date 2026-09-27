"""Shared execution stages for native external baselines.

The method backend trains one preregistered candidate.  This controller owns
artifact publication and validation-only candidate selection, so external
methods do not embed dataset scorers or test access.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .component_registry import COMPONENT_REGISTRY, ComponentRegistry
from .contracts import NativeBaselineTrainRequest, NativeBaselineTrainResult


class NativeBaselineControllerError(ValueError):
    pass


def _load(path: Path, *, field: str) -> dict[str, Any]:
    if not path.is_file():
        raise NativeBaselineControllerError(f"{field} does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise NativeBaselineControllerError(f"{field} must contain one object")
    return value


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise NativeBaselineControllerError(f"immutable artifact exists: {path}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(rendered)


def _artifact(path: Path, root: Path, *, field: str) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise NativeBaselineControllerError(f"{field} does not exist: {resolved}")
    try:
        relative = resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise NativeBaselineControllerError(f"{field} escapes evidence root") from exc
    return {"relative_path": relative.as_posix(), "size_bytes": resolved.stat().st_size}


def train_registered_candidate(
    *,
    backend_binding: Mapping[str, Any],
    candidate_id: str,
    candidate_config: Mapping[str, Any],
    method_plan: Mapping[str, Any],
    model_config: Mapping[str, Any],
    execution_config: Mapping[str, Any],
    training_data_manifest: Path,
    validation_data_manifest: Path | None,
    evidence_root: Path,
    output_dir: Path,
    controller_manifest_path: Path,
    position_freeze_path: Path | None = None,
    registry: ComponentRegistry = COMPONENT_REGISTRY,
) -> dict[str, Any]:
    registration = registry.resolve(
        backend_binding,
        expected_kind="native_baseline_backend",
        required_operations=("train_candidate",),
    )
    training_role = _load(training_data_manifest, field="training role")
    if training_role.get("status") != "frozen" or training_role.get("purpose") != "training":
        raise NativeBaselineControllerError("native training requires the frozen training role")
    if validation_data_manifest is not None:
        validation_role = _load(validation_data_manifest, field="validation role")
        if validation_role.get("status") != "frozen" or validation_role.get("purpose") != "training_validation":
            raise NativeBaselineControllerError(
                "native validation requires the frozen training-validation role"
            )
    resolved_candidate = copy.deepcopy(dict(candidate_config))
    if position_freeze_path is not None:
        from baseline.controllers.position_freeze_controller import select_frozen_candidate

        resolved_candidate["position"] = select_frozen_candidate(
            _load(position_freeze_path, field="frozen candidate positions"),
            rank=resolved_candidate.get("candidate_rank"),
        )
    result = registration.implementation.train_candidate(
        NativeBaselineTrainRequest(
            candidate_id=candidate_id,
            candidate_config=resolved_candidate,
            method_plan=copy.deepcopy(method_plan),
            model_config=copy.deepcopy(model_config),
            execution_config=copy.deepcopy(execution_config),
            training_data_manifest=training_data_manifest.resolve(),
            validation_data_manifest=(
                validation_data_manifest.resolve()
                if validation_data_manifest is not None
                else None
            ),
            evidence_root=evidence_root.resolve(),
            output_dir=output_dir.resolve(),
        )
    )
    if not isinstance(result, NativeBaselineTrainResult):
        raise NativeBaselineControllerError(
            "native baseline backend must return NativeBaselineTrainResult"
        )
    manifest = {
        "schema_version": 1,
        "manifest_type": "controller_freeze",
        "status": "frozen",
        "method": method_plan.get("method"),
        "candidate_id": candidate_id,
        "candidate_config": resolved_candidate,
        "backend": {"id": registration.component_id, "version": registration.version},
        "model_config": copy.deepcopy(dict(model_config)),
        "trainable_parameters": int(result.trainable_parameters),
        "payload_manifest": _artifact(
            result.payload_manifest, evidence_root, field="payload manifest"
        ),
        "training_manifest": _artifact(
            result.training_manifest, evidence_root, field="training manifest"
        ),
        "training_role": _artifact(
            training_data_manifest, evidence_root, field="training role"
        ),
        "validation_role": (
            _artifact(validation_data_manifest, evidence_root, field="validation role")
            if validation_data_manifest is not None
            else None
        ),
    }
    manifest["artifact_closure"] = [
        manifest["payload_manifest"],
        manifest["training_manifest"],
        manifest["training_role"],
    ] + ([manifest["validation_role"]] if manifest["validation_role"] else [])
    if position_freeze_path is not None:
        manifest["position_freeze"] = _artifact(
            position_freeze_path, evidence_root, field="frozen positions"
        )
        manifest["artifact_closure"].append(manifest["position_freeze"])
    _write_exclusive(controller_manifest_path, manifest)
    return manifest


def _metric(path: Path, name: str) -> float:
    value = _load(path, field="candidate summary metrics")
    metrics = value.get("metrics")
    if not isinstance(metrics, Mapping):
        raise NativeBaselineControllerError("candidate summary must contain metrics")
    score = metrics.get(name)
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise NativeBaselineControllerError(f"candidate metric {name!r} is missing")
    return float(score)


def freeze_best_candidate(
    *,
    candidates: Mapping[str, Mapping[str, Path]],
    selection_config: Mapping[str, Any],
    evidence_root: Path,
    controller_manifest_path: Path,
    selection_manifest_path: Path,
) -> dict[str, Any]:
    order = selection_config.get("candidate_order")
    metric = selection_config.get("primary_metric")
    direction = selection_config.get("direction")
    if (
        not isinstance(order, list)
        or not order
        or not all(isinstance(item, str) and item for item in order)
        or not isinstance(metric, str)
        or direction not in {"maximize", "minimize"}
        or set(order) != set(candidates)
    ):
        raise NativeBaselineControllerError("candidate selection config is invalid")
    trace = []
    for candidate_id in order:
        paths = candidates[candidate_id]
        controller = paths.get("controller")
        summary = paths.get("summary_metrics")
        if not isinstance(controller, Path) or not isinstance(summary, Path):
            raise NativeBaselineControllerError("candidate artifacts are incomplete")
        frozen = _load(controller, field=f"{candidate_id} controller")
        if frozen.get("manifest_type") != "controller_freeze" or frozen.get("status") != "frozen":
            raise NativeBaselineControllerError("candidate controller is not frozen")
        trace.append({"candidate_id": candidate_id, "score": _metric(summary, metric)})
    scores = [row["score"] for row in trace]
    best = max(scores) if direction == "maximize" else min(scores)
    selected = next(row["candidate_id"] for row in trace if row["score"] == best)
    source_path = candidates[selected]["controller"]
    source = _load(source_path, field="selected controller")
    source["candidate_selection"] = {
        "primary_metric": metric,
        "direction": direction,
        "tie_break": "first_in_preregistered_candidate_order",
        "selected_candidate": selected,
        "trace": trace,
    }
    source["selected_from"] = _artifact(
        source_path, evidence_root, field="selected source controller"
    )
    source.setdefault("artifact_closure", []).append(source["selected_from"])
    _write_exclusive(controller_manifest_path, source)
    selection = {
        "schema_version": 1,
        "manifest_type": "native_candidate_selection",
        "status": "frozen",
        "selection_config": copy.deepcopy(dict(selection_config)),
        "decision": {"selected_candidate": selected, "best_metric": best},
        "trace": trace,
        "controller": _artifact(
            controller_manifest_path, evidence_root, field="selected controller"
        ),
    }
    _write_exclusive(selection_manifest_path, selection)
    return selection
