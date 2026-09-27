"""Generate a blinded single-component audit with registered backends.

This controller only composes an already trained component payload, generates
through the registered backend, and scores through the registered evaluator.
It does not implement CAST, decoding, or a task score.
"""

from __future__ import annotations

import copy
import csv
import json
import os
import random
from pathlib import Path
from typing import Any, Mapping

from baseline.controllers.payload_controller import (
    compose_controller_manifest,
    register_position_vector_pair,
)
from baseline.implementations.loaders import load_rows
from experiments.shared.component_registry import COMPONENT_REGISTRY, ComponentRegistry
from experiments.shared.contracts import BankComposeRequest, BankTrainResult
from experiments.shared.evaluation_controller import run_registered_evaluator
from experiments.shared.generation_controller import run_registered_generator


class PostTrainingAuditExecutionError(ValueError):
    """Raised when a blinded post-training audit is malformed."""


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PostTrainingAuditExecutionError(f"{field} must be an object")
    return value


def _string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PostTrainingAuditExecutionError(f"{field} must be a non-empty string")
    return value


def _load(path: Path, *, field: str) -> dict[str, Any]:
    if not path.is_file():
        raise PostTrainingAuditExecutionError(f"{field} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PostTrainingAuditExecutionError(f"cannot load {field}") from exc
    if not isinstance(value, dict):
        raise PostTrainingAuditExecutionError(f"{field} must contain one object")
    return value


def _load_complete(path: Path, *, field: str) -> dict[str, Any]:
    value = _load(path, field=field)
    if value.get("status") != "complete":
        raise PostTrainingAuditExecutionError(f"{field} is not complete")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise PostTrainingAuditExecutionError(
            f"immutable audit artifact already exists: {path}"
        ) from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(dict(value), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _artifact(path: Path, root: Path, *, field: str) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise PostTrainingAuditExecutionError(
            f"{field} must remain under evidence_root"
        ) from exc
    if not resolved.is_file():
        raise PostTrainingAuditExecutionError(f"{field} does not exist: {resolved}")
    return {"relative_path": relative.as_posix(), "size_bytes": resolved.stat().st_size}


def _candidate_rows(audit_plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates = audit_plan.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise PostTrainingAuditExecutionError("audit plan has no candidates")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in candidates:
        row = dict(_mapping(raw, field="audit_plan.candidates[]"))
        component_id = _string(row.get("component_id"), field="candidate.component_id")
        if component_id in seen:
            raise PostTrainingAuditExecutionError("audit plan repeats a component")
        seen.add(component_id)
        row["component_id"] = component_id
        row["bank_id"] = _string(row.get("bank_id"), field="candidate.bank_id")
        rows.append(row)
    if audit_plan.get("expected_candidate_count") != len(rows):
        raise PostTrainingAuditExecutionError("audit plan candidate count drift")
    return rows


def _blind(candidates: list[dict[str, Any]], *, seed: int) -> list[tuple[str, dict[str, Any]]]:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise PostTrainingAuditExecutionError("audit blinding seed must be an integer")
    rows = list(candidates)
    random.Random(seed).shuffle(rows)
    return [(f"head_{index:02d}", row) for index, row in enumerate(rows, start=1)]


def _single_bank_plan(
    source_bank_plan: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    bank_id = _string(candidate.get("bank_id"), field="candidate.bank_id")
    component_id = _string(candidate.get("component_id"), field="candidate.component_id")
    matches = [
        _mapping(bank, field="source_bank_plan.banks[]")
        for bank in source_bank_plan.get("banks", [])
        if isinstance(bank, Mapping) and bank.get("bank_id") == bank_id
    ]
    if len(matches) != 1:
        raise PostTrainingAuditExecutionError("candidate source bank is ambiguous")
    positions = [
        copy.deepcopy(_mapping(position, field="source_bank_plan.positions[]"))
        for position in matches[0].get("positions", [])
        if isinstance(position, Mapping)
        and position.get("component_id") == component_id
    ]
    if len(positions) != 1:
        raise PostTrainingAuditExecutionError("candidate is absent from its source bank")
    return {
        "schema_version": 1,
        "method": _string(source_bank_plan.get("method"), field="source_bank_plan.method"),
        "component_type": _string(
            source_bank_plan.get("component_type"),
            field="source_bank_plan.component_type",
        ),
        "composition": {
            "mode": "single_component_audit",
            "bank_order": [bank_id],
            "merge": {
                "operator": "identity",
                "alpha_policy": "shared",
                "retrain_after_merge": False,
            },
        },
        "banks": [
            {
                "bank_id": bank_id,
                "training_mode": "joint_payload_slice",
                "positions": positions,
            }
        ],
    }


def _compose_controller(
    *,
    candidate: Mapping[str, Any],
    source_bank_plan: Mapping[str, Any],
    method_plan: Mapping[str, Any],
    backend_binding: Mapping[str, Any],
    execution_config: Mapping[str, Any],
    evidence_root: Path,
    output_dir: Path,
    registry: ComponentRegistry,
) -> Path:
    component_id = _string(candidate.get("component_id"), field="candidate.component_id")
    bank_id = _string(candidate.get("bank_id"), field="candidate.bank_id")
    bank_plan = _single_bank_plan(source_bank_plan, candidate)
    registration = registry.resolve(
        backend_binding,
        expected_kind="bank_backend",
        required_operations=("compose_bank",),
    )
    result = registration.implementation.compose_bank(
        BankComposeRequest(
            bank_id=bank_id,
            ordered_component_ids=(component_id,),
            component_manifests=(copy.deepcopy(dict(candidate)),),
            method_plan=copy.deepcopy(dict(method_plan)),
            execution_config=copy.deepcopy(dict(execution_config)),
            evidence_root=evidence_root.resolve(),
            output_dir=output_dir / "bank",
        )
    )
    if not isinstance(result, BankTrainResult) or result.optimizer_steps != 0:
        raise PostTrainingAuditExecutionError(
            "audit component composition must return a zero-step BankTrainResult"
        )
    pair = register_position_vector_pair(
        bank_plan,
        method_plan,
        bank_id=bank_id,
        payload_path=result.payload_path,
        training_manifest_path=result.training_manifest_path,
        evidence_root=evidence_root,
    )
    controller = compose_controller_manifest((pair,), bank_plan)
    controller["audit_unit"] = {
        "kind": "single_component_payload",
        "component_count": 1,
        "source_pair_id": candidate.get("source_pair_id"),
    }
    path = output_dir / "controller_manifest.json"
    _write(path, controller)
    return path


def _base_controller(path: Path) -> Path:
    _write(
        path,
        {
            "schema_version": 1,
            "manifest_type": "controller_freeze",
            "status": "frozen",
            "composition": {
                "mode": "base_no_external_controller",
                "bank_order": [],
                "merge": {
                    "operator": "identity",
                    "alpha_policy": "shared",
                    "retrain_after_merge": False,
                },
            },
            "ordered_pairs": [],
            "artifact_closure": [],
        },
    )
    return path


def _summary_rows(path: Path, *, audit_id: str) -> list[dict[str, Any]]:
    rows = _load(path, field="evaluation summary").get("rows")
    if not isinstance(rows, list):
        raise PostTrainingAuditExecutionError("audit summary must be grouped by alpha")
    result: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(_mapping(raw, field="evaluation summary row"))
        alpha = row.get("alpha")
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
            raise PostTrainingAuditExecutionError("audit summary alpha must be numeric")
        result.append({"audit_id": audit_id, **row})
    return result


def _write_blinded_samples(
    *,
    candidate_outputs: list[tuple[str, Path]],
    audit_role_manifest: Path,
    evidence_root: Path,
    output_path: Path,
) -> None:
    references = {
        str(row.get("sample_id", index)): row
        for index, row in enumerate(load_rows(audit_role_manifest, root=evidence_root))
    }
    with output_path.open("x", encoding="utf-8") as handle:
        for audit_id, predictions_path in candidate_outputs:
            for prediction in load_rows(predictions_path, root=evidence_root):
                sample_id = str(
                    prediction.get("source_sample_id", prediction.get("sample_id", ""))
                )
                reference = references.get(sample_id)
                if reference is None:
                    raise PostTrainingAuditExecutionError(
                        "audit prediction cannot be aligned to its reference"
                    )
                handle.write(
                    json.dumps(
                        {
                            "audit_id": audit_id,
                            "alpha": prediction.get("alpha"),
                            "sample_id": sample_id,
                            "prompt": reference.get("prompt"),
                            "human_reference": reference.get("reference_summary"),
                            "generated_text": prediction.get("generated_text"),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )


def execute_post_training_audit(
    *,
    audit_plan_path: Path,
    source_bank_plan_path: Path,
    audit_role_manifest_path: Path,
    method_plan: Mapping[str, Any],
    bank_backend_binding: Mapping[str, Any],
    generation_backend_binding: Mapping[str, Any],
    evaluator_binding: Mapping[str, Any],
    model_config: Mapping[str, Any],
    generation_config: Mapping[str, Any],
    evaluation_config: Mapping[str, Any],
    trajectory_config: Mapping[str, Any],
    execution_config: Mapping[str, Any],
    evidence_root: Path,
    output_dir: Path,
    packet_path: Path,
    decision_key_path: Path,
    registry: ComponentRegistry = COMPONENT_REGISTRY,
) -> dict[str, Path]:
    """Run base and 16 single-component audit curves, then package blind evidence."""

    audit_plan = _load(audit_plan_path, field="audit plan")
    source_bank_plan = _load(source_bank_plan_path, field="source bank plan")
    audit = _mapping(method_plan.get("audit"), field="method_plan.audit")
    locked = _mapping(audit.get("locked_audit"), field="method_plan.audit.locked_audit")
    alpha_grid = locked.get("alpha_grid")
    if not isinstance(alpha_grid, list) or not alpha_grid:
        raise PostTrainingAuditExecutionError("locked audit alpha grid is missing")
    if generation_config.get("alpha_grid") != alpha_grid:
        raise PostTrainingAuditExecutionError(
            "audit generation alpha grid differs from the locked audit grid"
        )
    blinded = _blind(_candidate_rows(audit_plan), seed=locked.get("blinding_seed"))
    phase = execution_config.get("audit_execution_phase", "complete")
    if phase not in {"complete", "generate_only", "evaluate_only"}:
        raise PostTrainingAuditExecutionError(f"unsupported audit execution phase: {phase}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if packet_path.is_file() and decision_key_path.is_file():
        return {"audit_packet": packet_path, "decision_key": decision_key_path}

    base_dir = output_dir / "shared_base"
    base_generation_config = dict(generation_config)
    base_generation_config.pop("alpha_grid", None)
    base_generation_config["alpha"] = 0.0
    base_generation_path = base_dir / "generation" / "generation_manifest.json"
    if phase == "evaluate_only":
        required = [base_generation_path] + [
            output_dir / "candidates" / audit_id / "generation/generation_manifest.json"
            for audit_id, _ in blinded
        ]
        for path in required:
            _load_complete(path, field="generation required before CPU audit")
    base_controller_path = base_dir / "controller_manifest.json"
    if not base_controller_path.is_file():
        _base_controller(base_controller_path)
    if base_generation_path.is_file():
        base_generation = _load_complete(
            base_generation_path, field="base generation manifest"
        )
    else:
        base_generation = run_registered_generator(
            backend_binding=generation_backend_binding,
            model_config=model_config,
            generation_config=base_generation_config,
            trajectory_config=trajectory_config,
            trace_kind="audit",
            execution_config=execution_config,
            evidence_root=evidence_root,
            output_dir=base_generation_path.parent,
            manifest_path=base_generation_path,
            controller_manifest_path=base_controller_path,
            data_role_manifest_path=audit_role_manifest_path,
            registry=registry,
        )
    base_evaluation_path = base_dir / "evaluation" / "evaluation_manifest.json"
    if phase == "generate_only":
        pass
    elif base_evaluation_path.is_file():
        _load_complete(base_evaluation_path, field="base evaluation manifest")
    else:
        run_registered_evaluator(
            evaluator_binding=evaluator_binding,
            predictions=evidence_root / base_generation["predictions"]["relative_path"],
            references=audit_role_manifest_path,
            evaluation_config=evaluation_config,
            evidence_root=evidence_root,
            output_dir=base_evaluation_path.parent,
            manifest_path=base_evaluation_path,
            registry=registry,
        )

    metric_rows: list[dict[str, Any]] = []
    blinded_predictions: list[tuple[str, Path]] = []
    packet_candidates: list[dict[str, Any]] = []
    decision_rows: list[dict[str, str]] = []
    for audit_id, candidate in blinded:
        candidate_dir = output_dir / "candidates" / audit_id
        controller = candidate_dir / "controller_manifest.json"
        if not controller.is_file():
            controller = _compose_controller(
                candidate=candidate,
                source_bank_plan=source_bank_plan,
                method_plan=method_plan,
                backend_binding=bank_backend_binding,
                execution_config=execution_config,
                evidence_root=evidence_root,
                output_dir=candidate_dir,
                registry=registry,
            )
        generation_path = candidate_dir / "generation" / "generation_manifest.json"
        if generation_path.is_file():
            generation = _load_complete(
                generation_path, field=f"{audit_id} generation manifest"
            )
        else:
            generation = run_registered_generator(
                backend_binding=generation_backend_binding,
                model_config=model_config,
                generation_config=generation_config,
                trajectory_config=trajectory_config,
                trace_kind="audit",
                execution_config=execution_config,
                evidence_root=evidence_root,
                output_dir=generation_path.parent,
                manifest_path=generation_path,
                controller_manifest_path=controller,
                data_role_manifest_path=audit_role_manifest_path,
                registry=registry,
            )
        if phase == "generate_only":
            continue
        evaluation_path = candidate_dir / "evaluation" / "evaluation_manifest.json"
        if evaluation_path.is_file():
            evaluation = _load_complete(
                evaluation_path, field=f"{audit_id} evaluation manifest"
            )
        else:
            evaluation = run_registered_evaluator(
                evaluator_binding=evaluator_binding,
                predictions=evidence_root / generation["predictions"]["relative_path"],
                references=audit_role_manifest_path,
                evaluation_config=evaluation_config,
                evidence_root=evidence_root,
                output_dir=evaluation_path.parent,
                manifest_path=evaluation_path,
                registry=registry,
            )
        metric_rows.extend(
            _summary_rows(
                evidence_root / evaluation["summary_metrics"]["relative_path"],
                audit_id=audit_id,
            )
        )
        blinded_predictions.append(
            (audit_id, evidence_root / generation["predictions"]["relative_path"])
        )
        packet_candidates.append(
            {
                "audit_id": audit_id,
                "generation_manifest": _artifact(
                    generation_path, evidence_root, field="candidate generation"
                ),
                "evaluation_manifest": _artifact(
                    evaluation_path, evidence_root, field="candidate evaluation"
                ),
            }
        )
        decision_rows.append(
            {
                "audit_id": audit_id,
                "component_id": candidate["component_id"],
                "bank_id": candidate["bank_id"],
            }
        )

    if phase == "generate_only":
        ready = output_dir / "generation_ready.json"
        if not ready.is_file():
            _write(ready, {"status": "complete", "phase": "audit_generation",
                           "candidate_count": len(blinded), "alpha_grid": alpha_grid})
        return {"generation_ready": ready}
    metric_table = output_dir / "blinded_metrics.csv"
    with metric_table.open("x", encoding="utf-8", newline="") as handle:
        fields = sorted({key for row in metric_rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metric_rows)
    samples = output_dir / "blinded_samples.jsonl"
    _write_blinded_samples(
        candidate_outputs=blinded_predictions,
        audit_role_manifest=audit_role_manifest_path,
        evidence_root=evidence_root,
        output_path=samples,
    )
    decision_template = output_dir / "decision_template.csv"
    with decision_template.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("audit_id", "decision", "written_reason")
        )
        writer.writeheader()
        for audit_id, _candidate in blinded:
            writer.writerow({"audit_id": audit_id, "decision": "", "written_reason": ""})
    _write(
        decision_key_path,
        {
            "schema_version": 1,
            "manifest_type": "blinded_audit_decision_key",
            "status": "internal",
            "rows": decision_rows,
        },
    )
    packet = {
        "schema_version": 1,
        "manifest_type": "post_training_audit_packet",
        "status": "ready_for_manual_audit",
        "audit_unit": "single_component_payload",
        "candidate_count": len(blinded),
        "alpha_grid": list(alpha_grid),
        "shared_base_generation": _artifact(
            base_generation_path, evidence_root, field="base generation"
        ),
        "shared_base_evaluation": _artifact(
            base_evaluation_path, evidence_root, field="base evaluation"
        ),
        "blinded_metrics": _artifact(metric_table, evidence_root, field="blinded metrics"),
        "blinded_samples": _artifact(samples, evidence_root, field="blinded samples"),
        "decision_template": _artifact(
            decision_template, evidence_root, field="decision template"
        ),
        "candidates": packet_candidates,
    }
    packet["artifact_closure"] = [
        packet[key]
        for key in (
            "shared_base_generation",
            "shared_base_evaluation",
            "blinded_metrics",
            "blinded_samples",
            "decision_template",
        )
    ] + [
        artifact
        for candidate in packet_candidates
        for artifact in (
            candidate["generation_manifest"],
            candidate["evaluation_manifest"],
        )
    ]
    _write(packet_path, packet)
    return {"audit_packet": packet_path, "decision_key": decision_key_path}


def _decision_rows(path: Path) -> list[Mapping[str, Any]]:
    if not path.is_file():
        raise PostTrainingAuditExecutionError("audit decision source does not exist")
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            rows: Any = list(csv.DictReader(handle))
    elif path.suffix.lower() == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(rows, Mapping):
            rows = rows.get("decisions")
    else:
        raise PostTrainingAuditExecutionError("audit decision source must be CSV or JSON")
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise PostTrainingAuditExecutionError("audit decision source must contain object rows")
    return rows


def resolve_blinded_audit_decisions(
    *,
    audit_plan_path: Path,
    decision_key_path: Path,
    decision_manifest_path: Path,
    output_path: Path,
) -> Path:
    """Map complete human blind-ID decisions back to component IDs."""
    audit_plan = _load(audit_plan_path, field="audit plan")
    key = _load(decision_key_path, field="audit decision key")
    if key.get("manifest_type") != "blinded_audit_decision_key":
        raise PostTrainingAuditExecutionError("audit decision key is invalid")
    mapping: dict[str, str] = {}
    for raw in key.get("rows", []):
        row = _mapping(raw, field="audit decision key row")
        audit_id = _string(row.get("audit_id"), field="audit_id")
        component_id = _string(row.get("component_id"), field="component_id")
        if audit_id in mapping:
            raise PostTrainingAuditExecutionError("audit decision key repeats a blind ID")
        mapping[audit_id] = component_id
    if set(mapping.values()) != {row["component_id"] for row in _candidate_rows(audit_plan)}:
        raise PostTrainingAuditExecutionError("audit decision key does not cover the plan")
    decision_field = _string(
        _mapping(audit_plan.get("decision"), field="audit_plan.decision").get(
            "decision_field"
        ),
        field="audit decision field",
    )
    resolved: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in _decision_rows(decision_manifest_path):
        row = _mapping(raw, field="audit decision row")
        audit_id = _string(row.get("audit_id"), field="decision.audit_id")
        if audit_id in seen or audit_id not in mapping:
            raise PostTrainingAuditExecutionError("audit decision blind IDs are invalid")
        decision = row.get(decision_field)
        if not isinstance(decision, str) or not decision:
            raise PostTrainingAuditExecutionError("audit decision value is missing")
        resolved.append(
            {
                "component_id": mapping[audit_id],
                decision_field: decision,
                "audit_id": audit_id,
                **(
                    {"written_reason": row["written_reason"]}
                    if isinstance(row.get("written_reason"), str)
                    else {}
                ),
            }
        )
        seen.add(audit_id)
    if set(seen) != set(mapping):
        raise PostTrainingAuditExecutionError("audit decisions are incomplete")
    _write(
        output_path,
        {
            "schema_version": 1,
            "manifest_type": "resolved_post_training_audit_decisions",
            "status": "complete",
            "decisions": resolved,
        },
    )
    return output_path
