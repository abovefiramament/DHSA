"""Freeze post-training component audits without owning method execution.

The method backend exposes one immutable payload view per trained component.
This controller binds those views to an audit split, validates one decision per
candidate, and emits the final position/bank plans. It never trains, scores, or
chooses a scientific threshold.
"""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .bank_controller import build_bank_plan


FINALIZATION_MODES = frozenset(
    {"reuse_component_payloads", "retrain_selected_banks"}
)


class PostTrainingAuditError(ValueError):
    """Raised when a post-training audit breaks its registered boundary."""


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PostTrainingAuditError(f"{field} must be an object")
    return value


def _string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PostTrainingAuditError(f"{field} must be a non-empty string")
    return value


def _positive_integer(value: Any, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise PostTrainingAuditError(f"{field} must be a positive integer")
    return value


def _artifact_ref(value: Any, *, field: str) -> dict[str, Any]:
    item = dict(_mapping(value, field=field))
    if "artifact" in item:
        if set(item) != {"artifact"}:
            raise PostTrainingAuditError(f"{field} symbolic reference is ambiguous")
        artifact = _string(item["artifact"], field=f"{field}.artifact")
        if not artifact.startswith("artifact://"):
            raise PostTrainingAuditError(f"{field}.artifact must use artifact://")
        return {"artifact": artifact}
    reference: dict[str, Any] = {}
    if "relative_path" in item:
        relative = Path(_string(item["relative_path"], field=f"{field}.relative_path"))
        if relative.is_absolute() or ".." in relative.parts:
            raise PostTrainingAuditError(f"{field}.relative_path escapes evidence root")
        reference["relative_path"] = relative.as_posix()
    elif "path" in item:
        path = Path(_string(item["path"], field=f"{field}.path"))
        reference["name"] = path.name
    else:
        raise PostTrainingAuditError(f"{field} requires path or relative_path")
    if "size_bytes" in item:
        reference["size_bytes"] = item["size_bytes"]
    return reference


def build_post_training_audit_plan(
    bank_plan: Mapping[str, Any],
    component_payloads: Sequence[Mapping[str, Any]],
    audit_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind every candidate position to its trained single-component payload."""

    if audit_config.get("phase") != "post_bank_training":
        raise PostTrainingAuditError("audit.phase must be post_bank_training")
    if audit_config.get("unit") != "component_payload":
        raise PostTrainingAuditError("audit.unit must be component_payload")
    expected_candidates = _positive_integer(
        audit_config.get("expected_candidate_count"),
        field="audit.expected_candidate_count",
    )
    expected_final = _positive_integer(
        audit_config.get("expected_final_count"), field="audit.expected_final_count"
    )
    if expected_final > expected_candidates:
        raise PostTrainingAuditError("final count cannot exceed candidate count")
    decision = dict(_mapping(audit_config.get("decision"), field="audit.decision"))
    decision_field = _string(
        decision.get("decision_field"), field="audit.decision.decision_field"
    )
    keep_values = decision.get("keep_values")
    if not isinstance(keep_values, list) or not keep_values:
        raise PostTrainingAuditError("audit.decision.keep_values must be non-empty")
    if any(keep_values[:index].count(value) for index, value in enumerate(keep_values)):
        raise PostTrainingAuditError("audit.decision.keep_values contains duplicates")
    require_complete = decision.get("require_complete_coverage")
    if not isinstance(require_complete, bool):
        raise PostTrainingAuditError(
            "audit.decision.require_complete_coverage must be boolean"
        )
    finalization = dict(
        _mapping(audit_config.get("finalization"), field="audit.finalization")
    )
    finalization_mode = _string(
        finalization.get("mode"), field="audit.finalization.mode"
    )
    if finalization_mode not in FINALIZATION_MODES:
        raise PostTrainingAuditError(
            f"unsupported post-training finalization mode: {finalization_mode}"
        )

    banks = bank_plan.get("banks")
    if not isinstance(banks, list) or not banks:
        raise PostTrainingAuditError("bank_plan.banks must be non-empty")
    candidates: list[dict[str, Any]] = []
    expected: dict[str, str] = {}
    for bank in banks:
        bank_map = _mapping(bank, field="bank_plan.banks[]")
        bank_id = _string(bank_map.get("bank_id"), field="bank.bank_id")
        positions = bank_map.get("positions")
        if not isinstance(positions, list) or not positions:
            raise PostTrainingAuditError(f"bank {bank_id!r} has no positions")
        for position in positions:
            row = _mapping(position, field=f"bank[{bank_id}].positions[]")
            component_id = _string(row.get("component_id"), field="component_id")
            if component_id in expected:
                raise PostTrainingAuditError(f"duplicate candidate: {component_id}")
            expected[component_id] = bank_id

    by_component: dict[str, Mapping[str, Any]] = {}
    for raw in component_payloads:
        item = _mapping(raw, field="component_payloads[]")
        component_id = _string(item.get("component_id"), field="component_id")
        if component_id in by_component:
            raise PostTrainingAuditError(f"duplicate component payload: {component_id}")
        by_component[component_id] = item
    if set(by_component) != set(expected):
        missing = sorted(set(expected) - set(by_component))
        extra = sorted(set(by_component) - set(expected))
        raise PostTrainingAuditError(
            f"component payload coverage mismatch: missing={missing}, extra={extra}"
        )
    if len(expected) != expected_candidates:
        raise PostTrainingAuditError(
            f"candidate count mismatch: expected {expected_candidates}, got {len(expected)}"
        )

    for component_id, bank_id in expected.items():
        payload = by_component[component_id]
        if payload.get("bank_id") != bank_id:
            raise PostTrainingAuditError(
                f"component {component_id} payload belongs to the wrong bank"
            )
        candidates.append(
            {
                "component_id": component_id,
                "bank_id": bank_id,
                "component_payload": _artifact_ref(
                    payload.get("component_payload"),
                    field=f"component_payload[{component_id}]",
                ),
                "source_pair_id": payload.get("source_pair_id"),
            }
        )

    plan = {
        "schema_version": 1,
        "phase": "post_bank_training",
        "unit": "component_payload",
        "audit_data_manifest": _artifact_ref(
            audit_config.get("audit_data_manifest"),
            field="audit.audit_data_manifest",
        ),
        "candidates": candidates,
        "expected_candidate_count": expected_candidates,
        "expected_final_count": expected_final,
        "decision": {
            "decision_field": decision_field,
            "keep_values": copy.deepcopy(keep_values),
            "require_complete_coverage": require_complete,
        },
        "finalization": {
            "mode": finalization_mode,
            "retrain_selected_banks": finalization_mode
            == "retrain_selected_banks",
        },
    }
    return plan


def _load_decisions(path: Path) -> list[Mapping[str, Any]]:
    if not path.is_file():
        raise PostTrainingAuditError(f"decision manifest does not exist: {path}")
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            rows: Any = list(csv.DictReader(handle))
    elif path.suffix.lower() == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(rows, Mapping):
            rows = rows.get("decisions")
    else:
        raise PostTrainingAuditError("decision manifest must be CSV or JSON")
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise PostTrainingAuditError("decision manifest must contain object rows")
    return rows


def freeze_post_training_selection(
    audit_plan: Mapping[str, Any],
    source_bank_plan: Mapping[str, Any],
    *,
    decision_manifest_path: Path,
) -> dict[str, Any]:
    """Freeze post-training decisions and emit final position and bank plans."""

    if audit_plan.get("phase") != "post_bank_training":
        raise PostTrainingAuditError("post-training audit plan phase is invalid")
    source_ids = {
        row["component_id"]
        for bank in source_bank_plan.get("banks", [])
        for row in bank.get("positions", [])
    }
    plan_ids = {row["component_id"] for row in audit_plan.get("candidates", [])}
    if source_ids != plan_ids:
        raise PostTrainingAuditError("audit plan and source bank candidates disagree")
    rows = _load_decisions(decision_manifest_path)
    decision_spec = _mapping(audit_plan.get("decision"), field="audit_plan.decision")
    decision_field = _string(
        decision_spec.get("decision_field"), field="decision.decision_field"
    )
    keep_values = decision_spec.get("keep_values", [])
    candidate_rows = audit_plan.get("candidates")
    if not isinstance(candidate_rows, list):
        raise PostTrainingAuditError("audit_plan.candidates must be a list")
    candidate_ids = [row["component_id"] for row in candidate_rows]
    decisions: dict[str, Any] = {}
    for index, raw in enumerate(rows):
        row = _mapping(raw, field=f"decisions[{index}]")
        component_id = _string(row.get("component_id"), field="decision.component_id")
        if component_id in decisions:
            raise PostTrainingAuditError(f"duplicate audit decision: {component_id}")
        if component_id not in candidate_ids:
            raise PostTrainingAuditError(
                f"post-training audit cannot add candidate: {component_id}"
            )
        if decision_field not in row:
            raise PostTrainingAuditError(
                f"decision {component_id} lacks field {decision_field!r}"
            )
        decisions[component_id] = row[decision_field]
    require_complete = decision_spec.get("require_complete_coverage")
    if require_complete and set(decisions) != set(candidate_ids):
        missing = sorted(set(candidate_ids) - set(decisions))
        raise PostTrainingAuditError(f"decision manifest misses candidates: {missing}")
    retained = [
        component_id
        for component_id in candidate_ids
        if component_id in decisions
        and decisions[component_id] in keep_values
    ]
    expected_final = audit_plan.get("expected_final_count")
    if len(retained) != expected_final:
        raise PostTrainingAuditError(
            f"post-training retained count mismatch: expected {expected_final}, got {len(retained)}"
        )

    original_positions = {
        row["component_id"]: copy.deepcopy(row)
        for bank in source_bank_plan["banks"]
        for row in bank["positions"]
    }
    positions = [original_positions[component_id] for component_id in retained]
    original_composition = copy.deepcopy(source_bank_plan["composition"])
    bank_order = [
        bank_id
        for bank_id in original_composition["bank_order"]
        if any(row.get("group", "all") == bank_id for row in positions)
    ]
    mode = original_composition["mode"]
    if mode in {"signed_banks", "task_banks", "extrema_banks"}:
        quotas = {
            bank_id: sum(row.get("group") == bank_id for row in positions)
            for bank_id in bank_order
        }
    else:
        bank_order = ["all"]
        quotas = {}
    finalization_mode = audit_plan["finalization"]["mode"]
    merge = copy.deepcopy(original_composition["merge"])
    merge["retrain_after_merge"] = finalization_mode == "retrain_selected_banks"
    position_plan = {
        "component_type": source_bank_plan["component_type"],
        "source_kind": "post_training_audit",
        "composition": {
            "mode": mode,
            "quotas": quotas,
            "bank_order": bank_order,
            "merge": merge,
        },
        "distinct_global": True,
        "selection": None,
        "strategies": [
            {
                "mode": "post_training_audit",
                "input_count": len(candidate_ids),
                "output_count": len(retained),
                "decision_manifest": decision_manifest_path.name,
            }
        ],
        "positions": positions,
    }
    method_plan = {
        "method": source_bank_plan["method"],
    }
    final_bank_plan = build_bank_plan(position_plan, method_plan)
    frozen = {
        "schema_version": 1,
        "status": "frozen",
        "decision_manifest": {"name": decision_manifest_path.name},
        "retained_component_ids": retained,
        "dropped_component_ids": [
            component_id for component_id in candidate_ids if component_id not in retained
        ],
        "finalization": copy.deepcopy(audit_plan["finalization"]),
        "final_position_plan": position_plan,
        "final_bank_plan": final_bank_plan,
    }
    return frozen
