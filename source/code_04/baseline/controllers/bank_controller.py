"""Build deterministic bank-training and inference-composition plans."""

from __future__ import annotations

import copy
from typing import Any, Mapping


GROUPED_MODES = frozenset({"signed_banks", "task_banks", "extrema_banks"})
SINGLE_MODES = frozenset({"unsigned", "joint", "native_tuple"})
MERGE_OPERATORS = frozenset({"identity", "sum", "method_native"})
ALPHA_POLICIES = frozenset({"shared", "method_native"})


class BankContractError(ValueError):
    """Raised when positions cannot form the registered training banks."""


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BankContractError(f"{field} must be an object")
    return value


def _string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BankContractError(f"{field} must be a non-empty string")
    return value


def build_bank_plan(
    position_plan: Mapping[str, Any],
    method_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Map a resolved position plan into independently trainable banks."""

    composition = dict(_mapping(position_plan.get("composition"), field="composition"))
    mode = _string(composition.get("mode"), field="composition.mode")
    if mode not in GROUPED_MODES | SINGLE_MODES:
        raise BankContractError(f"unsupported composition mode: {mode}")
    merge = dict(_mapping(composition.get("merge"), field="composition.merge"))
    operator = _string(merge.get("operator"), field="composition.merge.operator")
    alpha_policy = _string(
        merge.get("alpha_policy"), field="composition.merge.alpha_policy"
    )
    retrain = merge.get("retrain_after_merge")
    if operator not in MERGE_OPERATORS:
        raise BankContractError(f"unsupported merge operator: {operator}")
    if alpha_policy not in ALPHA_POLICIES:
        raise BankContractError(f"unsupported alpha policy: {alpha_policy}")
    if not isinstance(retrain, bool):
        raise BankContractError("composition.merge.retrain_after_merge must be boolean")
    if mode in GROUPED_MODES and operator == "identity":
        raise BankContractError(f"{mode} cannot use merge operator identity")
    if mode in SINGLE_MODES and operator == "sum":
        raise BankContractError(f"{mode} cannot sum multiple banks")

    raw_positions = position_plan.get("positions")
    if not isinstance(raw_positions, list) or not raw_positions:
        raise BankContractError("position plan must contain positions")
    positions = [dict(_mapping(row, field="positions[]")) for row in raw_positions]
    banks: list[dict[str, Any]] = []

    if mode in GROUPED_MODES:
        raw_order = composition.get("bank_order")
        if not isinstance(raw_order, list) or not raw_order:
            raise BankContractError(f"{mode} requires an explicit bank_order")
        bank_order = [_string(item, field="composition.bank_order[]") for item in raw_order]
        if len(bank_order) != len(set(bank_order)):
            raise BankContractError("composition.bank_order contains duplicates")
        quotas = dict(_mapping(composition.get("quotas"), field="composition.quotas"))
        if set(bank_order) != set(quotas):
            raise BankContractError("bank_order must contain every quota group exactly once")
        for bank_id in bank_order:
            members = [row for row in positions if row.get("group") == bank_id]
            if len(members) != quotas[bank_id]:
                raise BankContractError(
                    f"bank {bank_id!r} expected {quotas[bank_id]} positions, got {len(members)}"
                )
            banks.append(
                {
                    "bank_id": bank_id,
                    "training_mode": "joint_within_bank",
                    "positions": copy.deepcopy(members),
                }
            )
    else:
        if any("group" in row for row in positions):
            raise BankContractError(f"{mode} positions must not carry bank groups")
        banks.append(
            {
                "bank_id": "all",
                "training_mode": "joint_within_bank",
                "positions": copy.deepcopy(positions),
            }
        )

    plan = {
        "schema_version": 1,
        "method": _string(method_plan.get("method"), field="method_plan.method"),
        "component_type": _string(
            position_plan.get("component_type"), field="position_plan.component_type"
        ),
        "composition": {
            "mode": mode,
            "bank_order": [bank["bank_id"] for bank in banks],
            "merge": {
                "operator": operator,
                "alpha_policy": alpha_policy,
                "retrain_after_merge": retrain,
            },
        },
        "banks": banks,
    }
    return plan


def build_bank_plan_from_position_freeze(
    position_freeze: Mapping[str, Any],
    method_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Build an intervention-bank plan only from an immutable position cell."""

    from .position_freeze_controller import validate_position_freeze

    frozen = validate_position_freeze(position_freeze)
    if method_plan.get("method_kind") != "intervention_baseline":
        raise BankContractError(
            "bank execution requires one intervention-baseline method"
        )
    plan = build_bank_plan(
        _mapping(frozen.get("position_plan"), field="position_freeze.position_plan"),
        method_plan,
    )
    plan["position_freeze"] = {
        "dataset": frozen.get("dataset"),
        "selector": copy.deepcopy(frozen.get("selector")),
        "data_freeze": copy.deepcopy(frozen.get("data_freeze")),
    }
    return plan
