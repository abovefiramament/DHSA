"""Resolve explicit or manifest-backed intervention positions.

This controller applies only the deterministic ranking/filtering rule declared
by the experiment protocol, then validates the registered bank composition.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .bank_controller import ALPHA_POLICIES, GROUPED_MODES, MERGE_OPERATORS
except ImportError:  # direct file execution
    from bank_controller import ALPHA_POLICIES, GROUPED_MODES, MERGE_OPERATORS


COMPOSITION_MODES = frozenset(
    {"unsigned", "signed_banks", "task_banks", "native_tuple", "joint"}
)
POSITION_METHODS = frozenset({"rcm_zero", "rcm_patch", "iti", "random"})


class PositionContractError(ValueError):
    """Raised when a registered position selection is inconsistent."""


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PositionContractError(f"{field} must be an object")
    return value


def _string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PositionContractError(f"{field} must be a non-empty string")
    return value


def _integer(value: Any, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise PositionContractError(f"{field} must be a non-negative integer")
    return value


def _positive_integer(value: Any, *, field: str) -> int:
    resolved = _integer(value, field=field)
    if resolved == 0:
        raise PositionContractError(f"{field} must be positive")
    return resolved


def _load_manifest(source: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], dict[str, str]]:
    manifest_source = source.get("manifest", source)
    manifest_source = _mapping(
        manifest_source, field="position_control.source.manifest"
    )
    path = Path(
        _string(manifest_source.get("path"), field="position_control.source.path")
    ).expanduser()
    if not path.is_file():
        raise PositionContractError(f"position manifest does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            rows: Any = list(csv.DictReader(handle))
    elif suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif suffix == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(rows, Mapping):
            key = _string(
                manifest_source.get("rows_key"),
                field="position_control.source.rows_key",
            )
            rows = rows.get(key)
    else:
        raise PositionContractError("position manifest must be CSV, JSON, or JSONL")
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise PositionContractError("position manifest rows must be a list of objects")
    selected_field = manifest_source.get("selected_field")
    if selected_field is not None:
        selected_name = _string(selected_field, field="position_control.source.selected_field")
        rows = [row for row in rows if row.get(selected_name) in {True, 1, "1", "true", "True"}]
    return rows, {"path": str(path)}


def _explicit_rows(source: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    values = source.get("positions")
    if not isinstance(values, list) or not values:
        raise PositionContractError("explicit position source requires a non-empty positions list")
    rows: list[Mapping[str, Any]] = []
    for value in values:
        rows.append({"component_id": value} if isinstance(value, str) else _mapping(value, field="position"))
    return rows


def _select_rows(
    rows: Sequence[Mapping[str, Any]],
    control: Mapping[str, Any],
    *,
    group_field: str | None,
) -> tuple[list[Mapping[str, Any]], dict[str, Any] | None]:
    raw = control.get("selection")
    if raw is None:
        return list(rows), None
    selection = dict(_mapping(raw, field="position_control.selection"))
    mode = _string(selection.get("mode"), field="position_control.selection.mode")
    eligibility_field = selection.get("eligibility_field")
    eligible = list(rows)
    if eligibility_field is not None:
        field = _string(eligibility_field, field="selection.eligibility_field")
        eligible = [
            row for row in eligible
            if row.get(field) in {True, 1, "1", "true", "True"}
        ]
    if mode in {"selected_flag", "audited_manifest"}:
        field = _string(selection.get("selected_field"), field="selection.selected_field")
        return (
            [
                row for row in eligible
                if row.get(field) in {True, 1, "1", "true", "True"}
            ],
            selection,
        )
    if mode == "passthrough":
        return eligible, selection
    if mode not in {"top_k", "grouped_top_k"}:
        raise PositionContractError(f"unsupported selection mode: {mode}")
    score_field = _string(selection.get("score_field"), field="selection.score_field")
    order = _string(selection.get("order"), field="selection.order")
    if order not in {"ascending", "descending"}:
        raise PositionContractError("selection.order must be ascending or descending")
    tie_fields = selection.get("tie_break_fields")
    if not isinstance(tie_fields, list) or not tie_fields:
        raise PositionContractError("selection.tie_break_fields must be non-empty")
    tie_fields = [
        _string(field, field="selection.tie_break_fields[]")
        for field in tie_fields
    ]

    def rank_key(row: Mapping[str, Any]):
        try:
            score = float(row[score_field])
        except (KeyError, TypeError, ValueError) as exc:
            raise PositionContractError(f"invalid score field {score_field!r}") from exc
        oriented = -score if order == "descending" else score
        return (oriented, *(str(row.get(field, "")) for field in tie_fields))

    ranked = sorted(eligible, key=rank_key)
    if mode == "top_k":
        count = _positive_integer(selection.get("count"), field="selection.count")
        if len(ranked) < count:
            raise PositionContractError(f"top_k requested {count} from {len(ranked)} rows")
        return ranked[:count], selection
    if group_field is None:
        raise PositionContractError("grouped_top_k requires position_control.group_field")
    quotas = _mapping(selection.get("quotas"), field="selection.quotas")
    if not quotas:
        raise PositionContractError("selection.quotas must not be empty")
    selected: list[Mapping[str, Any]] = []
    for group, raw_count in quotas.items():
        count = _positive_integer(raw_count, field=f"selection.quotas.{group}")
        group_rows = [row for row in ranked if row.get(group_field) == group]
        if len(group_rows) < count:
            raise PositionContractError(f"group {group!r} has fewer than {count} candidates")
        selected.extend(group_rows[:count])
    return selected, selection


def _apply_strategies(
    rows: Sequence[Mapping[str, Any]],
    control: Mapping[str, Any],
    *,
    component_field: str,
    group_field: str | None,
) -> tuple[list[Mapping[str, Any]], list[dict[str, Any]]]:
    raw_strategies = control.get("strategies")
    if raw_strategies is None:
        selected, legacy = _select_rows(rows, control, group_field=group_field)
        return selected, [] if legacy is None else [{"legacy_selection": legacy}]
    if control.get("selection") is not None:
        raise PositionContractError("use either strategies or legacy selection, not both")
    if not isinstance(raw_strategies, list) or not raw_strategies:
        raise PositionContractError("position_control.strategies must be non-empty")
    current = list(rows)
    applied: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_strategies):
        strategy = dict(_mapping(raw, field=f"position_control.strategies[{index}]"))
        mode = _string(strategy.get("mode"), field=f"strategies[{index}].mode")
        before = len(current)
        if mode == "audit_intersection":
            audit_source = _mapping(
                strategy.get("source"), field=f"strategies[{index}].source"
            )
            audit_rows, audit_ref = _load_manifest(audit_source)
            audit_component_field = _string(
                strategy.get("component_field"),
                field=f"strategies[{index}].component_field",
            )
            selected_field = _string(
                strategy.get("selected_field"),
                field=f"strategies[{index}].selected_field",
            )
            require_complete = strategy.get("require_complete_coverage")
            if not isinstance(require_complete, bool):
                raise PositionContractError(
                    f"strategies[{index}].require_complete_coverage must be boolean"
                )
            decisions: dict[str, bool] = {}
            for audit_index, audit_row in enumerate(audit_rows):
                component = _string(
                    audit_row.get(audit_component_field),
                    field=f"strategies[{index}].rows[{audit_index}].{audit_component_field}",
                )
                if component in decisions:
                    raise PositionContractError(
                        f"audit strategy contains duplicate component: {component}"
                    )
                decision = audit_row.get(selected_field)
                if decision not in {True, False, 0, 1, "0", "1", "true", "false", "True", "False"}:
                    raise PositionContractError(
                        f"audit strategy has invalid decision for {component}"
                    )
                decisions[component] = decision in {True, 1, "1", "true", "True"}
            candidates = {
                _string(row.get(component_field), field=f"candidate.{component_field}")
                for row in current
            }
            additions = sorted(set(decisions) - candidates)
            if additions:
                raise PositionContractError(
                    f"audit strategy cannot add candidates: {additions}"
                )
            if require_complete and set(decisions) != candidates:
                missing = sorted(candidates - set(decisions))
                raise PositionContractError(
                    f"audit strategy does not cover every candidate: {missing}"
                )
            current = [
                row
                for row in current
                if decisions.get(str(row.get(component_field)), False)
            ]
            applied.append(
                {
                    "mode": mode,
                    "input_count": before,
                    "output_count": len(current),
                    "require_complete_coverage": require_complete,
                    "source_manifest": audit_ref,
                }
            )
            continue
        if mode not in {"passthrough", "selected_flag", "top_k", "grouped_top_k"}:
            raise PositionContractError(f"unsupported position strategy: {mode}")
        selected, normalized = _select_rows(
            current,
            {"selection": strategy},
            group_field=group_field,
        )
        current = selected
        applied.append(
            {
                "mode": mode,
                "input_count": before,
                "output_count": len(current),
                "parameters": normalized,
            }
        )
    return current, applied


def resolve_positions(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve one preregistered position set and verify its bank structure."""

    control = _mapping(config.get("position_control"), field="position_control")
    source = _mapping(control.get("source"), field="position_control.source")
    source_kind = _string(source.get("kind"), field="position_control.source.kind")
    position_method: str | None = None
    if source_kind == "explicit":
        rows = _explicit_rows(source)
        source_ref: dict[str, str] | None = None
    elif source_kind in {"selector_manifest", "audit_manifest", "native_manifest"}:
        rows, source_ref = _load_manifest(source)
    elif source_kind == "position_method_manifest":
        position_method = _string(
            source.get("method"), field="position_control.source.method"
        )
        if position_method not in POSITION_METHODS:
            raise PositionContractError(
                f"unsupported position method: {position_method}"
            )
        rows, source_ref = _load_manifest(source)
        source_ref["position_method"] = position_method
    else:
        raise PositionContractError(f"unsupported position source kind: {source_kind}")

    component_field = _string(
        control.get("component_field"), field="position_control.component_field"
    )
    group_field_raw = control.get("group_field")
    group_field = (
        _string(group_field_raw, field="position_control.group_field")
        if group_field_raw is not None
        else None
    )
    rows, strategies = _apply_strategies(
        rows,
        control,
        component_field=component_field,
        group_field=group_field,
    )
    legacy_selection = (
        strategies[0]["legacy_selection"]
        if len(strategies) == 1 and "legacy_selection" in strategies[0]
        else None
    )
    positions: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        component = _string(row.get(component_field), field=f"position[{index}].{component_field}")
        item: dict[str, Any] = {"component_id": component}
        if group_field is not None:
            item["group"] = _string(
                row.get(group_field), field=f"position[{index}].{group_field}"
            )
        positions.append(item)

    expected_count = _positive_integer(
        control.get("expected_count"), field="position_control.expected_count"
    )
    if len(positions) != expected_count:
        raise PositionContractError(
            f"position count mismatch: expected {expected_count}, got {len(positions)}"
        )

    distinct_global = control.get("distinct_global")
    if not isinstance(distinct_global, bool):
        raise PositionContractError("position_control.distinct_global must be explicit")
    components = [row["component_id"] for row in positions]
    if distinct_global and len(components) != len(set(components)):
        raise PositionContractError("position_control requires globally distinct components")

    composition = _mapping(control.get("composition"), field="position_control.composition")
    mode = _string(composition.get("mode"), field="position_control.composition.mode")
    if mode not in COMPOSITION_MODES:
        raise PositionContractError(f"unsupported position composition mode: {mode}")
    quotas_raw = composition.get("quotas")
    quotas: dict[str, int] = {}
    bank_order: list[str]
    if mode in GROUPED_MODES:
        if group_field is None:
            raise PositionContractError(f"{mode} requires position_control.group_field")
        quotas_map = _mapping(quotas_raw, field="position_control.composition.quotas")
        quotas = {
            _string(group, field="quota group"): _positive_integer(
                count, field=f"position_control.composition.quotas.{group}"
            )
            for group, count in quotas_map.items()
        }
        actual: dict[str, int] = {}
        for row in positions:
            group = row["group"]
            actual[group] = actual.get(group, 0) + 1
        if actual != quotas:
            raise PositionContractError(f"position bank mismatch: expected {quotas}, got {actual}")
        raw_order = composition.get("bank_order")
        if not isinstance(raw_order, list) or not raw_order:
            raise PositionContractError(f"{mode} requires composition.bank_order")
        bank_order = [
            _string(group, field="position_control.composition.bank_order[]")
            for group in raw_order
        ]
        if len(bank_order) != len(set(bank_order)) or set(bank_order) != set(quotas):
            raise PositionContractError(
                "composition.bank_order must contain every quota group exactly once"
            )
    elif quotas_raw is not None:
        raise PositionContractError(f"{mode} must not define group quotas")
    else:
        if composition.get("bank_order") is not None:
            raise PositionContractError(f"{mode} must not define bank_order")
        bank_order = ["all"]

    merge = _mapping(
        composition.get("merge"), field="position_control.composition.merge"
    )
    operator = _string(
        merge.get("operator"), field="position_control.composition.merge.operator"
    )
    alpha_policy = _string(
        merge.get("alpha_policy"),
        field="position_control.composition.merge.alpha_policy",
    )
    retrain_after_merge = merge.get("retrain_after_merge")
    if operator not in MERGE_OPERATORS:
        raise PositionContractError(f"unsupported merge operator: {operator}")
    if alpha_policy not in ALPHA_POLICIES:
        raise PositionContractError(f"unsupported alpha policy: {alpha_policy}")
    if not isinstance(retrain_after_merge, bool):
        raise PositionContractError("composition.merge.retrain_after_merge must be boolean")
    if mode in GROUPED_MODES and operator == "identity":
        raise PositionContractError(f"{mode} cannot use merge operator identity")
    if mode not in GROUPED_MODES and operator == "sum":
        raise PositionContractError(f"{mode} cannot sum multiple banks")

    plan: dict[str, Any] = {
        "component_type": _string(
            control.get("component_type"), field="position_control.component_type"
        ),
        "source_kind": source_kind,
        "composition": {
            "mode": mode,
            "quotas": quotas,
            "bank_order": bank_order,
            "merge": {
                "operator": operator,
                "alpha_policy": alpha_policy,
                "retrain_after_merge": retrain_after_merge,
            },
        },
        "distinct_global": distinct_global,
        "selection": legacy_selection,
        "strategies": strategies,
        "positions": positions,
    }
    if position_method is not None:
        plan["position_method"] = position_method
    if source_ref is not None:
        plan["source_manifest"] = source_ref
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve one registered position set.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    plan = resolve_positions(config)
    rendered = json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
