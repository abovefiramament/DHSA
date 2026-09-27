"""ConFiQA Site data controller.

This adapter owns the frozen Context-DPO reconstruction: exact official prompt
rendering, valid-group admission/refill, selector-derived train/validation,
the fixed alpha window, and held-out valid-row refill.  It never performs
model work or reads a machine path beyond its registered source mirror.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .._shared.controller_api import (
    ControllerError,
    execute_dataset_controller,
    load_registered_rows,
    require_keys,
    write_manifest,
)
from .._shared.prompt_registry import resolve_prompt


DATASET = "confiqa"


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [item for item in value if isinstance(item, str) and item]
    return []


def _first_present(row: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return None


def load_source(source_spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Retain official-order rows, including invalid candidates for refill audit."""

    rows = load_registered_rows(source_spec)
    source_name = str(source_spec.get("source_id", "confiqa"))
    normalized: list[dict[str, Any]] = []
    for source_index, row in enumerate(rows):
        question = row.get("question")
        context = _first_present(row, ("cf_context", "orig_context", "context"))
        target = _strings(_first_present(row, ("cf_answer", "cf_answers", "answer")))
        reference = _strings(_first_present(row, ("orig_answer", "prior_answer", "orig_answers")))
        item = {
            **dict(row),
            "sample_id": f"{source_name}-{source_index:06d}",
            "source_index": source_index,
            "question": question if isinstance(question, str) else "",
            "context": context if isinstance(context, str) else "",
            "target_answers": target,
            "reference_answers": reference,
        }
        normalized.append(item)
    return normalized


def _invalid_reason(row: Mapping[str, Any]) -> str | None:
    if not str(row.get("question", "")).strip() or not str(row.get("context", "")).strip():
        return "missing_required_question_context_or_endpoint"
    target = _strings(row.get("target_answers"))
    reference = _strings(row.get("reference_answers"))
    if not target or not reference:
        return "missing_required_question_context_or_endpoint"
    target_clean = [item for item in target if item.strip()]
    reference_clean = [item for item in reference if item.strip()]
    if not target_clean or not reference_clean:
        return "empty_endpoint_after_alias_cleanup"
    if {item.strip() for item in target_clean} == {item.strip() for item in reference_clean}:
        return "identical_normalized_competition_endpoints"
    return None


def _prepared_row(
    row: Mapping[str, Any],
    *,
    prompt_registry: Mapping[str, Any],
    prompt_id: str,
) -> dict[str, Any]:
    prompt, prompt_info = resolve_prompt(
        prompt_registry,
        prompt_id,
        {"context": row["context"], "question": row["question"]},
    )
    if prompt != f"{row['context']}\nQ: {row['question']}\nA: ":
        raise ControllerError("ConFiQA official_rag prompt contract drift")
    target = _strings(row["target_answers"])
    reference = _strings(row["reference_answers"])
    if not target or not reference:
        raise ControllerError("attempted to prepare invalid ConFiQA competition row")
    return {
        **dict(row),
        "prompt": prompt,
        "official_rag": prompt,
        "chosen": target[0],
        "rejected": reference[0],
        "chosen_answers": target,
        "rejected_answers": reference,
        "prompt_registration": prompt_info,
    }


def _requested(data_spec: Mapping[str, Any], role: str, default: int) -> int:
    registered = data_spec.get("role_counts")
    if isinstance(registered, Mapping) and role in registered:
        count = registered[role]
        if isinstance(count, int) and count > 0:
            return count
        raise ControllerError(f"registered count for {role} must be positive")
    scaled = data_spec.get("scaled_counts")
    if isinstance(scaled, Mapping) and role in scaled:
        count = scaled[role]
        if isinstance(count, int) and count > 0:
            return count
        raise ControllerError(f"scaled count for {role} must be positive")
    return default


def _admit(
    rows: Sequence[Mapping[str, Any]],
    *,
    start: int,
    target: int,
    prompt_registry: Mapping[str, Any],
    prompt_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    admitted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    cursor = start
    while cursor < len(rows) and len(admitted) < target:
        row = rows[cursor]
        reason = _invalid_reason(row)
        if reason is None:
            admitted.append(_prepared_row(row, prompt_registry=prompt_registry, prompt_id=prompt_id))
        else:
            rejected.append({"source_index": row["source_index"], "sample_id": row["sample_id"], "reason": reason})
        cursor += 1
    if len(admitted) != target:
        raise ControllerError(
            f"ConFiQA source exhausted while admitting {target} rows from source index {start}"
        )
    return admitted, rejected, cursor


def _fixed_window(
    rows: Sequence[Mapping[str, Any]],
    *,
    start: int,
    count: int,
    prompt_registry: Mapping[str, Any],
    prompt_id: str,
) -> list[dict[str, Any]]:
    window = rows[start : start + count]
    if len(window) != count:
        raise ControllerError("ConFiQA fixed alpha window exceeds source rows")
    invalid = [row["source_index"] for row in window if _invalid_reason(row) is not None]
    if invalid:
        raise ControllerError(
            "ConFiQA fixed nonrefilled alpha window contains invalid rows: "
            + ",".join(str(index) for index in invalid)
        )
    return [
        _prepared_row(row, prompt_registry=prompt_registry, prompt_id=prompt_id)
        for row in window
    ]


def _materialize_single(
    source_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    source_specs: Mapping[str, Mapping[str, Any]],
    data_spec: Mapping[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if len(source_rows) != 1:
        raise ControllerError("ConFiQA Site cell requires exactly one registered subset source")
    source_name, raw_rows = next(iter(source_rows.items()))
    construction = data_spec.get("construction")
    prompt_registry = data_spec.get("prompt_registry")
    if not isinstance(construction, Mapping) or not isinstance(prompt_registry, Mapping):
        raise ControllerError("ConFiQA requires construction and prompt_registry settings")
    selections = prompt_registry.get("selections", {}).get(DATASET, {})
    prompt_id = selections.get("official_rag") if isinstance(selections, Mapping) else None
    if not isinstance(prompt_id, str):
        raise ControllerError("ConFiQA official_rag prompt_id is missing")
    selector_spec = construction.get("selector")
    audit_spec = construction.get("head_audit")
    alpha_spec = construction.get("alpha_dev")
    test_spec = construction.get("test")
    if not all(isinstance(value, Mapping) for value in (selector_spec, alpha_spec, test_spec)):
        raise ControllerError("ConFiQA construction settings are incomplete")

    selector_target = _requested(data_spec, "selector", 300)
    admitted, selector_rejected, selector_stop = _admit(
        raw_rows,
        start=int(selector_spec["initial_window"][0]),
        target=selector_target,
        prompt_registry=prompt_registry,
        prompt_id=prompt_id,
    )
    training = [row for index, row in enumerate(admitted) if index % 5 != 0]
    validation = [row for index, row in enumerate(admitted) if index % 5 == 0]
    training_target = _requested(data_spec, "training", 240 if selector_target == 300 else len(training))
    validation_target = _requested(data_spec, "validation", 60 if selector_target == 300 else len(validation))
    if len(training) < training_target or len(validation) < validation_target:
        raise ControllerError("selector-derived ConFiQA train/validation counts cannot meet profile")
    training = training[:training_target]
    validation = validation[:validation_target]

    subset = data_spec.get("subset")
    requested_roles = data_spec.get("roles")
    include_audit = isinstance(requested_roles, Mapping) and "head_audit" in requested_roles
    audit: list[dict[str, Any]] = []
    audit_window: list[int] | None = None
    if include_audit:
        if not isinstance(audit_spec, Mapping):
            raise ControllerError("ConFiQA head-audit construction is missing")
        audit_windows = audit_spec.get("window_by_subset")
        if not isinstance(audit_windows, Mapping) or subset not in audit_windows:
            raise ControllerError("ConFiQA head-audit window is missing for subset")
        audit_start, audit_stop = audit_windows[subset]
        audit = _fixed_window(
            raw_rows,
            start=int(audit_start),
            count=_requested(
                data_spec, "head_audit", int(audit_stop) - int(audit_start)
            ),
            prompt_registry=prompt_registry,
            prompt_id=prompt_id,
        )
        audit_window = [int(audit_start), int(audit_start) + len(audit)]
    windows = alpha_spec.get("window_by_subset")
    if not isinstance(subset, str) or not isinstance(windows, Mapping) or subset not in windows:
        raise ControllerError("ConFiQA alpha window is missing for subset")
    alpha_start, alpha_stop = windows[subset]
    alpha_default = int(alpha_stop) - int(alpha_start)
    alpha_target = _requested(data_spec, "alpha_dev", alpha_default)
    if bool(alpha_spec.get("refill_rejected", False)):
        alpha, alpha_rejected, alpha_stop_actual = _admit(
            raw_rows,
            start=int(alpha_start),
            target=alpha_target,
            prompt_registry=prompt_registry,
            prompt_id=prompt_id,
        )
    else:
        alpha = _fixed_window(
            raw_rows,
            start=int(alpha_start),
            count=alpha_target,
            prompt_registry=prompt_registry,
            prompt_id=prompt_id,
        )
        alpha_rejected = []
        alpha_stop_actual = int(alpha_start) + len(alpha)
    test, test_rejected, test_stop = _admit(
        raw_rows,
        start=int(test_spec["source_start"]),
        target=_requested(data_spec, "test", int(test_spec["target_valid_rows"])),
        prompt_registry=prompt_registry,
        prompt_id=prompt_id,
    )
    roles = {
        "selector": admitted,
        "training": training,
        "validation": validation,
        "alpha_dev": alpha,
        "test": test,
    }
    if include_audit:
        roles["head_audit"] = audit
    manifest = {
        "dataset": DATASET,
        "role_mode": "named_roles",
        "sources": {
            source_name: {
                "source": {key: value for key, value in source_specs[source_name].items() if key != "location"},
                "validated_rows": len(raw_rows),
            }
        },
        "prompt_registration": {
            "registry_path": prompt_registry["registry_path"],
            "prompt_id": prompt_id,
            "exact_template": "{context}\\nQ: {question}\\nA: ",
        },
        "construction_trace": {
            "selector": {
                "source_start": int(selector_spec["initial_window"][0]),
                "target_admitted": selector_target,
                "source_stop_exclusive": selector_stop,
                "rejections": selector_rejected,
                "train_rule": "prepared_admission_index_mod_5_nonzero",
            **({"head_audit": {"source_window": audit_window, "refill": False}} if include_audit else {}),
                "validation_rule": "prepared_admission_index_mod_5_zero",
            },
            "alpha_dev": {
                "source_start": int(alpha_start),
                "target_admitted": len(alpha),
                "source_stop_exclusive": alpha_stop_actual,
                "rejections": alpha_rejected,
                "refill": bool(alpha_spec.get("refill_rejected", False)),
            },
            "test": {
                "source_start": int(test_spec["source_start"]),
                "target_admitted": len(test),
                "source_stop_exclusive": test_stop,
                "rejections": test_rejected,
            },
        },
        "role_counts": {name: len(rows) for name, rows in roles.items()},
        "overlap_policy": {"allow": [["selector", "training"], ["selector", "validation"]]},
    }
    return roles, manifest


def materialize(
    source_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    source_specs: Mapping[str, Mapping[str, Any]],
    data_spec: Mapping[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Materialize one Site subset or a fixed-order joint Performance view.

    Joint mode does not introduce a second split algorithm. Each registered
    subset is independently materialized by the Site rule above, then matching
    roles are concatenated in the protocol-declared order.
    """

    if data_spec.get("subset") != "joint":
        return _materialize_single(source_rows, source_specs, data_spec)

    subsets = data_spec.get("subsets")
    subset_sources = data_spec.get("subset_sources")
    if subsets != ["qa", "mr", "mc"] or not isinstance(subset_sources, Mapping):
        raise ControllerError(
            "ConFiQA joint mode requires subsets [qa, mr, mc] and subset_sources"
        )
    if set(subset_sources) != set(subsets):
        raise ControllerError("ConFiQA joint subset_sources must cover qa, mr, and mc")

    merged_roles: dict[str, list[dict[str, Any]]] = {}
    merged_sources: dict[str, Any] = {}
    traces: dict[str, Any] = {}
    prompt_registration = None
    overlap_policy = None
    for subset in subsets:
        source_name = subset_sources[subset]
        if not isinstance(source_name, str) or source_name not in source_rows:
            raise ControllerError(f"ConFiQA joint source is missing for subset {subset}")
        single_spec = dict(data_spec)
        single_spec["subset"] = subset
        single_spec.pop("subsets", None)
        single_spec.pop("subset_sources", None)
        role_counts_by_subset = data_spec.get("role_counts_by_subset")
        if isinstance(role_counts_by_subset, Mapping):
            counts = role_counts_by_subset.get(subset)
            if not isinstance(counts, Mapping):
                raise ControllerError(
                    f"ConFiQA joint role counts are missing for subset {subset}"
                )
            single_spec["role_counts"] = {
                role: int(count) for role, count in counts.items()
            }
        roles, manifest = _materialize_single(
            {source_name: source_rows[source_name]},
            {source_name: source_specs[source_name]},
            single_spec,
        )
        for role, rows in roles.items():
            merged_roles.setdefault(role, []).extend(
                [
                    {
                        **row,
                        "source_sample_id": row["sample_id"],
                        "sample_id": f"{subset}::{row['sample_id']}",
                        "source_subset": subset,
                    }
                    for row in rows
                ]
            )
        merged_sources.update(manifest["sources"])
        traces[subset] = manifest["construction_trace"]
        prompt_registration = prompt_registration or manifest["prompt_registration"]
        overlap_policy = overlap_policy or manifest["overlap_policy"]

    return merged_roles, {
        "dataset": DATASET,
        "role_mode": "named_roles",
        "composition": {
            "mode": "joint_after_independent_site_materialization",
            "subset_order": list(subsets),
        },
        "sources": merged_sources,
        "prompt_registration": prompt_registration,
        "construction_trace": {"by_subset": traces},
        "role_counts": {
            name: len(rows) for name, rows in merged_roles.items()
        },
        "role_counts_by_subset": {
            subset: {
                role: sum(1 for row in rows if row["source_subset"] == subset)
                for role, rows in merged_roles.items()
            }
            for subset in subsets
        },
        "overlap_policy": overlap_policy,
    }


def write_dataset_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    write_manifest(path, {"dataset": DATASET, **dict(payload)})


def execute(data_spec: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    return execute_dataset_controller(
        data_spec=data_spec,
        output_dir=output_dir,
        load_source=load_source,
        materialize=materialize,
    )
