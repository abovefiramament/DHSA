"""TL;DR formal Site roles reconstructed from complete pinned source projections."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from .._shared.controller_api import (
    ControllerError,
    execute_dataset_controller,
    load_registered_rows,
    write_manifest,
)
from .._shared.prompt_registry import resolve_prompt


DATASET = "tldr"
_REQUIRED_SOURCES = {"tldr_pairs", "tldr_test_prompts"}


def _count(data_spec: Mapping[str, Any], role: str) -> int:
    scaled = data_spec.get("scaled_counts")
    if isinstance(scaled, Mapping) and role in scaled:
        value = scaled[role]
    else:
        value = data_spec["construction"]["roles"][role].get("target_count")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ControllerError(f"TL;DR role {role!r} requires a positive count")
    return value


def register_source_prompt(
    source_prompt: str,
    prompt_spec: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    if not isinstance(source_prompt, str) or not source_prompt:
        raise ControllerError("TL;DR requires a non-empty source prompt")
    selected = prompt_spec.get("selections", {}).get(DATASET, {})
    prompt_id = selected.get("source_prompt") if isinstance(selected, Mapping) else None
    if not isinstance(prompt_id, str) or not prompt_id:
        raise ControllerError("prompt_registry.selections.tldr.source_prompt is missing")
    return resolve_prompt(prompt_spec, prompt_id, {"source_prompt": source_prompt})


def load_source(source_spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    return load_registered_rows(source_spec)


def _pair(row: Mapping[str, Any], prompt_spec: Mapping[str, Any]) -> dict[str, Any]:
    source_prompt = row.get("prompt")
    chosen = row.get("chosen", row.get("y_plus_continuation", row.get("y_plus")))
    rejected = row.get("rejected", row.get("y_minus_continuation", row.get("y_minus")))
    prompt_id = row.get("prompt_id")
    if not all(isinstance(value, str) and value for value in (source_prompt, chosen, rejected, prompt_id)):
        raise ControllerError("TL;DR pair requires prompt_id, prompt, chosen, and rejected")
    prompt, registration = register_source_prompt(source_prompt, prompt_spec)
    return {
        **dict(row),
        "source_prompt": source_prompt,
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "prompt_registration": registration,
    }


def _admitted(row: Mapping[str, Any]) -> bool:
    value = row.get("admitted", 1)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return value is True or value == 1


def _evaluation_row(row: Mapping[str, Any], prompt_spec: Mapping[str, Any]) -> dict[str, Any]:
    source_prompt = row.get("prompt")
    reference = row.get("reference_summary", row.get("chosen"))
    prompt_id = row.get("prompt_id")
    if not all(isinstance(value, str) and value for value in (source_prompt, reference, prompt_id)):
        raise ControllerError("TL;DR test prompt requires prompt_id, prompt, and reference_summary")
    prompt, registration = register_source_prompt(source_prompt, prompt_spec)
    return {
        **dict(row),
        "source_prompt": source_prompt,
        "prompt": prompt,
        "reference_summary": reference,
        "prompt_registration": registration,
    }


def _take(rows: Sequence[Mapping[str, Any]], count: int, *, role: str) -> list[dict[str, Any]]:
    if len(rows) < count:
        raise ControllerError(f"TL;DR {role} has {len(rows)} rows; requires {count}")
    return [dict(row) for row in rows[:count]]


def _clean_partitions(
    rows: Sequence[Mapping[str, Any]],
    construction: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    clean = construction.get("clean_split")
    if not isinstance(clean, Mapping):
        raise ControllerError("TL;DR clean_split settings must be registered")
    contaminated_prefix = clean.get("contaminated_source_prefix_rows")
    source_rows = clean.get("source_rows")
    calibration_rows = clean.get("calibration_rows")
    original_final_guard_rows = clean.get("original_final_guard_rows")
    seed = clean.get("shuffle_seed")
    for field, value in (
        ("contaminated_source_prefix_rows", contaminated_prefix),
        ("source_rows", source_rows),
        ("calibration_rows", calibration_rows),
        ("original_final_guard_rows", original_final_guard_rows),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ControllerError(f"TL;DR clean_split.{field} must be non-negative")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ControllerError("TL;DR clean_split.shuffle_seed must be an integer")
    if len(rows) != source_rows:
        raise ControllerError(f"TL;DR test prompt mirror has {len(rows)} rows; requires {source_rows}")
    indexed: list[tuple[int, Mapping[str, Any]]] = []
    for row in rows:
        value = row.get("source_row_index")
        if isinstance(value, str) and value.isdigit():
            value = int(value)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ControllerError("TL;DR test source requires source_row_index")
        indexed.append((value, row))
    if [index for index, _ in indexed] != sorted(index for index, _ in indexed):
        raise ControllerError("TL;DR test prompt mirror must preserve source order")
    contaminated = [dict(row) for index, row in indexed if index < contaminated_prefix]
    clean_pool = [dict(row) for index, row in indexed if index >= contaminated_prefix]
    if len(contaminated) != contaminated_prefix:
        raise ControllerError("TL;DR contaminated source prefix is incomplete")
    random.Random(seed).shuffle(clean_pool)
    calibration = clean_pool[:calibration_rows]
    guard_start = calibration_rows
    guard_stop = guard_start + original_final_guard_rows
    original_final_guard = clean_pool[guard_start:guard_stop]
    test_reserve = clean_pool[guard_stop:]
    if len(calibration) != calibration_rows or len(original_final_guard) != original_final_guard_rows:
        raise ControllerError("TL;DR clean pool cannot satisfy registered partitions")
    expected_reserve = clean.get("test_reserve_rows")
    if len(test_reserve) != expected_reserve:
        raise ControllerError(
            f"TL;DR clean test reserve has {len(test_reserve)} rows; requires {expected_reserve}"
        )
    return calibration, original_final_guard, test_reserve


def _assert_prompt_disjointness(roles: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    ids = {name: {str(row["prompt_id"]) for row in rows} for name, rows in roles.items()}
    names = list(roles)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if {left, right} == {"selector", "training"}:
                continue
            if ids[left] & ids[right]:
                raise ControllerError(f"TL;DR roles {left} and {right} overlap by prompt_id")


def materialize(
    source_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    source_specs: Mapping[str, Mapping[str, Any]],
    data_spec: Mapping[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if set(source_rows) != _REQUIRED_SOURCES:
        raise ControllerError(f"TL;DR requires complete sources {sorted(_REQUIRED_SOURCES)}")
    construction = data_spec.get("construction")
    prompt_spec = data_spec.get("prompt_registry")
    if not isinstance(construction, Mapping) or not isinstance(prompt_spec, Mapping):
        raise ControllerError("TL;DR construction and prompt registry must be registered")
    pairs = [
        _pair(row, prompt_spec)
        for row in source_rows["tldr_pairs"]
        if _admitted(row)
    ]
    train_pool = [row for row in pairs if str(row.get("split", "")).lower() == "train"]
    validation_pool = [
        row for row in pairs if str(row.get("split", "")).lower() in {"val", "validation"}
    ]
    selector: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in train_pool:
        prompt_id = str(row["prompt_id"])
        if prompt_id in seen:
            continue
        seen.add(prompt_id)
        selector.append(dict(row))
        if len(selector) == _count(data_spec, "selector"):
            break
    if len(selector) != _count(data_spec, "selector"):
        raise ControllerError("TL;DR selector cannot reach its unique-prompt target")

    calibration, guard, test_reserve = _clean_partitions(
        source_rows["tldr_test_prompts"], construction
    )
    calibration_spec = construction.get("calibration")
    test_spec = construction.get("test")
    if not isinstance(calibration_spec, Mapping) or not isinstance(test_spec, Mapping):
        raise ControllerError("TL;DR calibration and test settings must be registered")
    audit_start, _audit_stop = calibration_spec["head_audit_range"]
    alpha_start, _alpha_stop = calibration_spec["alpha_dev_range"]
    reserve_start, _reserve_stop = calibration_spec["reserve_range"]
    ranges = {
        "head_audit": calibration_spec["head_audit_range"],
        "alpha_dev": calibration_spec["alpha_dev_range"],
        "reserve": calibration_spec["reserve_range"],
    }
    for role, value in ranges.items():
        if (
            not isinstance(value, list)
            or len(value) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
            or value[0] < 0
            or value[1] <= value[0]
            or _count(data_spec, role) > value[1] - value[0]
        ):
            raise ControllerError(f"TL;DR calibration range for {role} is invalid")
    evaluation_calibration = [_evaluation_row(row, prompt_spec) for row in calibration]
    test_source = test_spec.get("source")
    if test_source not in {"original_final_guard", "clean_test_reserve"}:
        raise ControllerError("TL;DR test.source must select a registered clean partition")
    test_rows = guard if test_source == "original_final_guard" else test_reserve
    evaluation_test = [_evaluation_row(row, prompt_spec) for row in test_rows]
    training_options = construction.get("training", {})
    shuffle_before_take = training_options.get("shuffle_before_take", False)
    if not isinstance(shuffle_before_take, bool):
        raise ControllerError("TL;DR training.shuffle_before_take must be boolean")
    training_pool = list(train_pool)
    if shuffle_before_take:
        sampling_seed = training_options.get("seed")
        if isinstance(sampling_seed, bool) or not isinstance(sampling_seed, int):
            raise ControllerError("TL;DR training pool shuffle requires an integer seed")
        random.Random(sampling_seed).shuffle(training_pool)
    roles = {
        "selector": selector,
        "training": _take(training_pool, _count(data_spec, "training"), role="training"),
        "validation": _take(validation_pool, _count(data_spec, "validation"), role="validation"),
        "head_audit": _take(evaluation_calibration[audit_start:], _count(data_spec, "head_audit"), role="head_audit"),
        "alpha_dev": _take(evaluation_calibration[alpha_start:], _count(data_spec, "alpha_dev"), role="alpha_dev"),
        "reserve": _take(evaluation_calibration[reserve_start:], _count(data_spec, "reserve"), role="reserve"),
        "test": _take(evaluation_test, _count(data_spec, "test"), role="test"),
    }
    _assert_prompt_disjointness(roles)
    return roles, {
        "dataset": DATASET,
        "role_mode": "named_roles",
        "sources": {
            name: {
                "source": {key: value for key, value in source_specs[name].items() if key != "location"},
                "validated_rows": len(rows),
            }
            for name, rows in source_rows.items()
        },
        "prompt_registration": {
            "registry_path": prompt_spec.get("registry_path"),
            "selection": prompt_spec.get("selections", {}).get(DATASET),
        },
        "construction_trace": {
            "selector": "first_unique_prompt_id_groups_from_train_pair_order",
            "training": (
                {"order": "python_random_shuffle_before_take", "seed": sampling_seed,
                 "target_count": _count(data_spec, "training")}
                if shuffle_before_take else "first_8192_train_pairs"
            ),
            "validation": "first_512_validation_pairs",
            "clean_calibration_rows": len(calibration),
            "original_final_guard_rows": len(guard),
            "test_reserve_rows": len(test_reserve),
            "head_audit_range": list(calibration_spec["head_audit_range"]),
            "alpha_dev_range": list(calibration_spec["alpha_dev_range"]),
            "reserve_range": list(calibration_spec["reserve_range"]),
            "test_first_rows": int(test_spec["first_rows"]),
            "test_source": test_source,
        },
        "role_counts": {name: len(rows) for name, rows in roles.items()},
        "overlap_policy": {"allow": [["selector", "training"]]},
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
