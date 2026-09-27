"""Tulu multi-source data-controller adapter.

Tulu is the explicit mixed-data case. Source-specific parsers normalize each
registered task source; the shared controller preserves task provenance while
materializing isolated experiment roles.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .._shared.controller_api import (
    ControllerError,
    execute_dataset_controller,
    load_registered_rows,
    materialize_registered_splits,
    require_keys,
    write_manifest,
)


DATASET = "tulu"


def load_source(source_spec: Mapping[str, Any]):
    return load_registered_rows(source_spec)


def validate_source(
    rows: Sequence[Mapping[str, Any]],
    source_spec: Mapping[str, Any],
) -> dict[str, Any]:
    schema = source_spec.get("schema")
    if schema == "preference_pair":
        fields = ("sample_id", "task_id", "prompt", "preferred", "dispreferred")
    elif schema in {"selector_state", "evaluation_prompt"}:
        fields = ("sample_id", "task_id", "prompt")
    else:
        raise ControllerError(f"unsupported registered Tulu schema: {schema!r}")
    for index, row in enumerate(rows):
        require_keys(row, fields, context=f"Tulu {schema} row {index}")
    return {
        "source": {key: value for key, value in source_spec.items() if key != "location"},
        "schema": schema,
        "validated_rows": len(rows),
    }


def materialize(
    source_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    source_specs: Mapping[str, Mapping[str, Any]],
    data_spec: Mapping[str, Any],
):
    validations = {
        name: validate_source(rows, source_specs[name])
        for name, rows in source_rows.items()
    }
    splits, manifest = materialize_registered_splits(source_rows, data_spec)
    for split_name, rows in splits.items():
        for index, row in enumerate(rows):
            require_keys(
                row,
                ("sample_id", "task_id", "source_name", "data_role"),
                context=f"Tulu materialized {split_name} row {index}",
            )
    return splits, {
        "dataset": DATASET,
        "multi_source": len(source_rows) > 1,
        "sources": validations,
        **manifest,
    }


def write_dataset_manifest(path, payload: Mapping[str, Any]) -> None:
    write_manifest(path, {"dataset": DATASET, **dict(payload)})


def execute(data_spec: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    return execute_dataset_controller(
        data_spec=data_spec,
        output_dir=output_dir,
        load_source=load_source,
        materialize=materialize,
    )
