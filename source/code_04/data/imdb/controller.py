"""IMDb formal Site data controller over raw pinned train/test mirrors."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .._shared.controller_api import (
    ControllerError,
    execute_dataset_controller,
    load_registered_rows,
    write_manifest,
)
from .._shared.prompt_registry import resolve_prompt
from .model_native import prepare_roles


DATASET = "imdb"
_REQUIRED_SOURCES = {"imdb_train", "imdb_test"}
_PROMPT_ROLES = ("good_state", "base_state", "preference_prefix")


def render_registered_prompts(
    prefix: str,
    prompt_spec: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(prefix, str) or not prefix:
        raise ControllerError("IMDb prompt registration requires a non-empty prefix")
    selections = prompt_spec.get("selections")
    selected = selections.get(DATASET) if isinstance(selections, Mapping) else None
    if not isinstance(selected, Mapping):
        raise ControllerError("prompt_registry.selections.imdb must be registered")
    prompts: dict[str, str] = {}
    registration: dict[str, Any] = {}
    for role in _PROMPT_ROLES:
        prompt_id = selected.get(role)
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ControllerError(f"IMDb prompt role {role!r} is not registered")
        prompts[role], registration[role] = resolve_prompt(
            prompt_spec,
            prompt_id,
            {"prefix": prefix},
        )
    return {"prefix": prefix, "prompts": prompts, "prompt_registration": registration}


def load_source(source_spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    return load_registered_rows(source_spec)


def _lineage_keys(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_split: str,
    role: str,
) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    for row in rows:
        value = row.get("source_row_index")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ControllerError(f"IMDb {role} row requires source_row_index")
        key = (source_split, value)
        if key in keys:
            raise ControllerError(f"IMDb {role} repeats an upstream source row")
        keys.add(key)
    return keys


def materialize(
    source_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    source_specs: Mapping[str, Mapping[str, Any]],
    data_spec: Mapping[str, Any],
    *,
    model_provider: Any,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    construction = data_spec.get("construction")
    frozen_roles = construction.get("frozen_role_sources") if isinstance(construction, Mapping) else None
    if frozen_roles is not None:
        if set(frozen_roles) != {"training", "selector", "validation", "test"}:
            raise ControllerError("IMDb frozen role reuse requires all four registered roles")
        roles = {name: [dict(row) for row in source_rows[source]] for name, source in frozen_roles.items()}
        for name, rows in roles.items():
            if len(rows) != construction["roles"][name]["target_count"]:
                raise ControllerError(f"IMDb frozen {name} count differs from registered construction")
        lineage = {
            name: {(row["source_split"], row["source_row_index"]) for row in rows}
            for name, rows in roles.items() if name != "training"
        }
        lineage["training"] = {
            tuple(match) for row in roles["training"] for match in row["source_comment_matches"]
        }
        for i, left in enumerate(lineage):
            for right in list(lineage)[i + 1:]:
                if lineage[left] & lineage[right]:
                    raise ControllerError(f"IMDb reused roles overlap: {left} / {right}")
        return roles, {
            "dataset": DATASET, "role_mode": "named_roles",
            "sources": {
                name: {"source": {key: value for key, value in spec.items() if key != "location"},
                       "validated_rows": len(source_rows[name])}
                for name, spec in source_specs.items()
            },
            "construction_trace": {"mode": "exact_frozen_role_reuse",
                "provenance": dict(data_spec["provenance"]),
                "generation_performed": False, "resampling_performed": False},
            "role_counts": {name: len(rows) for name, rows in roles.items()},
            "overlap_policy": {"allow": []},
        }
    pair_spec = construction.get("pair_preparation") if isinstance(construction, Mapping) else None
    if isinstance(pair_spec, Mapping) and pair_spec.get("source_kind") == "ma921_cleaned_token_pairs":
        if model_provider is None:
            raise ControllerError("IMDb public pair materialization requires the registered model provider")
        from .public_pairs import prepare_public_roles
        roles, trace = prepare_public_roles(source_rows, data_spec=data_spec, model_provider=model_provider)
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
            "construction_trace": trace,
            "role_counts": {name: len(rows) for name, rows in roles.items()},
            "overlap_policy": {"allow": []},
        }
    if set(source_rows) != _REQUIRED_SOURCES:
        raise ControllerError(f"IMDb requires raw sources {sorted(_REQUIRED_SOURCES)}")
    if model_provider is None:
        raise ControllerError("IMDb model-native construction requires the shared model provider")
    prompt_spec = data_spec.get("prompt_registry")
    if not isinstance(prompt_spec, Mapping):
        raise ControllerError("IMDb data.prompt_registry must be registered")
    roles, trace = prepare_roles(
        source_rows["imdb_train"],
        source_rows["imdb_test"],
        data_spec=data_spec,
        model_provider=model_provider,
        render_prompts=render_registered_prompts,
    )
    role_lineage = {
        name: _lineage_keys(
            rows,
            source_split="test" if name == "test" else "train",
            role=name,
        )
        for name, rows in roles.items()
    }
    names = list(roles)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if role_lineage[left] & role_lineage[right]:
                raise ControllerError(
                    f"IMDb roles {left} and {right} overlap by upstream source row"
                )
    return roles, {
        "dataset": DATASET,
        "role_mode": "named_roles",
        "sources": {
            name: {
                "source": {
                    key: value
                    for key, value in source_specs[name].items()
                    if key != "location"
                },
                "validated_rows": len(rows),
            }
            for name, rows in source_rows.items()
        },
        "prompt_registration": {
            "registry_path": prompt_spec.get("registry_path"),
            "selection": prompt_spec.get("selections", {}).get(DATASET),
        },
        "construction_trace": trace,
        "role_counts": {name: len(rows) for name, rows in roles.items()},
        "overlap_policy": {"allow": []},
    }


def write_dataset_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    write_manifest(path, {"dataset": DATASET, **dict(payload)})


def execute(
    data_spec: Mapping[str, Any],
    output_dir: Path,
    *,
    model_provider: Any = None,
) -> dict[str, Any]:
    def materialize_bound(
        source_rows: Mapping[str, Sequence[Mapping[str, Any]]],
        source_specs: Mapping[str, Mapping[str, Any]],
        resolved_data_spec: Mapping[str, Any],
    ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
        return materialize(
            source_rows,
            source_specs,
            resolved_data_spec,
            model_provider=model_provider,
        )

    return execute_dataset_controller(
        data_spec=data_spec,
        output_dir=output_dir,
        load_source=load_source,
        materialize=materialize_bound,
    )
