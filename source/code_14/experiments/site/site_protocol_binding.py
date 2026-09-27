"""Bind the single new-framework Site protocol to executable matrix rows.

This module is intentionally self-contained. It does not import a legacy
protocol, legacy runner, or dataset-specific reproduction script. The JSON
matrix is the sole scientific source; shared controllers own all execution
stages after a row is produced.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterator, Mapping

from experiments.shared.model_registry import (
    DEFAULT_MODEL_REGISTRY,
    ModelRegistryError,
    resolve_registered_evaluator_model,
    resolve_registered_model,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATRIX = REPOSITORY_ROOT / "configs" / "site" / "site_cell_matrix_20260829_v1.json"
FLOW_ID = "site_cast_fixed_flow_v1"
POSITION_METHOD_BY_SELECTOR = {
    "rcm_zero_signed": "rcm_zero",
    "rcm_patch_signed": "rcm_patch",
    "iti": "iti",
    "random": "random",
}
DATA_ROLES_BY_DATASET = {
    "confiqa": ("selector", "training", "validation", "alpha_dev", "test"),
    "imdb": ("selector", "training", "validation", "test"),
    "tldr": (
        "selector",
        "training",
        "validation",
        "head_audit",
        "alpha_dev",
        "reserve",
        "test",
    ),
}
REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class SiteProtocolBindingError(ValueError):
    """Raised when the formal Site matrix is ambiguous or incomplete."""


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SiteProtocolBindingError(f"cannot load formal Site matrix: {path}") from exc
    if not isinstance(value, dict):
        raise SiteProtocolBindingError("formal Site matrix must contain one object")
    return value


def _validate_data_provenance(matrix: Mapping[str, Any]) -> None:
    """Require an immutable network-source-to-role lineage in the protocol."""

    protocol = matrix.get("protocol")
    if not isinstance(protocol, Mapping):
        raise SiteProtocolBindingError("formal Site protocol section is missing")
    data = protocol.get("data")
    if not isinstance(data, Mapping):
        raise SiteProtocolBindingError("formal Site protocol data section is missing")
    provenance = data.get("provenance")
    if not isinstance(provenance, Mapping) or provenance.get("schema_version") != 1:
        raise SiteProtocolBindingError("Site data provenance schema_version must be 1")
    if provenance.get("network_source_required") is not True:
        raise SiteProtocolBindingError("Site data provenance must require network sources")
    lineage = provenance.get("lineage_contract")
    required_lineage = {
        "download_pinned_source",
        "decode_and_normalize",
        "filter_and_deduplicate",
        "deterministic_order",
        "assign_disjoint_roles",
        "freeze_output_files",
    }
    if not isinstance(lineage, list) or not required_lineage.issubset(set(lineage)):
        raise SiteProtocolBindingError("Site data provenance lineage contract is incomplete")
    freeze_outputs = provenance.get("freeze_outputs")
    if not isinstance(freeze_outputs, Mapping):
        raise SiteProtocolBindingError("Site data provenance freeze_outputs is missing")
    if not isinstance(freeze_outputs.get("dataset_manifest"), str):
        raise SiteProtocolBindingError("Site provenance must name dataset_manifest.json")
    if freeze_outputs.get("data_request") != "data_request.local.json":
        raise SiteProtocolBindingError(
            "Site provenance must register data_request.local.json"
        )
    if not isinstance(freeze_outputs.get("role_manifest_pattern"), str):
        raise SiteProtocolBindingError("Site provenance must name role manifests")
    required_fields = freeze_outputs.get("required_fields")
    if not isinstance(required_fields, list) or not {
        "upstream_source_ids",
        "operation_ids",
        "source_files",
        "output_files",
        "role_assignment",
    }.issubset(set(required_fields)):
        raise SiteProtocolBindingError("Site provenance freeze fields are incomplete")
    datasets = provenance.get("datasets")
    if not isinstance(datasets, Mapping) or set(datasets) != {"confiqa", "imdb", "tldr"}:
        raise SiteProtocolBindingError("Site provenance must cover ConfiQA, IMDb, and TL;DR")
    for dataset, spec in datasets.items():
        if not isinstance(spec, Mapping):
            raise SiteProtocolBindingError(f"{dataset} provenance must be an object")
        adapter_id = spec.get("adapter_id")
        if not isinstance(adapter_id, str) or not adapter_id:
            raise SiteProtocolBindingError(f"{dataset} provenance adapter_id is missing")
        sources = spec.get("sources")
        if not isinstance(sources, list) or not sources:
            raise SiteProtocolBindingError(f"{dataset} provenance sources are missing")
        source_ids: set[str] = set()
        for source in sources:
            if not isinstance(source, Mapping):
                raise SiteProtocolBindingError(f"{dataset} provenance source must be an object")
            source_id = source.get("id")
            uri = source.get("uri")
            revision = source.get("revision")
            files = source.get("files")
            if (
                not isinstance(source_id, str)
                or not source_id
                or source_id in source_ids
                or not isinstance(uri, str)
                or not uri.startswith(("https://", "http://"))
                or not isinstance(revision, str)
                or REVISION_PATTERN.fullmatch(revision) is None
                or not isinstance(files, list)
                or not files
            ):
                raise SiteProtocolBindingError(f"{dataset} provenance source identity is incomplete")
            source_ids.add(source_id)
            for file_spec in files:
                if not isinstance(file_spec, Mapping):
                    raise SiteProtocolBindingError(f"{dataset} provenance file must be an object")
                file_uri = file_spec.get("uri")
                file_path = file_spec.get("path")
                if (
                    not isinstance(file_path, str)
                    or not file_path
                    or not isinstance(file_uri, str)
                    or not file_uri.startswith(("https://", "http://"))
                ):
                    raise SiteProtocolBindingError(f"{dataset} provenance file identity is incomplete")
            parent = source.get("parent_source_id")
            if parent is not None and (not isinstance(parent, str) or not parent):
                raise SiteProtocolBindingError(f"{dataset} provenance parent source is invalid")
        for source in sources:
            parent = source.get("parent_source_id")
            if parent is not None and parent not in source_ids:
                raise SiteProtocolBindingError(f"{dataset} provenance parent source is unknown")
        pipeline = spec.get("pipeline")
        if not isinstance(pipeline, list) or not pipeline:
            raise SiteProtocolBindingError(f"{dataset} provenance pipeline is missing")
        operation_ids: set[str] = set()
        for operation in pipeline:
            if not isinstance(operation, Mapping):
                raise SiteProtocolBindingError(f"{dataset} provenance operation must be an object")
            operation_id = operation.get("id")
            version = operation.get("version")
            if (
                not isinstance(operation_id, str)
                or not operation_id
                or operation_id in operation_ids
                or not isinstance(version, int)
                or isinstance(version, bool)
                or version <= 0
                or not isinstance(operation.get("parameters"), Mapping)
            ):
                raise SiteProtocolBindingError(f"{dataset} provenance operation identity is incomplete")
            operation_ids.add(operation_id)
        ordering = spec.get("ordering")
        if not isinstance(ordering, Mapping):
            raise SiteProtocolBindingError(f"{dataset} provenance ordering is missing")
        for phase in ("source", "training"):
            rule = ordering.get(phase)
            if not isinstance(rule, Mapping) or not isinstance(rule.get("enabled"), bool):
                raise SiteProtocolBindingError(f"{dataset} provenance {phase} ordering is incomplete")
            if rule["enabled"]:
                if not isinstance(rule.get("seed"), int) or isinstance(rule.get("seed"), bool):
                    raise SiteProtocolBindingError(f"{dataset} provenance {phase} shuffle seed is missing")
            elif not isinstance(rule.get("algorithm"), str) or not rule["algorithm"]:
                raise SiteProtocolBindingError(f"{dataset} provenance {phase} order algorithm is missing")
        roles = spec.get("role_mapping")
        expected_roles = set(DATA_ROLES_BY_DATASET[dataset])
        if not isinstance(roles, Mapping) or set(roles) != expected_roles:
            raise SiteProtocolBindingError(f"{dataset} provenance role mapping is incomplete")
        for role in DATA_ROLES_BY_DATASET[dataset]:
            mapping = roles[role]
            if (
                not isinstance(mapping, Mapping)
                or not isinstance(mapping.get("source_split"), str)
                or not mapping["source_split"]
                or not isinstance(mapping.get("assignment"), str)
                or not mapping["assignment"]
                or not isinstance(mapping.get("counts_ref"), str)
                or not mapping["counts_ref"].startswith("protocol.data.")
            ):
                raise SiteProtocolBindingError(f"{dataset} provenance mapping for {role} is incomplete")


def _validate_randomness(matrix: Mapping[str, Any]) -> None:
    """Require every stochastic Site operation to use a registered seed."""

    protocol = matrix.get("protocol")
    if not isinstance(protocol, Mapping):
        raise SiteProtocolBindingError("formal Site protocol section is missing")
    randomness = protocol.get("randomness")
    if not isinstance(randomness, Mapping) or randomness.get("schema_version") != 1:
        raise SiteProtocolBindingError("Site randomness registry schema_version must be 1")
    if randomness.get("implicit_seed_forbidden") is not True or randomness.get("time_or_process_seed_forbidden") is not True:
        raise SiteProtocolBindingError("Site randomness registry must forbid implicit seeds")
    dispatch = randomness.get("seed_dispatch")
    if not isinstance(dispatch, list) or set(dispatch) != {"python_random", "numpy", "torch_cpu", "torch_cuda", "dataloader_workers"}:
        raise SiteProtocolBindingError("Site randomness seed dispatch is incomplete")
    entries = randomness.get("entries")
    if not isinstance(entries, list) or not entries:
        raise SiteProtocolBindingError("Site randomness registry entries are missing")
    by_scope: dict[str, Mapping[str, Any]] = {}
    by_id: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise SiteProtocolBindingError("Site randomness entry must be an object")
        entry_id = entry.get("id")
        scope = entry.get("scope")
        if (
            not isinstance(entry_id, str)
            or not entry_id
            or entry_id in by_id
            or not isinstance(scope, str)
            or not scope
            or scope in by_scope
            or not isinstance(entry.get("rng"), str)
            or not entry["rng"]
            or not isinstance(entry.get("algorithm"), str)
            or not entry["algorithm"]
        ):
            raise SiteProtocolBindingError("Site randomness entry identity is incomplete")
        by_id.add(entry_id)
        by_scope[scope] = entry
        has_seed = "seed" in entry
        has_seeds = "seeds" in entry
        if has_seed == has_seeds:
            raise SiteProtocolBindingError(f"Site randomness entry {entry_id!r} must declare seed or seeds")
        if has_seed and (not isinstance(entry["seed"], int) or isinstance(entry["seed"], bool)):
            raise SiteProtocolBindingError(f"Site randomness entry {entry_id!r} seed is invalid")
        if has_seeds:
            seeds = entry["seeds"]
            if not isinstance(seeds, list) or not seeds or not all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds):
                raise SiteProtocolBindingError(f"Site randomness entry {entry_id!r} seeds are invalid")
    deterministic = randomness.get("deterministic_scopes")
    if not isinstance(deterministic, list) or not deterministic or not all(isinstance(item, str) and item for item in deterministic):
        raise SiteProtocolBindingError("Site deterministic scope registry is incomplete")
    selector_contract = matrix.get("selector_contract")
    if not isinstance(selector_contract, Mapping):
        raise SiteProtocolBindingError("Site selector contract is missing")
    primary = by_scope.get("position_search.primary")
    if primary is None or primary.get("seed") != selector_contract.get("primary_seed"):
        raise SiteProtocolBindingError("position-search primary seed drift")
    random_entry = by_scope.get("position_search.random")
    random_seeds = selector_contract.get("random_seeds")
    if random_entry is None or random_entry.get("seeds") != random_seeds:
        raise SiteProtocolBindingError("position-search Random seed registry drift")
    data = protocol.get("data")
    provenance = data.get("provenance") if isinstance(data, Mapping) else None
    datasets = provenance.get("datasets") if isinstance(provenance, Mapping) else None
    if not isinstance(datasets, Mapping):
        raise SiteProtocolBindingError("Site data provenance is missing for randomness validation")
    for dataset, spec in datasets.items():
        ordering = spec.get("ordering") if isinstance(spec, Mapping) else None
        if not isinstance(ordering, Mapping):
            raise SiteProtocolBindingError(f"{dataset} ordering is missing for randomness validation")
        for phase in ("source", "training"):
            rule = ordering.get(phase)
            if not isinstance(rule, Mapping):
                raise SiteProtocolBindingError(f"{dataset} {phase} ordering is missing")
            if rule.get("enabled"):
                scope = f"data.{dataset}.{phase}_order"
                registered = by_scope.get(scope)
                if registered is None or registered.get("seed") != rule.get("seed"):
                    raise SiteProtocolBindingError(f"{dataset} {phase} shuffle seed is not registered")
    cast = protocol.get("cast")
    if not isinstance(cast, Mapping):
        raise SiteProtocolBindingError("Site CAST protocol is missing")
    sv = cast.get("sv")
    common = sv.get("training_common") if isinstance(sv, Mapping) else None
    bank = by_scope.get("bank_training")
    if bank is None or not isinstance(common, Mapping) or bank.get("seed") != common.get("seed"):
        raise SiteProtocolBindingError("CAST bank-training seed drift")
    generation = by_scope.get("generation.all_datasets")
    sv_tasks = sv.get("tasks") if isinstance(sv, Mapping) else None
    reft = cast.get("reft")
    generation_configs = []
    if isinstance(sv_tasks, Mapping):
        generation_configs.extend(
            task.get("generation") for task in sv_tasks.values() if isinstance(task, Mapping)
        )
    if isinstance(reft, Mapping):
        generation_configs.append(reft.get("generation"))
    if (
        generation is None
        or not generation_configs
        or any(
            not isinstance(config, Mapping)
            or generation.get("seed") != config.get("seed")
            for config in generation_configs
        )
    ):
        raise SiteProtocolBindingError("CAST generation seed drift")


def _validate_scientific_contract(matrix: Mapping[str, Any]) -> None:
    protocol = matrix.get("protocol")
    if not isinstance(protocol, Mapping):
        raise SiteProtocolBindingError("formal Site protocol section is missing")
    boundary = protocol.get("implementation_boundary")
    if not isinstance(boundary, Mapping) or any(
        boundary.get(field) is not True
        for field in (
            "new_framework_only",
            "prior_protocol_code_forbidden",
            "prior_runner_or_backend_forbidden",
            "prior_frozen_data_or_result_import_forbidden",
            "public_source_reconstruction_required",
        )
    ):
        raise SiteProtocolBindingError("Site new-framework implementation boundary drift")
    positions = protocol.get("positions")
    data = protocol.get("data")
    cast = protocol.get("cast")
    evaluation = protocol.get("evaluation_profiles")
    if not all(isinstance(item, Mapping) for item in (positions, data, cast, evaluation)):
        raise SiteProtocolBindingError("formal Site scientific profiles are incomplete")
    by_dataset = positions.get("by_dataset")
    if not isinstance(by_dataset, Mapping) or {
        dataset: (spec.get("candidate_count"), spec.get("final_count"))
        for dataset, spec in by_dataset.items()
        if isinstance(spec, Mapping)
    } != {"confiqa": (8, 8), "imdb": (8, 8), "tldr": (16, 8)}:
        raise SiteProtocolBindingError("Site candidate/final position counts drift")
    rcm = positions.get("rcm")
    if (
        not isinstance(rcm, Mapping)
        or rcm.get("coarse_parent_layers_per_role") != 4
        or rcm.get("tldr_candidate_quota") != {
            "target_support": 8,
            "competitor_support": 8,
        }
        or rcm.get("tldr_final_role_quota") is not None
    ):
        raise SiteProtocolBindingError("Site RCM selection contract drift")
    profiles = data.get("construction_profiles")
    expected_roles = {
        "confiqa": {"selector": 300, "training": 240, "validation": 60, "alpha_dev": 120, "test": 2048},
        "imdb": {"selector": 300, "training": 1024, "validation": 256, "test": 2048},
        "tldr": {"selector": 300, "training": 8192, "validation": 512, "head_audit": 40, "alpha_dev": 40, "reserve": 240, "test": 2048},
    }
    if not isinstance(profiles, Mapping) or set(profiles) != set(expected_roles):
        raise SiteProtocolBindingError("Site data construction profiles are incomplete")
    for dataset, expected in expected_roles.items():
        role_specs = profiles[dataset].get("roles") if isinstance(profiles[dataset], Mapping) else None
        actual = {}
        if isinstance(role_specs, Mapping):
            for role, spec in role_specs.items():
                if isinstance(spec, Mapping):
                    actual[role] = spec.get("target_count", spec.get("max_count"))
        if actual != expected:
            raise SiteProtocolBindingError(f"{dataset} formal role counts drift")
    sv = cast.get("sv")
    reft = cast.get("reft")
    if not isinstance(sv, Mapping) or not isinstance(reft, Mapping):
        raise SiteProtocolBindingError("Site CAST SV/ReFT profiles are missing")
    common = sv.get("training_common")
    tasks = sv.get("tasks")
    if (
        not isinstance(common, Mapping)
        or common.get("lr") != 0.05
        or common.get("epochs") != 2
        or not isinstance(tasks, Mapping)
        or tasks.get("confiqa", {}).get("apply_mode") != "decision_tokens"
        or tasks.get("imdb", {}).get("apply_mode") != "all"
    ):
        raise SiteProtocolBindingError("Site CAST-SV protocol drift")
    audit = reft.get("audit")
    finalization = audit.get("finalization") if isinstance(audit, Mapping) else None
    if (
        reft.get("rank") != 4
        or reft.get("alpha_grid") != [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        or not isinstance(finalization, Mapping)
        or finalization.get("retrain") is not False
        or finalization.get("optimizer_steps") != 0
    ):
        raise SiteProtocolBindingError("Site TL;DR ReFT/audit protocol drift")
    if set(evaluation) != {"confiqa", "imdb", "tldr"}:
        raise SiteProtocolBindingError("Site evaluator profiles are incomplete")


def load_formal_site_matrix(path: Path = DEFAULT_MATRIX) -> dict[str, Any]:
    """Load only the new formal matrix and reject all legacy references."""

    resolved = path.expanduser().resolve()
    if resolved != DEFAULT_MATRIX.resolve():
        raise SiteProtocolBindingError("Site rows must use the registered formal matrix")
    matrix = _load(resolved)
    if matrix.get("schema_version") != 3:
        raise SiteProtocolBindingError("formal Site matrix schema_version must be 3")
    if matrix.get("protocol_id") != "site_reproduction_20260829_v1":
        raise SiteProtocolBindingError("formal Site protocol identity drift")
    if matrix.get("flow_id") != FLOW_ID:
        raise SiteProtocolBindingError("formal Site flow identity drift")
    if matrix.get("model_registry") != str(DEFAULT_MODEL_REGISTRY).replace("\\", "/"):
        raise SiteProtocolBindingError("formal Site model registry is invalid")
    if any("legacy" in str(value).lower() for value in matrix.values()):
        raise SiteProtocolBindingError("formal Site matrix may not reference legacy artifacts")
    _validate_data_provenance(matrix)
    _validate_randomness(matrix)
    _validate_scientific_contract(matrix)
    evaluation = matrix["protocol"]["evaluation_profiles"]
    try:
        imdb_model = resolve_registered_evaluator_model(
            evaluation["imdb"]["scorer"]["model_registry_id"],
            registry_path=matrix["model_registry"],
        )
        tldr_model = resolve_registered_evaluator_model(
            evaluation["tldr"]["judge"]["model_registry_id"],
            registry_path=matrix["model_registry"],
        )
    except (KeyError, TypeError, ModelRegistryError) as exc:
        raise SiteProtocolBindingError("Site evaluator model registration is invalid") from exc
    evaluation["imdb"]["scorer"].update(imdb_model)
    evaluation["tldr"]["judge"].update(tldr_model)
    evaluation["tldr"]["judge"]["request"]["base_url"] = tldr_model["base_url"]
    protocol = matrix["protocol"]
    cells = matrix.get("experiment_cells")
    if not isinstance(cells, list) or len(cells) != 10:
        raise SiteProtocolBindingError("formal Site matrix must expose ten model/stratum cells")
    seen: set[tuple[str, str, str]] = set()
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise SiteProtocolBindingError("Site experiment cell must be an object")
        key = (cell.get("dataset"), cell.get("subset"), cell.get("model_family"))
        if not all(isinstance(item, str) and item for item in key):
            raise SiteProtocolBindingError("Site cell identity is incomplete")
        if key in seen:
            raise SiteProtocolBindingError(f"duplicate Site cell: {key}")
        seen.add(key)
        if cell.get("intervention_family") not in {"sv", "reft"}:
            raise SiteProtocolBindingError("Site intervention family must be sv or reft")
        if not isinstance(cell.get("model_registry_id"), str) or not isinstance(cell.get("scorer"), str):
            raise SiteProtocolBindingError("Site cell model/scorer contract is incomplete")
        try:
            resolve_registered_model(
                cell["model_registry_id"], registry_path=matrix["model_registry"]
            )
        except ModelRegistryError as exc:
            raise SiteProtocolBindingError(
                f"Site cell model registration is invalid: {cell['cell_id']}"
            ) from exc
        allowed_subsets = protocol["data"]["construction_profiles"][cell["dataset"]]["subsets"]
        if cell.get("subset") not in allowed_subsets:
            raise SiteProtocolBindingError("Site cell subset is outside its data profile")
    expected_cells = {
        ("confiqa", subset, model)
        for subset in ("qa", "mr", "mc")
        for model in ("llama3_8b", "qwen25_14b")
    } | {
        ("imdb", "sentiment", "ma921_gpt2_large_sft"),
        ("imdb", "sentiment", "qwen25_14b"),
        ("tldr", "summary_preference", "gptj_6b"),
        ("tldr", "summary_preference", "qwen25_14b"),
    }
    if seen != expected_cells:
        raise SiteProtocolBindingError("formal Site model/stratum coverage drift")
    selectors = matrix.get("selector_contract")
    if not isinstance(selectors, Mapping):
        raise SiteProtocolBindingError("Site selector contract is missing")
    if tuple(selectors.get("selectors", ())) != tuple(POSITION_METHOD_BY_SELECTOR):
        raise SiteProtocolBindingError("Site selector order drift")
    seeds = selectors.get("random_seeds")
    if not isinstance(seeds, list) or len(seeds) != 3 or not all(isinstance(seed, int) for seed in seeds):
        raise SiteProtocolBindingError("Site Random seed contract must contain three integers")
    evidence_reuse = matrix.get("evidence_reuse", {})
    if not isinstance(evidence_reuse, Mapping):
        raise SiteProtocolBindingError("Site evidence_reuse must be an object")
    for job_id, spec in evidence_reuse.items():
        if not isinstance(job_id, str) or not isinstance(spec, Mapping):
            raise SiteProtocolBindingError("Site evidence_reuse entries are invalid")
        if spec.get("reuse_through") not in {"position", "bank", "alpha", "test"}:
            raise SiteProtocolBindingError("Site evidence reuse level is invalid")
        evidence_id = spec.get("evidence_id")
        manifest = spec.get("manifest")
        if (
            not isinstance(evidence_id, str)
            or not evidence_id
            or manifest != f"registry://evidence/{evidence_id}"
        ):
            raise SiteProtocolBindingError(
                "Site evidence reuse must bind evidence/<evidence_id> through the machine registry"
            )
    return matrix


def _snapshot(matrix: Mapping[str, Any], cell: Mapping[str, Any]) -> dict[str, Any]:
    protocol = matrix.get("protocol")
    if not isinstance(protocol, Mapping):
        raise SiteProtocolBindingError("formal Site protocol section is missing")
    snapshot = json.loads(json.dumps(protocol, ensure_ascii=False))
    snapshot["cell"] = json.loads(json.dumps(dict(cell), ensure_ascii=False))
    return snapshot


def _bind_rows(matrix: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    selectors = matrix["selector_contract"]
    random_seeds = selectors["random_seeds"]
    for cell in matrix["experiment_cells"]:
        dataset = str(cell["dataset"])
        subset = str(cell["subset"])
        model_family = str(cell["model_family"])
        model = resolve_registered_model(
            cell["model_registry_id"], registry_path=matrix["model_registry"]
        )
        snapshot = _snapshot(matrix, cell)
        for selector, position_method in POSITION_METHOD_BY_SELECTOR.items():
            seeds = random_seeds if selector == "random" else [selectors["primary_seed"]]
            for seed in seeds:
                replicate = f"seed_{seed}" if selector == "random" else "primary"
                job_id = f"{dataset}__{subset}__{model_family}__{selector}__{replicate}"
                row = {
                    "job_id": job_id,
                    "dataset": dataset,
                    "subset": subset,
                    "model_family": model_family,
                    "model": model["id"],
                    "model_registry_id": model["model_registry_id"],
                    "model_checkpoint": model["checkpoint"],
                    "model_revision": model["revision"],
                    "model_source_url": model["source_url"],
                    "model_revision_url": model["revision_url"],
                    "use_chat_template": model["use_chat_template"],
                    "model_architecture": json.loads(
                        json.dumps(model["architecture"], ensure_ascii=False)
                    ),
                    "scorer": cell["scorer"],
                    "intervention_family": cell["intervention_family"],
                    "selector": selector,
                    "position_method": position_method,
                    "replicate": replicate,
                    "seed": seed,
                    "flow_id": FLOW_ID,
                    "protocol_snapshot": snapshot,
                    "protocol": {
                        "id": matrix["protocol_id"],
                        "version": matrix["protocol_version"],
                        "path": "configs/site/site_cell_matrix_20260829_v1.json",
                    },
                }
                reuse = matrix.get("evidence_reuse", {}).get(job_id)
                if reuse is not None:
                    row["evidence_reuse"] = json.loads(
                        json.dumps(dict(reuse), ensure_ascii=False)
                    )
                yield row


def _validated_bound_rows(matrix: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = list(_bind_rows(matrix))
    known = {row["job_id"] for row in rows}
    unknown = set(matrix.get("evidence_reuse", {})) - known
    if unknown:
        raise SiteProtocolBindingError(
            f"Site evidence_reuse names an unknown job: {sorted(unknown)[0]}"
        )
    return rows


def iter_bound_site_jobs(matrix_path: Path = DEFAULT_MATRIX) -> Iterator[dict[str, Any]]:
    """Yield all 60 formal jobs from the one matrix definition."""

    matrix = load_formal_site_matrix(matrix_path)
    yield from _validated_bound_rows(matrix)


def resolve_bound_site_job(job_id: str, *, matrix_path: Path = DEFAULT_MATRIX) -> dict[str, Any]:
    if not isinstance(job_id, str) or not job_id:
        raise SiteProtocolBindingError("job_id must be non-empty")
    matches = [row for row in iter_bound_site_jobs(matrix_path) if row["job_id"] == job_id]
    if len(matches) != 1:
        raise SiteProtocolBindingError(f"unknown or ambiguous formal Site job: {job_id}")
    return matches[0]


def bound_site_matrix(matrix_path: Path = DEFAULT_MATRIX) -> dict[str, Any]:
    matrix = load_formal_site_matrix(matrix_path)
    rows = _validated_bound_rows(matrix)
    return {
        "schema_version": 1,
        "row_source": "site_formal_matrix_v1",
        "flow_id": FLOW_ID,
        "protocol": {
            "id": matrix["protocol_id"],
            "version": matrix["protocol_version"],
            "path": "configs/site/site_cell_matrix_20260829_v1.json",
        },
        "rows": rows,
    }
