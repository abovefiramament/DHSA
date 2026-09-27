"""Bind the formal non-Tulu Performance protocol to explicit matrix cells."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from experiments.shared.model_registry import (
    DEFAULT_MODEL_REGISTRY,
    ModelRegistryError,
    resolve_registered_model,
)
from experiments.site.site_protocol_binding import load_formal_site_matrix


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = (
    REPOSITORY_ROOT
    / "configs"
    / "performance"
    / "performance_protocol_20260831_v1.json"
)
class PerformanceProtocolError(ValueError):
    """Raised when one declared Performance condition is ambiguous."""


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PerformanceProtocolError(f"cannot load Performance protocol: {path}") from exc
    if not isinstance(value, dict):
        raise PerformanceProtocolError("Performance protocol must contain one object")
    return value


def _site_authority(protocol: Mapping[str, Any]) -> dict[str, Any]:
    authority = protocol.get("site_data_authority")
    if not isinstance(authority, Mapping):
        raise PerformanceProtocolError("site_data_authority is missing")
    relative = authority.get("path")
    if not isinstance(relative, str) or not relative:
        raise PerformanceProtocolError("site_data_authority.path is missing")
    path = (REPOSITORY_ROOT / relative).resolve()
    try:
        path.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as exc:
        raise PerformanceProtocolError("site_data_authority.path escapes repository") from exc
    site = load_formal_site_matrix(path)
    if site.get("protocol_id") != authority.get("protocol_id"):
        raise PerformanceProtocolError("Site authority protocol_id drift")
    return site


def _dataset_protocol(
    protocol: Mapping[str, Any], dataset: str
) -> dict[str, Any] | None:
    registrations = protocol.get("dataset_protocols")
    if not isinstance(registrations, Mapping):
        raise PerformanceProtocolError("dataset_protocols is missing")
    registration = registrations.get(dataset)
    if not isinstance(registration, Mapping):
        raise PerformanceProtocolError(f"{dataset} dataset protocol is missing")
    mode = registration.get("mode")
    if mode == "reuse_site_dataset_protocol":
        return None
    if mode != "performance_specific":
        raise PerformanceProtocolError(f"{dataset} dataset protocol mode is invalid")
    relative = registration.get("path")
    if not isinstance(relative, str) or not relative:
        raise PerformanceProtocolError(f"{dataset} dataset protocol path is missing")
    path = (REPOSITORY_ROOT / relative).resolve()
    try:
        path.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as exc:
        raise PerformanceProtocolError(
            f"{dataset} dataset protocol path escapes repository"
        ) from exc
    value = _load(path)
    if (
        value.get("protocol_id") != registration.get("protocol_id")
        or value.get("status") != "canonical_formal_data_protocol"
        or value.get("dataset") != dataset
    ):
        raise PerformanceProtocolError(f"{dataset} dataset protocol identity drift")
    return value

def load_performance_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol = _load(path)
    if (
        protocol.get("schema_version") != 1
        or protocol.get("status") != "canonical_formal_matrix"
        or protocol.get("flow_id") != "performance_cast_site_aligned_flow_v1"
    ):
        raise PerformanceProtocolError("Performance protocol identity is invalid")
    if protocol.get("model_registry") != str(DEFAULT_MODEL_REGISTRY).replace("\\", "/"):
        raise PerformanceProtocolError("Performance model registry is invalid")
    scope = protocol.get("scope")
    if not isinstance(scope, Mapping) or scope.get("datasets") != ["confiqa", "imdb", "tldr"]:
        raise PerformanceProtocolError("Performance dataset scope drift")
    if scope.get("external_baselines_deferred") != ["bipo", "loreft"]:
        raise PerformanceProtocolError("BiPO/LoReFT must remain explicitly deferred")
    timing = protocol.get("timing_contract")
    rows = timing.get("formal_rows") if isinstance(timing, Mapping) else None
    if not isinstance(rows, list) or [row.get("id") for row in rows] != ["all", "prefill"]:
        raise PerformanceProtocolError("formal timing rows must be all and prefill")
    for row in rows:
        if row.get("training") != row.get("inference"):
            raise PerformanceProtocolError("formal timing rows must be matched")
    model_pairs = protocol.get("model_pairs")
    if not isinstance(model_pairs, Mapping) or set(model_pairs) != {"confiqa", "imdb", "tldr"}:
        raise PerformanceProtocolError("model_pairs must cover three datasets")
    for dataset, pairs in model_pairs.items():
        if not isinstance(pairs, list) or not pairs:
            raise PerformanceProtocolError(f"{dataset} model_pairs are missing")
        for pair in pairs:
            if not isinstance(pair, Mapping) or pair.get("status") not in {"ready", "pending_dpo_revision"}:
                raise PerformanceProtocolError(f"{dataset} model pair status is invalid")
            resolved_models: dict[str, dict[str, Any]] = {}
            for role in ("sft", "dpo"):
                registry_id = pair.get(f"{role}_model_registry_id")
                if not isinstance(registry_id, str) or not registry_id:
                    raise PerformanceProtocolError(f"{dataset} {role} model registration is missing")
                pending = pair.get("status") == "pending_dpo_revision" and role == "dpo"
                try:
                    resolved_models[role] = resolve_registered_model(
                        registry_id,
                        registry_path=protocol["model_registry"],
                        allow_pending=pending,
                    )
                except ModelRegistryError as exc:
                    raise PerformanceProtocolError(
                        f"{dataset} {role} model registration is invalid"
                    ) from exc
                if pending != (resolved_models[role]["status"] == "pending_revision"):
                    raise PerformanceProtocolError(f"{dataset} pending model status drifts")
            if resolved_models["sft"]["architecture"] != resolved_models["dpo"]["architecture"]:
                raise PerformanceProtocolError(f"{dataset} transfer architecture drift")
    baselines = protocol.get("baseline_rows")
    if not isinstance(baselines, list) or len(baselines) != 16:
        raise PerformanceProtocolError("Performance must declare exactly 16 baseline row types")
    ids = [row.get("id") for row in baselines]
    if len(ids) != len(set(ids)) or not all(isinstance(value, str) and value for value in ids):
        raise PerformanceProtocolError("baseline row IDs are invalid")
    for row in baselines:
        datasets = row.get("datasets")
        if not isinstance(datasets, list) or not datasets:
            raise PerformanceProtocolError("baseline row datasets are missing")
        if row.get("kind") == "cast":
            if row.get("timing") not in {"all", "prefill"}:
                raise PerformanceProtocolError("CAST row timing is invalid")
            if "tldr" in datasets and row.get("family") != "reft":
                raise PerformanceProtocolError("TL;DR Performance is ReFT-only")
        elif row.get("kind") != "direct_policy":
            raise PerformanceProtocolError("unknown Performance baseline kind")
    operator = protocol.get("operator_contract")
    if not isinstance(operator, Mapping) or operator.get("reft_rank") != 4:
        raise PerformanceProtocolError("Performance CAST-ReFT must use Site rank 4")
    registrations = protocol.get("dataset_protocols")
    if not isinstance(registrations, Mapping) or set(registrations) != {
        "confiqa", "imdb", "tldr"
    }:
        raise PerformanceProtocolError("dataset protocols must cover three datasets")
    confiqa_data = _dataset_protocol(protocol, "confiqa")
    imdb_data = _dataset_protocol(protocol, "imdb")
    if _dataset_protocol(protocol, "tldr") is not None:
        raise PerformanceProtocolError("TLDR must reuse the Site data protocol")
    confiqa_profiles = confiqa_data.get("joint_construction", {}).get("profiles", {})
    formal_confiqa = confiqa_profiles.get("formal", {})
    totals = formal_confiqa.get("total_role_counts", {})
    by_subset = formal_confiqa.get("role_counts_by_subset", {})
    if totals != {
        "selector": 300,
        "training": 240,
        "validation": 60,
        "alpha_dev": 120,
        "test": 2048,
    }:
        raise PerformanceProtocolError("ConFiQA joint total role counts drift")
    if set(by_subset) != {"qa", "mr", "mc"}:
        raise PerformanceProtocolError("ConFiQA joint subset allocation is incomplete")
    for role, total in totals.items():
        if sum(int(by_subset[subset][role]) for subset in ("qa", "mr", "mc")) != total:
            raise PerformanceProtocolError(f"ConFiQA {role} allocation does not sum")
    imdb_roles = imdb_data.get("construction", {}).get("roles", {})
    if "head_audit" in imdb_roles:
        raise PerformanceProtocolError("IMDb must not expose a head-audit role")
    fixed_k8 = protocol.get("position_contract", {}).get(
        "fixed_k8_without_post_training_audit"
    )
    if (
        not isinstance(fixed_k8, Mapping)
        or fixed_k8.get("datasets") != ["confiqa", "imdb"]
        or fixed_k8.get("families") != ["sv", "reft"]
        or fixed_k8.get("target_support") != 4
        or fixed_k8.get("competitor_support") != 4
        or fixed_k8.get("total") != 8
        or fixed_k8.get("human_audit") is not False
        or fixed_k8.get("post_training_head_reselection") is not False
    ):
        raise PerformanceProtocolError("ConFiQA/IMDb fixed K4+K4 contract drift")
    audit = protocol.get("reft_audit_contract")
    pool = audit.get("candidate_pool") if isinstance(audit, Mapping) else None
    if (
        not isinstance(pool, Mapping)
        or pool.get("count") != 16
        or pool.get("target_support") != 8
        or pool.get("competitor_support") != 8
        or audit.get("final_count") != 8
        or audit.get("final_role_quota") is not None
    ):
        raise PerformanceProtocolError("ReFT K16-to-K8 audit contract drift")
    if set(audit.get("by_dataset", {})) != {"tldr"}:
        raise PerformanceProtocolError("ReFT human audit must be TLDR-only")
    external = protocol.get("external_baseline_search_contract")
    if (
        not isinstance(external, Mapping)
        or external.get("methods") != ["bipo", "loreft"]
        or external.get("retain_native_intervention_geometry") is not True
        or external.get("select_best_validation_configuration") is not True
        or external.get("test_based_selection_forbidden") is not True
        or external.get("componentwise_human_audit") is not False
        or external.get("artificial_search_budget_reduction_to_compensate_for_CAST")
        is not False
    ):
        raise PerformanceProtocolError("external baseline search contract drift")
    alpha = protocol.get("alpha_contract")
    if not isinstance(alpha, Mapping) or not alpha.get("signed_banks_share_one_alpha"):
        raise PerformanceProtocolError("shared alpha contract is missing")
    imports = protocol.get("position_imports", {})
    if not isinstance(imports, Mapping):
        raise PerformanceProtocolError("position_imports must be an object when present")
    for cell_id, spec in imports.items():
        if not isinstance(cell_id, str) or not isinstance(spec, Mapping):
            raise PerformanceProtocolError("position_imports entries are invalid")
        if spec.get("mode") not in {"rcm_scan", "fixed_k8"}:
            raise PerformanceProtocolError("position import mode is invalid")
        manifest = spec.get("manifest")
        if not isinstance(manifest, str) or not manifest.startswith("registry://"):
            raise PerformanceProtocolError(
                "position import manifests must use the machine registry"
            )
    evidence_reuse = protocol.get("evidence_reuse", {})
    if not isinstance(evidence_reuse, Mapping):
        raise PerformanceProtocolError("evidence_reuse must be an object when present")
    for cell_id, spec in evidence_reuse.items():
        if not isinstance(cell_id, str) or not isinstance(spec, Mapping):
            raise PerformanceProtocolError("evidence_reuse entries are invalid")
        if spec.get("reuse_through") not in {"position", "bank", "alpha", "test"}:
            raise PerformanceProtocolError("evidence reuse level is invalid")
        evidence_id = spec.get("evidence_id")
        if (
            not isinstance(evidence_id, str)
            or not evidence_id
            or spec.get("manifest") != f"registry://evidence/{evidence_id}"
        ):
            raise PerformanceProtocolError(
                "evidence reuse must bind evidence/<evidence_id> through the machine registry"
            )
    _site_authority(protocol)
    return protocol


def iter_declared_performance_cells(
    protocol_path: Path = DEFAULT_PROTOCOL,
) -> tuple[dict[str, Any], ...]:
    protocol = load_performance_protocol(protocol_path)
    site = _site_authority(protocol)
    timing_by_id = {
        row["id"]: row for row in protocol["timing_contract"]["formal_rows"]
    }
    scorer_by_dataset = {
        dataset: next(
            cell["scorer"] for cell in site["experiment_cells"]
            if cell["dataset"] == dataset
        )
        for dataset in ("confiqa", "imdb", "tldr")
    }
    cells: list[dict[str, Any]] = []
    for dataset, pairs in protocol["model_pairs"].items():
        for pair in pairs:
            models = {
                role: resolve_registered_model(
                    pair[f"{role}_model_registry_id"],
                    registry_path=protocol["model_registry"],
                    allow_pending=(
                        pair.get("status") == "pending_dpo_revision" and role == "dpo"
                    ),
                )
                for role in ("sft", "dpo")
            }
            for baseline in protocol["baseline_rows"]:
                if dataset not in baseline["datasets"]:
                    continue
                requires_dpo = (
                    baseline.get("inference_base") == "dpo"
                    or baseline.get("train_base") == "dpo"
                )
                runnable = not (
                    requires_dpo and pair.get("status") == "pending_dpo_revision"
                )
                cell = {
                    "job_id": (
                        f"performance__{dataset}__{pair['model_family']}__{baseline['id']}"
                    ),
                    "dataset": dataset,
                    "model_family": pair["model_family"],
                    "baseline_id": baseline["id"],
                    "baseline": dict(baseline),
                    "scorer": scorer_by_dataset[dataset],
                    "models": models,
                    "status": "runnable" if runnable else "pending_dpo_revision",
                    "protocol": {
                        "id": protocol["protocol_id"],
                        "path": str(protocol_path.resolve().relative_to(REPOSITORY_ROOT)),
                    },
                    "performance_protocol": protocol,
                    "dataset_protocol": _dataset_protocol(protocol, dataset),
                    "site_protocol": site,
                }
                if baseline.get("kind") == "cast":
                    cell["timing"] = dict(timing_by_id[baseline["timing"]])
                position_import = protocol.get("position_imports", {}).get(
                    cell["job_id"]
                )
                if position_import is not None:
                    cell["position_import"] = dict(position_import)
                evidence_reuse = protocol.get("evidence_reuse", {}).get(
                    cell["job_id"]
                )
                if evidence_reuse is not None:
                    cell["evidence_reuse"] = dict(evidence_reuse)
                cells.append(cell)
    if len(cells) != 64 or len({cell["job_id"] for cell in cells}) != 64:
        raise PerformanceProtocolError(
            f"formal Performance matrix must contain 64 unique cells, got {len(cells)}"
        )
    unknown_reuse = set(protocol.get("evidence_reuse", {})) - {
        cell["job_id"] for cell in cells
    }
    if unknown_reuse:
        raise PerformanceProtocolError(
            f"evidence_reuse names an unknown Performance job: {sorted(unknown_reuse)[0]}"
        )
    return tuple(cells)


def iter_performance_cells(
    protocol_path: Path = DEFAULT_PROTOCOL,
) -> tuple[dict[str, Any], ...]:
    return tuple(
        cell for cell in iter_declared_performance_cells(protocol_path)
        if cell["status"] == "runnable"
    )


def resolve_performance_cell(
    cell_id: str,
    *,
    protocol_path: Path = DEFAULT_PROTOCOL,
) -> dict[str, Any]:
    matches = [
        cell for cell in iter_declared_performance_cells(protocol_path)
        if cell["job_id"] == cell_id
    ]
    if len(matches) != 1:
        raise PerformanceProtocolError(f"unknown Performance cell: {cell_id}")
    if matches[0]["status"] != "runnable":
        raise PerformanceProtocolError(
            f"Performance cell {cell_id} is pending a verified DPO revision"
        )
    return matches[0]
