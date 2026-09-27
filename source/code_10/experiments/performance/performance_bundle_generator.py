"""Build one Performance bundle by composing the formal Site functions."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

from experiments.performance.performance_protocol_binding import (
    DEFAULT_PROTOCOL,
    resolve_performance_cell,
)
from experiments.site.site_bundle_generator import (
    _alpha_contract as site_alpha_contract,
    _bank_stage,
    _binding,
    _data_stage,
    _evaluation_stage,
    _generation_stage,
    _imdb_full_curve_flow,
    _method as site_cast_method,
    _portable_protocol_value,
    _position_control as site_position_control,
    _position_stages,
    _roles,
    _selected_alpha_flow,
    _source_bindings,
    _stage,
    _tldr_audit_flow,
    _write_exclusive,
)


class PerformanceBundleError(ValueError):
    """Raised when one formal Performance cell cannot form a bundle."""


def _model(row: Mapping[str, Any], role: str) -> dict[str, Any]:
    raw = copy.deepcopy(row["models"][role])
    if raw.get("revision") is None:
        raise PerformanceBundleError(f"{role} model revision is unresolved")
    raw["local_path"] = f"registry://models/{raw['model_registry_id']}"
    return raw


def _site_like_row(row: Mapping[str, Any], *, model_role: str) -> dict[str, Any]:
    model = _model(row, model_role)
    protocol = row["site_protocol"]["protocol"]
    adapted = {
        "job_id": row["job_id"],
        "dataset": row["dataset"],
        "subset": "joint" if row["dataset"] == "confiqa" else (
            "sentiment" if row["dataset"] == "imdb" else "summary_preference"
        ),
        "position_method": "rcm_zero",
        "selector": "rcm_zero_signed",
        "seed": 42,
        "model_family": f"performance/{row['dataset']}/{row['model_family']}/{model_role}",
        "model_registry_id": model["model_registry_id"],
        "model_checkpoint": model["checkpoint"],
        "model_revision": model["revision"],
        "model_source_url": model["source_url"],
        "model_revision_url": model["revision_url"],
        "use_chat_template": model["use_chat_template"],
        "model_architecture": model["architecture"],
        "intervention_family": row["baseline"].get("family", "sv"),
        "scorer": row["scorer"],
        "protocol": {
            "id": row["protocol"]["id"],
            "path": row["protocol"]["path"],
        },
        "protocol_snapshot": protocol,
    }
    if "position_import" in row:
        adapted["position_import"] = copy.deepcopy(row["position_import"])
    return adapted


def _sources(row: Mapping[str, Any]) -> dict[str, Any]:
    dataset = row["dataset"]
    if dataset == "confiqa":
        merged: dict[str, Any] = {}
        for subset in ("qa", "mr", "mc"):
            merged.update(_source_bindings(dataset, subset, row["model_family"]))
        return merged
    subset = "sentiment" if dataset == "imdb" else "summary_preference"
    return _source_bindings(dataset, subset, row["model_family"])


def _performance_roles(
    row: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, str]]:
    roles, outputs = _roles(row["dataset"])
    baseline = row["baseline"]
    if (
        row["dataset"] == "tldr"
        and baseline.get("kind") == "cast"
        and baseline.get("family") == "reft"
    ):
        roles["head_audit"] = {"purpose": "calibration"}
        outputs["audit_manifest"] = "head_audit"
    return roles, outputs


def _performance_construction(
    row: Mapping[str, Any], profile: str
) -> tuple[dict[str, Any], dict[str, Any] | None, str]:
    dataset = row["dataset"]
    registered = row.get("dataset_protocol")
    if dataset == "confiqa":
        if not isinstance(registered, Mapping):
            raise PerformanceBundleError("ConFiQA Performance data protocol is missing")
        joint = copy.deepcopy(registered["joint_construction"])
        profile_config = copy.deepcopy(joint.pop("profiles")[profile])
        joint["test"]["target_valid_rows"] = profile_config["total_role_counts"]["test"]
        joint["roles"] = {
            role: {"target_count": count}
            for role, count in profile_config["total_role_counts"].items()
        }
        return (
            joint,
            copy.deepcopy(profile_config["role_counts_by_subset"]),
            registered["protocol_id"],
        )
    if dataset == "imdb":
        if not isinstance(registered, Mapping):
            raise PerformanceBundleError("IMDb Performance data protocol is missing")
        scaled = (
            copy.deepcopy(registered["formal_scaled_gpu_counts"])
            if profile == "formal_scaled_gpu"
            else None
        )
        return copy.deepcopy(registered["construction"]), scaled, registered["protocol_id"]
    site = row["site_protocol"]
    construction = copy.deepcopy(
        site["protocol"]["data"]["construction_profiles"][dataset]
    )
    scaled = (
        copy.deepcopy(
            site["protocol"]["data"]["scaled_validation"]["role_counts_by_dataset"][dataset]
        )
        if profile == "formal_scaled_gpu"
        else None
    )
    return construction, scaled, site["protocol_id"]

def _data(row: Mapping[str, Any], profile: str) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset = row["dataset"]
    site_data = row["site_protocol"]["protocol"]["data"]
    roles, role_outputs = _performance_roles(row)
    construction, scaled, data_protocol_id = _performance_construction(row, profile)
    protocol = {"id": row["protocol"]["id"], "path": row["protocol"]["path"]}
    subset = "joint" if dataset == "confiqa" else (
        "sentiment" if dataset == "imdb" else "summary_preference"
    )
    value: dict[str, Any] = {
        "backend_binding": _binding(f"site_data_{dataset}"),
        "protocol": protocol,
        "shared_artifact_key": (
            f"performance_data__{data_protocol_id}__{dataset}__"
            f"{subset}__{row['model_family']}__{profile}"
        ),
        "provenance": site_data["provenance"]["datasets"][dataset],
        "construction": construction,
        "prompt_registry": site_data["prompt_registry"],
        "subset": subset,
        "model_family": row["model_family"],
        "roles": roles,
        "role_outputs": role_outputs,
        "sources": _sources(row),
    }
    if dataset == "confiqa":
        value.update(
            {
                "subsets": ["qa", "mr", "mc"],
                "role_counts_by_subset": {
                    subset_name: {role: count for role, count in counts.items() if role in roles}
                    for subset_name, counts in scaled.items()
                },
                "subset_sources": {
                    "qa": "confiqa_qa",
                    "mr": "confiqa_mr",
                    "mc": "confiqa_mc",
                },
            }
        )
    if dataset == "imdb":
        scorer = row["site_protocol"]["protocol"]["evaluation_profiles"]["imdb"]["scorer"]
        value["runtime"] = {
            "model": _model(
                row,
                row["baseline"].get("train_base", "sft"),
            ),
            "sentiment_scorer": {
                "local_path": "registry://models/siebert_sentiment",
                "revision": scorer["revision"],
                "reward_batch_size": scorer["reward_batch_size"],
            },
        }
    if profile == "formal_scaled_gpu":
        value["execution_profile"] = profile
        if dataset != "confiqa" and isinstance(scaled, Mapping):
            value["scaled_counts"] = {
                role: int(count) for role, count in scaled.items() if role in roles
            }
    return value, role_outputs


def _position(row: Mapping[str, Any], profile: str) -> dict[str, Any]:
    baseline = row["baseline"]
    train_role = baseline["train_base"]
    control = site_position_control(
        _site_like_row(row, model_role=train_role), profile
    )
    if baseline.get("family") == "reft" and row["dataset"] == "tldr":
        audit = row["performance_protocol"]["reft_audit_contract"]
        pool = audit["candidate_pool"]
        quotas = {
            "target_support": int(pool["target_support"]),
            "competitor_support": int(pool["competitor_support"]),
        }
        control["expected_count"] = int(pool["count"])
        control["selection"] = {
            "mode": "grouped_top_k",
            "score_field": "ranking_score",
            "order": "descending",
            "tie_break_fields": ["layer_idx", "head_idx", "component_id"],
            "quotas": quotas,
        }
        control["composition"] = {
            "mode": "signed_banks",
            "quotas": quotas,
            "bank_order": ["target_support", "competitor_support"],
            "merge": {
                "operator": "sum",
                "alpha_policy": "shared",
                "retrain_after_merge": False,
            },
        }
        control["scan_config"]["downstream_quota"][row["dataset"]] = quotas
        search = control["method_config"]["baseline"]["parameters"]["position_search"]
        search["positive_count"] = quotas["target_support"]
        search["negative_count"] = quotas["competitor_support"]
    side = baseline["position_side"]
    if side in {"target_support", "competitor_support"}:
        quota = control["composition"]["quotas"][side]
        control["expected_count"] = quota
        control["selection"] = {
            "mode": "grouped_top_k",
            "score_field": "ranking_score",
            "order": "descending",
            "tie_break_fields": ["layer_idx", "head_idx", "component_id"],
            "quotas": {side: quota},
        }
        control["composition"] = {
            "mode": "signed_banks",
            "quotas": {side: quota},
            "bank_order": [side],
            "merge": {
                "operator": "sum",
                "alpha_policy": "shared",
                "retrain_after_merge": False,
            },
        }
    return {
        "bundle_id": f"performance__{row['job_id']}",
        "plan_type": "position",
        "plan_id": "rcm_zero",
        "position_control": control,
    }


def _cast_method(row: Mapping[str, Any], profile: str) -> dict[str, Any]:
    baseline = row["baseline"]
    train_role = baseline["train_base"]
    inference_role = baseline["inference_base"]
    fake = _site_like_row(row, model_role=train_role)
    method = site_cast_method(fake, profile)
    parameters = method["baseline"]["parameters"]
    train_model = _model(row, train_role)
    inference_model = _model(row, inference_role)
    parameters["model"] = {"train": train_model, "inference": inference_model}
    if row["dataset"] == "imdb":
        parameters["model"]["reference"] = _model(row, "sft")
    if baseline["transfer"] == "controller_transfer":
        parameters["transfer"] = {
            "mode": "controller_transfer",
            "compatibility": {
                "component_mapping": "identity_by_layer_and_head",
                "architecture_match": True,
            },
        }
    else:
        parameters["transfer"] = {"mode": "none"}
    timing = row["timing"]
    parameters["controller"]["training_timings"] = [timing["training"]]
    parameters["controller"]["inference_timings"] = [timing["inference"]]
    parameters["controller"]["family"] = baseline["family"]
    if baseline["family"] == "reft":
        parameters["controller"].update({"rank": 4, "reft_mode": "low_rank"})
        parameters["hyperparameters"].update({"rank": 4, "reft_mode": "low_rank"})
    if baseline["family"] == "reft" and row["dataset"] == "tldr":
        audit_contract = row["performance_protocol"]["reft_audit_contract"]
        dataset_audit = audit_contract["by_dataset"][row["dataset"]]
        keep_values = [
            value
            for value in audit_contract["decision_values"]
            if value.startswith("keep_")
        ]
        parameters["audit"] = {
            "phase": "post_bank_training",
            "unit": "component_payload",
            "expected_candidate_count": int(audit_contract["candidate_pool"]["count"]),
            "expected_final_count": int(audit_contract["final_count"]),
            "audit_data_manifest": {"artifact": "artifact://data/audit_manifest"},
            "decision": {
                "decision_field": "decision",
                "keep_values": keep_values,
                "require_complete_coverage": True,
            },
            "finalization": {"mode": "reuse_component_payloads"},
            "locked_audit": {
                **copy.deepcopy(audit_contract),
                **copy.deepcopy(dataset_audit),
                "blinding_seed": 20260723,
            },
        }
        parameters["audit_gate"] = {
            "gate_id": "performance_reft_single_head_audit_v1",
            "decision_schema": "performance_reft_single_head_audit_v1_decision",
        }
        parameters["bank_options"]["require_component_payloads"] = True
    parameters["hyperparameters"].update(
        {
            "performance_baseline_id": row["baseline_id"],
            "position_side": baseline["position_side"],
            "training_model_role": train_role,
            "inference_model_role": inference_role,
        }
    )
    method.update(
        {
            "bundle_id": f"performance__{row['job_id']}",
            "plan_id": "cast",
        }
    )
    return method


def _direct_method(row: Mapping[str, Any]) -> dict[str, Any]:
    role = row["baseline"]["inference_base"]
    models = {"inference": _model(row, role)}
    if row["dataset"] == "imdb":
        models["reference"] = _model(row, "sft")
    return {
        "bundle_id": f"performance__{row['job_id']}",
        "plan_type": "method",
        "plan_id": "direct_policy",
        "baseline": {
            "method": "direct_policy",
            "parameters": {
                "model": models,
                "data": {
                    "inference_manifest": {
                        "artifact": "artifact://data/test_manifest"
                    }
                },
                "inference": {"policy_role": role},
                "hyperparameters": {
                    "performance_baseline_id": row["baseline_id"]
                },
                "generation_backend_binding": _binding("cast_generation"),
            },
        },
    }


def _selection(row: Mapping[str, Any]) -> dict[str, Any]:
    fake = _site_like_row(
        row,
        model_role=row["baseline"].get("train_base", row["baseline"].get("inference_base", "sft")),
    )
    contract = site_alpha_contract(fake)
    if row["baseline"]["kind"] == "direct_policy":
        locked = copy.deepcopy(contract["test_generation"]["locked_generation"])
        locked.pop("apply_mode", None)
        direct = {
            "test_policy": {"mode": "direct_policy"},
            "test_generation": {"stage": "test", "locked_generation": locked},
        }
        if row["dataset"] == "imdb":
            direct["test_generation"]["policy_audit"] = {
                "enabled": True,
                "reference_policy": "registered_sft",
                "reference_model": _model(row, "sft"),
                "metrics": [
                    "sampled_sequence_logprob_ratio",
                    "token_kl_audit",
                ],
            }
        return direct
    timing_name = row["timing"]["inference"]["name"]
    for key in ("alpha_generation", "test_generation"):
        if key in contract and isinstance(contract[key].get("locked_generation"), dict):
            contract[key]["locked_generation"]["apply_mode"] = timing_name
    if row["dataset"] == "imdb":
        audit = {
            "reference_policy": "registered_sft",
            "reference_model": _model(row, "sft"),
        }
        if row["baseline"]["inference_base"] == "dpo":
            audit["additional_incremental_reference"] = (
                "inference_checkpoint_without_controller"
            )
        contract["test_generation"]["policy_audit"].update(audit)
    return contract


def _direct_flow() -> list[dict[str, Any]]:
    gate = _stage(
        "test_gate",
        "test_gate",
        {
            "test_role_manifest": "artifact://data/test_manifest",
            "test_policy": "config://shared.selection_calibration_test.test_policy",
        },
        {"test_authorization": "test/test_authorization.json"},
    )
    return [
        gate,
        _generation_stage(
            "test_generate",
            controller=None,
            role=None,
            authorization="artifact://test_gate/test_authorization",
            config_ref="config://shared.selection_calibration_test.test_generation",
            prefix="test",
            trace_kind="generation_test",
        ),
        _evaluation_stage(
            "test_evaluate",
            generation_stage="test_generate",
            role_output=None,
            authorization="artifact://test_gate/test_authorization",
            prefix="test",
            config_ref="config://shared.evaluation",
        ),
    ]


def build_performance_job_bundle(
    job_id: str,
    *,
    output_dir: Path,
    matrix_path: Path = DEFAULT_PROTOCOL,
    execution_profile: str = "formal",
) -> dict[str, Any]:
    if execution_profile not in {"formal", "formal_scaled_gpu"}:
        raise PerformanceBundleError("unknown Performance execution profile")
    row = resolve_performance_cell(job_id, protocol_path=matrix_path)
    dataset = row["dataset"]
    data, role_outputs = _data(row, execution_profile)
    method_kind = row["baseline"]["kind"]
    flow = [_data_stage(role_outputs)]
    if method_kind == "direct_policy":
        flow.extend(_direct_flow())
        method = _direct_method(row)
        plans_position: dict[str, str] = {}
        position = None
    else:
        is_reft = row["baseline"].get("family") == "reft"
        requires_audit = is_reft and dataset == "tldr"
        flow.extend([*_position_stages(), _bank_stage(audited=requires_audit)])
        controller_ref = "artifact://bank_train/controller_manifest"
        if requires_audit:
            flow.extend(_tldr_audit_flow())
            controller_ref = "artifact://bank_finalize/controller_manifest"
        if dataset in {"confiqa", "tldr"}:
            flow.extend(_selected_alpha_flow(dataset, controller_ref))
        else:
            flow.extend(_imdb_full_curve_flow(controller_ref))
        method = _cast_method(row, execution_profile)
        position = _position(row, execution_profile)
        plans_position = {"rcm_zero": "position.rcm_zero.json"}
    inference_role = row["baseline"].get("inference_base", "sft")
    train_role = row["baseline"].get("train_base", inference_role)
    evaluation_profile = row["site_protocol"]["protocol"]["evaluation_profiles"][dataset]
    shared = {
        "data": data,
        "models": {
            "train": _model(row, train_role),
            "inference": _model(row, inference_role),
        },
        "flow": flow,
        "selection_calibration_test": _selection(row),
        "evaluation": {
            "scorer": row["scorer"],
            "backend_binding": _binding(f"site_evaluator_{dataset}"),
            "profile": evaluation_profile,
            "alpha": {
                "dataset": dataset,
                "subset": data["subset"],
                "stage": "alpha_dev",
            },
            "test": {
                "dataset": dataset,
                "subset": data["subset"],
                "stage": "test",
            },
        },
        "execution": {
            "profile": execution_profile,
            "publishable": execution_profile == "formal",
            "evidence_root": "registry://outputs/experiments",
            "trajectory": {"enabled": True, "coverage": "all_samples"},
            "serial_only": True,
        },
    }
    if "evidence_reuse" in row:
        shared["execution"]["evidence_reuse"] = _portable_protocol_value(
            row["evidence_reuse"]
        )
    if dataset == "imdb":
        shared["evaluation"]["runtime"] = {
            "sentiment_model_path": "registry://models/siebert_sentiment"
        }
    bundle_id = f"performance__{job_id}"
    master = {
        "schema_version": 1,
        "bundle_id": bundle_id,
        "experiment": "performance",
        "dataset": dataset,
        "model_family": row["model_family"],
        "protocol": {"id": row["protocol"]["id"], "path": row["protocol"]["path"]},
        "shared": shared,
        "plans": {
            "method": {method["plan_id"]: f"method.{method['plan_id']}.json"},
            "position": plans_position,
        },
        "matrix": [
            {
                "cell_id": job_id,
                "method_ref": method["plan_id"],
                "position_ref": "rcm_zero" if position is not None else None,
                "locked_job": {
                    "baseline_id": row["baseline_id"],
                    "status": row["status"],
                },
            }
        ],
    }
    output_dir = output_dir.expanduser().resolve()
    _write_exclusive(output_dir / "bundle.json", master)
    _write_exclusive(output_dir / f"method.{method['plan_id']}.json", method)
    if position is not None:
        _write_exclusive(output_dir / "position.rcm_zero.json", position)
    return {
        "schema_version": 1,
        "status": "derived",
        "job_id": job_id,
        "execution_profile": execution_profile,
        "bundle_directory": str(output_dir),
        "files": [path.name for path in sorted(output_dir.glob("*.json"))],
    }
