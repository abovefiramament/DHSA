"""Derive one portable Site job bundle from the locked matrix row.

Generated bundles are planning evidence.  They contain no copied result values,
machine paths, or hand-maintained scientific defaults: the retained protocol
and immutable job identity remain the authority consumed by typed backends.
"""

from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping

from experiments.site.site_cell_controller import DEFAULT_MATRIX, resolve_site_cell


class SiteBundleError(ValueError):
    """Raised when a locked Site row cannot form one unambiguous bundle."""


_OMIT = object()


def _portable_protocol_value(value: Any) -> Any:
    """Drop only machine-local path leaves from copied scientific sections.

    Dataset/model locations are supplied by typed data adapters and the runtime
    registry.  Keeping their old absolute values in a portable bundle would
    silently reintroduce the legacy machine layout.
    """

    if isinstance(value, str) and (
        PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()
    ):
        return _OMIT
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            portable = _portable_protocol_value(item)
            if portable is not _OMIT:
                result[str(key)] = portable
        return result
    if isinstance(value, list):
        result = []
        for item in value:
            portable = _portable_protocol_value(item)
            if portable is not _OMIT:
                result.append(portable)
        return result
    return value


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise SiteBundleError(f"derived bundle artifact already exists: {path}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(rendered)


def _binding(component_id: str) -> dict[str, Any]:
    if not isinstance(component_id, str) or not component_id:
        raise SiteBundleError("component binding id must be non-empty")
    return {"id": component_id, "version": 1}


def _protocol_reference(row: Mapping[str, Any]) -> dict[str, Any]:
    frozen = row["protocol"]
    value = {
        "id": frozen["id"],
        "path": frozen["path"],
    }
    return value


def _model_config(row: Mapping[str, Any]) -> dict[str, Any]:
    """Expose one resolved public model registration plus its machine path reference."""

    return {
        "model_registry_id": row["model_registry_id"],
        "checkpoint": row["model_checkpoint"],
        "revision": row["model_revision"],
        "source_url": row["model_source_url"],
        "revision_url": row["model_revision_url"],
        "local_path": f"registry://models/{row['model_registry_id']}",
        "use_chat_template": row["use_chat_template"],
        "architecture": row["model_architecture"],
    }


def _roles(dataset: str) -> tuple[dict[str, Any], dict[str, str]]:
    common = {
        "selector": {"purpose": "selection"},
        "training": {"purpose": "training"},
        "validation": {"purpose": "training_validation"},
        "test": {"purpose": "final_test"},
    }
    outputs = {
        "selector_manifest": "selector",
        "training_manifest": "training",
        "validation_manifest": "validation",
        "test_manifest": "test",
    }
    if dataset in {"confiqa", "tldr"}:
        common["alpha_dev"] = {"purpose": "calibration"}
        outputs["alpha_manifest"] = "alpha_dev"
    if dataset == "tldr":
        common["head_audit"] = {"purpose": "calibration"}
        common["reserve"] = {"purpose": "reserve"}
        outputs["audit_manifest"] = "head_audit"
        outputs["reserve_manifest"] = "reserve"
    return common, outputs


def _source_bindings(
    dataset: str,
    subset: str,
    model_family: str,
) -> dict[str, dict[str, Any]]:
    """Declare only the prepared artifacts consumed by formal data roles."""

    if dataset == "confiqa":
        if subset not in {"qa", "mr", "mc"}:
            raise SiteBundleError(f"unknown ConFiQA subset: {subset!r}")
        return {
            f"confiqa_{subset}": {
                "location": f"registry://datasets/confiqa/{subset}",
                "format": "json",
                "source_id": "confiqa_context_dpo_repo",
            }
        }
    if dataset == "imdb":
        return {
            "imdb_train": {
                "location": "registry://datasets/imdb/train_source",
                "format": "jsonl",
                "source_id": "stanfordnlp_imdb_train_mirror",
                "upstream_source_id": "stanfordnlp_imdb_hf",
            },
            "imdb_test": {
                "location": "registry://datasets/imdb/test_source",
                "format": "jsonl",
                "source_id": "stanfordnlp_imdb_test_mirror",
                "upstream_source_id": "stanfordnlp_imdb_hf",
            },
        }
    if dataset == "tldr":
        return {
            "tldr_pairs": {
                "location": "registry://datasets/tldr/preference_pairs",
                "format": "csv",
                "source_id": "tldr_openai_preference_pairs",
                "upstream_source_id": "openai_summarize_from_feedback_upstream",
            },
            "tldr_test_prompts": {
                "location": "registry://datasets/tldr/test_prompts",
                "format": "jsonl",
                "source_id": "tldr_openai_test_prompt_projection",
                "upstream_source_id": "openai_summarize_from_feedback_upstream",
            },
        }
    raise SiteBundleError(f"unknown Site dataset: {dataset!r}")


def _stage(
    stage_id: str,
    handler: str,
    inputs: Mapping[str, str],
    outputs: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "id": stage_id,
        "handler": handler,
        "inputs": dict(inputs),
        "outputs": dict(outputs),
    }


def _data_stage(role_outputs: Mapping[str, str]) -> dict[str, Any]:
    outputs = {"dataset_manifest": "data/dataset_manifest.reference.json"}
    outputs.update(
        {
            name: f"data/roles/{role}.manifest.json"
            for name, role in role_outputs.items()
        }
    )
    return _stage(
        "data",
        "dataset_controller",
        {"data_config": "config://shared.data"},
        outputs,
    )


def _position_stages(
    *,
    shared_scan_ref: str | None = None,
    training_role_ref: str | None = None,
) -> list[dict[str, Any]]:
    search_inputs = {
        "backend_binding": "config://position_control.backend_binding",
        "scanner_binding": "config://position_control.scanner_binding",
        "method_config": "config://position_control.method_config",
        "model_config": "config://shared.models.train",
        "scan_config": "config://position_control.scan_config",
        "selector_data_manifest": "artifact://data/selector_manifest",
        "trajectory_config": "config://shared.execution.trajectory",
    }
    if shared_scan_ref is not None:
        search_inputs["shared_scan"] = shared_scan_ref
    if training_role_ref is not None:
        search_inputs["training_data_manifest"] = training_role_ref
    resolve_inputs = {
        "position_config": "config://position_control",
        "candidate_manifest": "artifact://position_search/candidate_manifest",
        "selector_data_manifest": "artifact://data/selector_manifest",
    }
    if training_role_ref is not None:
        resolve_inputs["training_data_manifest"] = training_role_ref
    return [
        _stage(
            "position_search",
            "position_search",
            search_inputs,
            {
                "candidate_manifest": "position/candidate_manifest.jsonl",
                "backend_execution_manifest": "position/backend_execution_manifest.json",
                "position_search_manifest": "position/position_search_manifest.json",
            },
        ),
        _stage(
            "position_resolve",
            "position_controller",
            resolve_inputs,
            {"position_plan": "position/position_plan.json"},
        ),
        _stage(
            "position_freeze",
            "position_freeze",
            {
                "position_plan": "artifact://position_resolve/position_plan",
                "data_freeze": "artifact://data/dataset_manifest",
                "method_config": "config://position_control.method_config",
            },
            {"position_freeze": "position/position_freeze.json"},
        ),
    ]


def _behavior_advantage(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the protocol-owned target-vs-non-target behavior advantage."""

    profile = row["protocol_snapshot"]["evaluation_profiles"][row["dataset"]]
    advantage = profile.get("behavior_advantage")
    if not isinstance(advantage, Mapping):
        raise SiteBundleError("dataset behavior advantage is missing")
    return dict(advantage)


def _bank_stage(*, audited: bool) -> dict[str, Any]:
    outputs = {
        "bank_plan": "bank/candidate/bank_plan.json",
        "bank_execution_manifest": "bank/candidate/bank_execution_manifest.json",
        "controller_manifest": "bank/candidate/controller_manifest.json",
    }
    if audited:
        outputs["component_index"] = "bank/candidate/component_index.json"
    return _stage(
        "bank_train",
        "bank_execute",
        {
            "position_freeze": "artifact://position_freeze/position_freeze",
            "training_manifest": "artifact://data/training_manifest",
            "validation_manifest": "artifact://data/validation_manifest",
            "backend_binding": "config://baseline.parameters.backend_binding",
            "bank_options": "config://baseline.parameters.bank_options",
        },
        outputs,
    )


def _generation_stage(
    stage_id: str,
    *,
    controller: str | None,
    role: str | None,
    authorization: str | None,
    config_ref: str,
    prefix: str,
    trace_kind: str,
) -> dict[str, Any]:
    inputs = {
        "backend_binding": "config://baseline.parameters.generation_backend_binding",
        "model_config": "config://shared.models.inference",
        "generation_config": config_ref,
        "trajectory_config": "config://shared.execution.trajectory",
    }
    if controller is not None:
        inputs["controller_manifest"] = controller
    if role is not None:
        inputs["data_role_manifest"] = role
    if authorization is not None:
        inputs["test_authorization"] = authorization
    stage = _stage(
        stage_id,
        "generation",
        inputs,
        {
            "predictions": f"{prefix}/predictions.jsonl",
            "backend_execution_manifest": f"{prefix}/backend_execution_manifest.json",
            "generation_manifest": f"{prefix}/generation_manifest.json",
        },
    )
    stage["trace_kind"] = trace_kind
    return stage


def _evaluation_stage(
    stage_id: str,
    *,
    generation_stage: str,
    role_output: str | None,
    authorization: str | None,
    prefix: str,
    config_ref: str,
) -> dict[str, Any]:
    inputs = {
        "predictions": f"artifact://{generation_stage}/predictions",
        "evaluator_binding": "config://shared.evaluation.backend_binding",
        "evaluation_config": config_ref,
    }
    if (role_output is None) == (authorization is None):
        raise SiteBundleError(
            "evaluation stage requires exactly one role output or test authorization"
        )
    if role_output is not None:
        inputs["references"] = f"artifact://data/{role_output}"
    else:
        inputs["test_authorization"] = str(authorization)
    return _stage(
        stage_id,
        "evaluation",
        inputs,
        {
            "per_sample_scores": f"{prefix}/per_sample_scores.jsonl",
            "summary_metrics": f"{prefix}/summary_metrics.json",
            "evaluation_manifest": f"{prefix}/evaluation_manifest.json",
        },
    )


def _selected_alpha_flow(dataset: str, controller_ref: str) -> list[dict[str, Any]]:
    return [
        _generation_stage(
            "alpha_generate",
            controller=controller_ref,
            role="artifact://data/alpha_manifest",
            authorization=None,
            config_ref="config://shared.selection_calibration_test.alpha_generation",
            prefix="alpha",
            trace_kind="audit" if dataset == "tldr" else "generation_test",
        ),
        _evaluation_stage(
            "alpha_evaluate",
            generation_stage="alpha_generate",
            role_output="alpha_manifest",
            authorization=None,
            prefix="alpha",
            config_ref="config://shared.evaluation",
        ),
        _stage(
            "alpha_freeze",
            "alpha_freeze",
            {
                "curve": "artifact://alpha_evaluate/summary_metrics",
                "controller_manifest": controller_ref,
                "calibration_manifest": "artifact://data/alpha_manifest",
                "evaluator_manifest": "artifact://alpha_evaluate/evaluation_manifest",
                "selector_binding": "config://shared.selection_calibration_test.alpha_selector_binding",
                "selection_config": "config://shared.selection_calibration_test.alpha_selection",
            },
            {"alpha_freeze": "alpha/alpha_freeze.json"},
        ),
        _stage(
            "test_gate",
            "test_gate",
            {
                "alpha_freeze": "artifact://alpha_freeze/alpha_freeze",
                "controller_manifest": controller_ref,
                "test_role_manifest": "artifact://data/test_manifest",
            },
            {"test_authorization": "test/test_authorization.json"},
        ),
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


def _imdb_full_curve_flow(controller_ref: str) -> list[dict[str, Any]]:
    return [
        _stage(
            "test_gate",
            "test_gate",
            {
                "test_policy": "config://shared.selection_calibration_test.test_policy",
                "controller_manifest": controller_ref,
                "test_role_manifest": "artifact://data/test_manifest",
            },
            {"test_authorization": "test/test_authorization.json"},
        ),
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


def _tldr_audit_flow() -> list[dict[str, Any]]:
    return [
        _stage(
            "audit_prepare",
            "post_training_audit_prepare",
            {
                "bank_plan": "artifact://bank_train/bank_plan",
                "component_index": "artifact://bank_train/component_index",
                "audit_manifest": "artifact://data/audit_manifest",
                "audit_config": "config://baseline.parameters.audit",
            },
            {"audit_plan": "audit/audit_plan.json"},
        ),
        _stage(
            "audit_execute",
            "post_training_audit_execute",
            {
                "audit_plan": "artifact://audit_prepare/audit_plan",
                "source_bank_plan": "artifact://bank_train/bank_plan",
                "audit_role_manifest": "artifact://data/audit_manifest",
                "bank_backend_binding": "config://baseline.parameters.backend_binding",
                "generation_backend_binding": "config://baseline.parameters.generation_backend_binding",
                "evaluator_binding": "config://shared.evaluation.backend_binding",
                "model_config": "config://shared.models.train",
                "generation_config": "config://shared.selection_calibration_test.alpha_generation",
                "evaluation_config": "config://shared.evaluation",
                "trajectory_config": "config://shared.execution.trajectory",
            },
            {
                "audit_packet": "audit/audit_packet.json",
                "decision_key": "audit/decision_key.json",
            },
        ),
        _stage(
            "audit_gate",
            "manual_audit",
            {
                "candidate_manifest": "artifact://audit_execute/audit_packet",
                "gate_config": "config://baseline.parameters.audit_gate",
            },
            {
                "request": "audit/request.json",
                "decision": "audit/decision.json",
                "approval": "audit/approval.json",
            },
        ),
        _stage(
            "audit_freeze",
            "post_training_audit_freeze",
            {
                "audit_plan": "artifact://audit_prepare/audit_plan",
                "audit_packet": "artifact://audit_execute/audit_packet",
                "source_bank_plan": "artifact://bank_train/bank_plan",
                "decision_key": "artifact://audit_execute/decision_key",
                "decision_manifest": "artifact://audit_gate/decision",
            },
            {"audit_freeze": "audit/audit_freeze.json"},
        ),
        _stage(
            "bank_finalize",
            "bank_finalize_audit",
            {
                "audit_freeze": "artifact://audit_freeze/audit_freeze",
                "component_index": "artifact://bank_train/component_index",
                "backend_binding": "config://baseline.parameters.backend_binding",
            },
            {
                "bank_execution_manifest": "bank/final/bank_execution_manifest.json",
                "controller_manifest": "bank/final/controller_manifest.json",
            },
        ),
    ]


def _position_control(row: Mapping[str, Any], profile: str) -> dict[str, Any]:
    dataset = row["dataset"]
    method = row["position_method"]
    position_protocol = row["protocol_snapshot"]["positions"]
    dataset_positions = position_protocol["by_dataset"][dataset]
    candidate_count = dataset_positions["candidate_count"]
    signed = method in {"rcm_zero", "rcm_patch"}
    if signed:
        role_count = (
            position_protocol["rcm"]["tldr_candidate_quota"]["target_support"]
            if dataset == "tldr"
            else position_protocol["rcm"]["downstream_quota"][dataset]["target_support"]
        )
        group_field = "selection_role"
        composition = {
            "mode": "signed_banks",
            "quotas": {
                "target_support": role_count,
                "competitor_support": role_count,
            },
            "bank_order": ["target_support", "competitor_support"],
            "merge": {
                "operator": "sum",
                "alpha_policy": "shared",
                "retrain_after_merge": False,
            },
        }
    else:
        group_field = None
        composition = {
            "mode": "unsigned",
            "merge": {
                "operator": "identity",
                "alpha_policy": "shared",
                "retrain_after_merge": False,
            },
        }
    model = _model_config(row)
    if signed:
        position_search = {
            "positive_count": role_count,
            "negative_count": role_count,
            "parent_layers_per_role": position_protocol["rcm"][
                "coarse_parent_layers_per_role"
            ],
            "score_fields": ["selection_score", "mean_score_delta", "effect"],
        }
        scan_config = {
            **_portable_protocol_value(position_protocol["rcm"]),
            "score_mode": "avglogp" if dataset == "tldr" else "answer_rest_margin",
            "parent_layers_per_role": position_protocol["rcm"][
                "coarse_parent_layers_per_role"
            ],
        }
    elif method == "iti":
        position_search = {
            "count": candidate_count,
            "score_fields": [
                "ranking_score",
                "mean_grouped_twofold_heldout_accuracy",
                "selection_score",
            ],
        }
        scan_config = _portable_protocol_value(position_protocol["iti"])
        scan_config["random_state"] = position_protocol["iti"]["hyperparameters"][
            "random_state"
        ]
    else:
        position_search = {"count": candidate_count, "seed": row["seed"]}
        scan_config = _portable_protocol_value(position_protocol["random"])
        scan_config["seed"] = row["seed"]
    if dataset == "imdb" and method in {"rcm_zero", "rcm_patch", "iti"}:
        imdb_construction = row["protocol_snapshot"]["data"]["construction_profiles"]["imdb"]
        imdb_evaluation = row["protocol_snapshot"]["evaluation_profiles"]["imdb"]
        scan_config = {
            **scan_config,
            "dataset": "imdb",
            "selector_state_semantics": "shared_model_native_good_base_complete_states",
            "selector_state_generation": _portable_protocol_value(
                imdb_construction["selector_state_generation"]
            ),
            "sentiment_scorer": {
                **_portable_protocol_value(imdb_evaluation["scorer"]),
                "local_path": "registry://models/siebert_sentiment",
            },
            "iti_terminal_eos_policy": (
                "exclude_terminal_eos_if_content_tokens_remain_to_match_official_prompt_last_token"
            ),
        }
    method_config = {
        "baseline": {
            "method": method,
            "parameters": {
                "model": {"inference": model},
                "data": {
                    "selector_manifest": {
                        "artifact": "artifact://data/selector_manifest"
                    }
                },
                "hyperparameters": {
                    "protocol_job_id": row["job_id"],
                },
                "position_search": position_search,
            },
        }
    }
    if signed:
        variant = method.removeprefix("rcm_")
        method_config["baseline"]["parameters"]["control_objective"] = {
            "family": "state_contrast",
            "variant": variant,
            "advantage": _behavior_advantage(row),
            "contrast": {
                "state_a": "zero_missing" if variant == "zero" else "reference",
                "state_b": "native_present" if variant == "zero" else "target_prototype",
                "direction": "zero_to_native" if variant == "zero" else "reference_to_target",
                "effect_type": "existence" if variant == "zero" else "replacement",
                "single_component": True,
                "recompute_downstream": True,
            },
        }
    control: dict[str, Any] = {
        "method": method,
        "scanner_binding": _binding(f"{method}_scanner"),
        "backend_binding": _binding(f"{method}"),
        "scan_config": scan_config,
        "method_config": method_config,
        "source": {
            "kind": "position_method_manifest",
            "method": method,
            "manifest": {"artifact": "artifact://position_search/candidate_manifest"},
        },
        "component_field": "component_id",
        "component_type": "pre_o_attention_head",
        "expected_count": candidate_count,
        "distinct_global": True,
        "composition": composition,
    }
    if group_field is not None:
        control["group_field"] = group_field
    position_import = row.get("position_import")
    if position_import is not None:
        if method not in {"rcm_zero", "rcm_patch"}:
            raise SiteBundleError("external RCM/K8 import is valid only for RCM methods")
        if not isinstance(position_import, Mapping):
            raise SiteBundleError("position_import must be an object")
        if position_import.get("mode") not in {"rcm_scan", "fixed_k8"}:
            raise SiteBundleError("position_import.mode must be rcm_scan or fixed_k8")
        manifest = position_import.get("manifest")
        if not isinstance(manifest, str) or not manifest.startswith("registry://"):
            raise SiteBundleError("position_import.manifest must use registry://")
        control["scanner_binding"] = _binding("external_rcm_import")
        control["scan_config"]["external_position_source"] = _portable_protocol_value(
            position_import
        )
    return control


def _method(row: Mapping[str, Any], profile: str) -> dict[str, Any]:
    snapshot = row["protocol_snapshot"]
    dataset = row["dataset"]
    cast_protocol = snapshot["cast"]
    if dataset == "tldr":
        base = cast_protocol["reft"]
        training = dict(base["training"])
        task_profile = base
        rank = base["rank"]
        apply_mode = base["apply_mode"]
        timing_id = base["timing_id"]
    else:
        base = cast_protocol["sv"]
        task_profile = base["tasks"][dataset]
        training = {**base["training_common"], **{
            key: task_profile[key]
            for key in (
                "endpoint_objective",
                "score_mode",
                "option_selection_mode",
                "preference_loss_mode",
                "dpo_beta",
                "causal_train_mask",
                "train_batch_size",
                "train_rows",
                "validation_rows",
            )
        }}
        rank = None
        apply_mode = task_profile["apply_mode"]
        timing_id = task_profile["timing_id"]
    model = _model_config(row)
    hyperparameters: dict[str, Any] = {
        "protocol_job_id": row["job_id"],
        "locked_training": _portable_protocol_value(training),
        "locked_task_profile": _portable_protocol_value(task_profile),
    }
    if rank is not None:
        hyperparameters["rank"] = rank
    parameters: dict[str, Any] = {
        "model": {"train": model, "inference": model},
        "transfer": {"mode": "none"},
        "control_objective": {
            "family": "relative_advantage",
            "advantage": _behavior_advantage(row),
            "reference_policy": "frozen_base",
            "controlled_policy": "base_plus_external_residual",
            "surrogate": "dpo_relative_log_odds",
            "base_frozen": True,
            "residual_operation": "add_external_delta_at_registered_write",
        },
        "data": {
            "training_manifest": {"artifact": "artifact://data/training_manifest"},
            "validation_manifest": {"artifact": "artifact://data/validation_manifest"},
        },
        "controller": {
            "family": row["intervention_family"],
            "training_timings": [{"name": apply_mode, "timing_id": (
                "all_token_states" if apply_mode == "all" else timing_id
            )}],
            "inference_timings": [{"name": apply_mode, "timing_id": timing_id}],
            "position_manifest": {"artifact": "artifact://position_freeze/position_freeze"},
        },
        "training": _portable_protocol_value(training),
        "inference": {"protocol": _protocol_reference(row)},
        "hyperparameters": hyperparameters,
        "backend_binding": _binding("cast"),
        "generation_backend_binding": _binding("cast_generation"),
        "bank_options": {"require_component_payloads": dataset == "tldr"},
    }
    if dataset == "tldr":
        base = cast_protocol["reft"]
        parameters["audit"] = {
            "phase": "post_bank_training",
            "unit": "component_payload",
            "expected_candidate_count": 16,
            "expected_final_count": 8,
            "audit_data_manifest": {"artifact": "artifact://data/audit_manifest"},
            "decision": {
                "decision_field": "decision",
                "keep_values": ["keep_clean", "keep_overshoot_recovered"],
                "require_complete_coverage": True,
            },
            "finalization": {"mode": "reuse_component_payloads"},
            "locked_audit": _portable_protocol_value(base["audit"]),
        }
        parameters["audit_gate"] = {
            "gate_id": "tldr_single_head_audit_v4",
            "decision_schema": "tldr_single_head_audit_v4_decision",
        }
    return {
        "bundle_id": f"site__{row['job_id']}",
        "plan_type": "method",
        "plan_id": "cast",
        "baseline": {"method": "cast", "parameters": parameters},
    }


def _alpha_contract(row: Mapping[str, Any]) -> dict[str, Any]:
    dataset = row["dataset"]
    snapshot = row["protocol_snapshot"]
    cast_protocol = snapshot["cast"]
    if dataset == "confiqa":
        downstream = cast_protocol["sv"]["tasks"]["confiqa"]
        grid = downstream["alpha_grid"]
        rule = downstream["alpha_selection"]
        selection = {
            "alpha_field": "alpha",
            "alpha_grid": grid,
            "primary_metric": rule["metric"],
            "direction": rule["direction"],
            "constraints": [],
            "near_best_tolerance": rule["near_best_absolute_tolerance"],
            "tie_break": rule["tie_break"],
        }
        generation = {**downstream["generation"], "apply_mode": downstream["apply_mode"]}
    elif dataset == "tldr":
        base = cast_protocol["reft"]
        grid = base["alpha_grid"]
        selection = {
            "alpha_field": "alpha",
            "alpha_grid": grid,
            "primary_metric": base["alpha_selection"]["metric"],
            "direction": base["alpha_selection"]["direction"],
            "constraints": [],
            "near_best_tolerance": base["alpha_selection"]["near_best_absolute_tolerance"],
            "tie_break": base["alpha_selection"]["tie_break"],
        }
        generation = base["generation"]
    else:
        downstream = cast_protocol["sv"]["tasks"]["imdb"]
        grid = downstream["alpha_grid"]
        return {
            "test_policy": {"mode": "report_full_curve", "alpha_grid": grid},
            "test_generation": {
                "stage": "test",
                "alpha_grid": grid,
                "locked_generation": {**downstream["generation"], "apply_mode": downstream["apply_mode"]},
                "policy_audit": {
                    "enabled": True,
                    "reference_policy": "frozen_base",
                    "metrics": [
                        "sampled_sequence_logprob_ratio",
                        "token_kl_audit",
                    ],
                },
            },
        }
    return {
        "alpha_selector_binding": {"id": "deterministic_metric", "version": 1},
        "alpha_selection": selection,
        "alpha_generation": {
            "stage": "alpha_dev",
            "alpha_grid": grid,
            "locked_generation": generation,
        },
        "test_generation": {"stage": "test", "locked_generation": generation},
    }


def build_site_job_bundle(
    job_id: str,
    *,
    output_dir: Path,
    matrix_path: Path = DEFAULT_MATRIX,
    execution_profile: str = "formal",
) -> dict[str, Any]:
    """Write one immutable derived bundle for exactly one locked Site job."""

    if execution_profile not in {"formal", "formal_scaled_gpu"}:
        raise SiteBundleError("execution_profile must be formal or formal_scaled_gpu")
    row = resolve_site_cell(job_id, matrix_path=matrix_path)["cell"]
    dataset = row["dataset"]
    roles, role_outputs = _roles(dataset)
    protocol = _protocol_reference(row)
    model = _model_config(row)
    data = {
        "backend_binding": _binding(f"site_data_{dataset}"),
        "protocol": protocol,
        "shared_artifact_key": (
            f"site_data__{protocol['id']}__{dataset}__{row['subset']}__"
            f"{row['model_family']}__{execution_profile}"
        ),
        "provenance": row["protocol_snapshot"]["data"]["provenance"]["datasets"][dataset],
        "construction": row["protocol_snapshot"]["data"]["construction_profiles"][dataset],
        "prompt_registry": row["protocol_snapshot"]["data"]["prompt_registry"],
        "subset": row["subset"],
        "model_family": row["model_family"],
        "roles": roles,
        "role_outputs": role_outputs,
        "sources": _source_bindings(dataset, row["subset"], row["model_family"]),
    }
    if dataset == "imdb":
        scorer = row["protocol_snapshot"]["evaluation_profiles"]["imdb"]["scorer"]
        data["runtime"] = {
            "model": model,
            "sentiment_scorer": {
                "local_path": "registry://models/siebert_sentiment",
                "revision": scorer["revision"],
                "reward_batch_size": scorer["reward_batch_size"],
            },
        }
    if execution_profile == "formal_scaled_gpu":
        profiles = row["protocol_snapshot"]["data"].get("scaled_validation", {}).get("role_counts_by_dataset")
        scaled = profiles.get(dataset) if isinstance(profiles, Mapping) else None
        if not isinstance(scaled, Mapping):
            raise SiteBundleError(
                "formal_scaled_gpu requires explicit scaled_validation.role_counts_by_dataset"
            )
        data["execution_profile"] = execution_profile
        data["scaled_counts"] = {role: int(count) for role, count in scaled.items() if role in roles}
    flow = [_data_stage(role_outputs), *_position_stages(), _bank_stage(audited=dataset == "tldr")]
    controller_ref = "artifact://bank_train/controller_manifest"
    if dataset == "tldr":
        flow.extend(_tldr_audit_flow())
        controller_ref = "artifact://bank_finalize/controller_manifest"
        flow.extend(_selected_alpha_flow(dataset, controller_ref))
    elif dataset == "confiqa":
        flow.extend(_selected_alpha_flow(dataset, controller_ref))
    else:
        flow.extend(_imdb_full_curve_flow(controller_ref))
    selection = _alpha_contract(row)
    shared = {
        "data": data,
        "models": {"train": model, "inference": model},
        "flow": flow,
        "selection_calibration_test": selection,
        "evaluation": {
            "scorer": row["scorer"],
            "backend_binding": _binding(f"site_evaluator_{dataset}"),
            "profile": row["protocol_snapshot"]["evaluation_profiles"][dataset],
            "alpha": {"dataset": dataset, "subset": row["subset"], "stage": "alpha_dev", "protocol": protocol},
            "test": {"dataset": dataset, "subset": row["subset"], "stage": "test", "protocol": protocol},
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
            "sentiment_model_path": "registry://models/siebert_sentiment",
        }
    bundle_id = f"site__{job_id}"
    master = {
        "schema_version": 1,
        "bundle_id": bundle_id,
        "experiment": "site",
        "dataset": dataset,
        "model_family": row["model_family"],
        "protocol": {"id": protocol["id"], "path": protocol["path"]},
        "shared": shared,
        "plans": {
            "method": {"cast": "method.cast.json"},
            "position": {row["position_method"]: f"position.{row['position_method']}.json"},
        },
        "matrix": [
            {
                "cell_id": job_id,
                "method_ref": "cast",
                "position_ref": row["position_method"],
                "locked_job": {
                    "selector": row["selector"],
                    "replicate": row["replicate"],
                    "seed": row["seed"],
                    "subset": row["subset"],
                },
            }
        ],
    }
    method = _method(row, execution_profile)
    position = {
        "bundle_id": bundle_id,
        "plan_type": "position",
        "plan_id": row["position_method"],
        "position_control": _position_control(row, execution_profile),
    }
    output_dir = output_dir.expanduser().resolve()
    _write_exclusive(output_dir / "bundle.json", master)
    _write_exclusive(output_dir / "method.cast.json", method)
    _write_exclusive(
        output_dir / f"position.{row['position_method']}.json", position
    )
    manifest = {
        "schema_version": 1,
        "status": "derived",
        "execution_profile": execution_profile,
        "job_id": job_id,
        "bundle_directory": str(output_dir),
        "protocol": protocol,
        "files": [path.name for path in sorted(output_dir.glob("*.json"))],
    }
    return manifest
