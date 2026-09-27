#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


EXPECTED_SELECTORS = {"rcm_zero_signed", "rcm_patch_signed", "iti", "random"}
EXPECTED_NATIVE = {
    "base_or_strong_prompt",
    "caa",
    "bipo",
    "reps",
    "loreft",
    "cast",
    "context_dpo",
    "imdb_dpo",
}
HEAD_PATTERN = re.compile(r"^L[0-9]+[.]attn[.]h[0-9]+$")


def fail(message: str) -> None:
    raise ValueError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def load_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(data, dict), "config root must be an object")
    return data


def validate_config(data: dict[str, Any]) -> None:
    require(data.get("protocol_id") == "site_position_full_fixed_protocol_20260717_v8", "unexpected protocol_id")
    require(data.get("version") == 8, "Site protocol version must be eight")
    require(data.get("status") == "locked_before_gpu_execution", "unexpected protocol status")
    require(data.get("documentation") == "docs/SITE_POSITION_FULL_FIXED_PROTOCOL_20260717_V8.md", "documentation drift")
    require(data.get("preflight") == "scripts/paper/25_prepare_site_position_full_protocol_v8.py", "preflight drift")
    supersedes = data.get("supersedes", {})
    require(supersedes.get("protocol_id") == "site_position_full_fixed_protocol_20260716_v7", "supersession drift")
    require(supersedes.get("version") == 7, "superseded version drift")
    require(
        supersedes.get("config_sha256")
        == "db555d79c61f26997ce8d9179094f3a9e1bf373a73c2ee13cda65f8b87ab26d2",
        "superseded version-7 hash drift",
    )
    site = data.get("site", {})
    contract = site.get("selection_contract", {})
    require(contract.get("candidate_space") == "pre_o_attention_head", "candidate space must be pre-O heads")
    require(contract.get("selected_count") == 8, "Site selected_count must be exactly eight")
    require(
        contract.get("direction_by_selector")
        == {
            "rcm_zero_signed": "signed_target_support_and_competitor_support",
            "rcm_patch_signed": "signed_target_support_and_competitor_support",
            "iti": "unsigned_heldout_probe_accuracy",
            "random": "unsigned_uniform_control",
        },
        "selector direction policy drift",
    )
    require(
        contract.get("signed_partition")
        == {"target_support_count": 4, "competitor_support_count": 4},
        "signed zero partition drift",
    )
    require(contract.get("nonpositive_fallback") is False, "non-positive fallback is forbidden")
    require(contract.get("ci_policy") == "report_only", "CI must be report_only")
    require(contract.get("tie_break") == "ascending_layer_then_head", "unexpected selector tie break")

    trajectory = site.get("decision_trajectory", {})
    require(trajectory.get("name") == "generation_decision_states", "wrong RCM decision trajectory")
    require(trajectory.get("include_padding") is False, "padding cannot enter the trajectory")
    require(
        trajectory.get("include_post_final_output_state") is False,
        "post-final-output state cannot enter the trajectory",
    )

    common = site.get("rcm_common", {})
    require(common.get("coarse_beam_width") == 4, "RCM coarse beam must be four layers")
    require(common.get("final_unit") == "pre_o_attention_head", "RCM final unit must be a pre-O head")
    require(common.get("state_timing") == "generation_decision_states", "RCM timing drift")
    require(common.get("patch_constructor") == "trajectory_mean_state_clamp", "RCM patch constructor drift")
    require(
        common.get("prototype_aggregation_order")
        == [
            "decision_state_mean_within_sequence",
            "target_alias_mean_within_sample_when_present",
            "sample_mean",
        ],
        "RCM prototype averaging order drift",
    )
    require(common.get("prototype_accumulation_dtype") == "float32", "prototype accumulation drift")
    require(common.get("prototype_reward_filter") is False, "prototype reward filtering is forbidden")
    require(common.get("prototype_health_filter") is False, "prototype health filtering is forbidden")
    require(common.get("patch_operation") == "overwrite", "RCM patch must overwrite")
    require(common.get("patch_selector_alpha") == 1.0, "selector patch alpha must be one")
    require(common.get("patch_normalization") == "none", "selector patch normalization is forbidden")
    require(common.get("patch_runtime_target_stream") is False, "runtime target stream is forbidden")
    require(
        common.get("share_candidate_scoring_rows_between_zero_and_patch") is True,
        "RCM zero/patch must share candidate scoring rows",
    )
    require(
        common.get("zero_positive_layer_ranking")
        == "descending_strictly_positive_mean_paired_effect",
        "positive layer ranking drift",
    )
    require(
        common.get("zero_negative_layer_ranking")
        == "ascending_strictly_negative_mean_paired_effect",
        "negative layer ranking drift",
    )
    require(
        common.get("zero_positive_head_ranking")
        == "descending_strictly_positive_mean_paired_effect_within_positive_layer_beam",
        "positive head ranking drift",
    )
    require(
        common.get("zero_negative_head_ranking")
        == "ascending_strictly_negative_mean_paired_effect_within_negative_layer_beam",
        "negative head ranking drift",
    )
    require(
        common.get("shared_zero_scan_artifacts")
        == [
            "layer_results.jsonl",
            "positive_layer_beam.jsonl",
            "negative_layer_beam.jsonl",
            "positive_head_results.jsonl",
            "negative_head_results.jsonl",
            "scan_manifest.json",
        ],
        "shared zero artifact contract drift",
    )
    require(common.get("patch_positive_layer_ranking") == "descending_strictly_positive_mean_paired_effect", "patch positive layer ranking drift")
    require(common.get("patch_negative_layer_ranking") == "ascending_strictly_negative_mean_paired_effect", "patch negative layer ranking drift")
    require(common.get("patch_positive_head_ranking") == "descending_strictly_positive_mean_paired_effect_within_positive_layer_beam", "patch positive head ranking drift")
    require(common.get("patch_negative_head_ranking") == "ascending_strictly_negative_mean_paired_effect_within_negative_layer_beam", "patch negative head ranking drift")

    downstream_contract = site.get("downstream_contract", {})
    require(downstream_contract.get("native_selected_writes") == "kept_active", "native writes must remain active")
    require(
        downstream_contract.get("locator_operations_present_during_training_or_evaluation") is False,
        "zero/patch must be locator-only operations",
    )
    require(
        downstream_contract.get("selector_specific_downstream_variable") == "component_id_only",
        "selector may change only downstream component ids",
    )
    require(downstream_contract.get("warm_start") is False, "downstream warm start is forbidden")
    require(downstream_contract.get("learned_gate") is False, "downstream learned gate is forbidden")

    downstream_common = site.get("downstream_common", {})
    require(downstream_common.get("optimizer") == "torch.optim.AdamW", "optimizer drift")
    require(downstream_common.get("lr") == 0.05, "learning-rate drift")
    require(downstream_common.get("weight_decay") == 0.01, "AdamW weight-decay drift")
    require(downstream_common.get("epochs") == 2, "epoch drift")
    require(downstream_common.get("seed") == 42, "training seed drift")
    require(downstream_common.get("alpha_train") == 1.0, "training alpha drift")
    require(downstream_common.get("lambda_norm") == 0.0001, "lambda_norm drift")
    require(downstream_common.get("vector_initialization") == "float32_zeros", "vector init drift")

    selectors = site.get("selectors", {})
    require(set(selectors) == EXPECTED_SELECTORS, "full position selector set drift")
    require(
        list(selectors) == ["rcm_zero_signed", "rcm_patch_signed", "iti", "random"],
        "selector order drift",
    )
    require(
        selectors["rcm_zero_signed"].get("delta")
        == "C(native_target)-C(zero_i(native_target))",
        "signed zero delta drift",
    )
    require(selectors["rcm_zero_signed"].get("shared_scan_key") == "rcm_zero", "signed zero scan drift")
    require(selectors["rcm_zero_signed"].get("direction") == "signed", "signed zero direction drift")
    require(
        selectors["rcm_patch_signed"].get("delta")
        == "C(clamp_i(recipient,mu_i_target))-C(recipient)",
        "wrong patch delta",
    )
    require(
        selectors["rcm_patch_signed"].get("operation")
        == "overwrite_candidate_with_transferable_trajectory_mean_target_state_at_every_generation_decision_state",
        "wrong patch operation",
    )
    require(selectors["rcm_patch_signed"].get("direction") == "signed", "patch direction drift")
    require(selectors["rcm_patch_signed"].get("selection") == "site.selection_contract.signed_partition", "patch signed selection drift")
    require(selectors["iti"].get("direction") == "unsigned", "ITI direction drift")
    require(selectors["iti"].get("folds") == 2, "ITI must use two folds")
    require(selectors["iti"].get("max_iter") == 1000, "ITI max_iter must match pinned code")
    require(selectors["iti"].get("solver") == "lbfgs", "ITI solver drift")
    require(selectors["iti"].get("C") == 1.0, "ITI C drift")
    require(selectors["iti"].get("feature_standardization") == "none", "ITI scaling drift")
    require(selectors["random"].get("direction") == "unsigned", "Random direction drift")
    require(
        selectors["random"].get("replicate_seeds") == [20260702, 20260703, 20260704],
        "unexpected Random seeds",
    )

    tasks = site.get("tasks", {})
    require(set(tasks) == {"imdb", "confiqa"}, "Site tasks must be IMDb and ConFiQA")
    imdb = tasks["imdb"]
    require(imdb["states"]["base_template"] == "{prefix}", "IMDb base state drift")
    require(
        imdb["states"]["good_template"] == "{prefix}\n\nContinue the review with a positive sentiment:",
        "IMDb good state drift",
    )
    require(
        imdb["states"]["shared_behavior_edge"]
        == "good_template_complete_generation_vs_base_template_complete_generation",
        "IMDb shared behavior edge drift",
    )
    require(imdb["states"]["rcm_zero_native"] == "good_template", "IMDb zero target drift")
    require(imdb["states"]["rcm_patch_recipient"] == "base_template", "IMDb patch base drift")
    require(
        imdb["states"]["rcm_patch_target_prototype"]
        == "trajectory_mean_over_shared_good_template_generations",
        "IMDb target prototype drift",
    )
    require(
        imdb["states"]["iti_positive_state"] == "shared_good_template_complete_generation",
        "IMDb ITI positive state drift",
    )
    require(
        imdb["states"]["iti_negative_state"] == "shared_base_template_complete_generation",
        "IMDb ITI negative state drift",
    )
    require(imdb["score"]["name"] == "generated_sentiment_competition", "IMDb selector score drift")
    state_generation = imdb["selector_state_generation"]
    require(state_generation["prompt_rows"] == 300, "IMDb selector window drift")
    require(state_generation["conditions"] == ["good", "base"], "IMDb selector condition order drift")
    require(
        state_generation["max_new_tokens"] == 64,
        "IMDb selector state window drift",
    )
    require(
        state_generation["state_filter"] == "none",
        "IMDb selector state filtering is forbidden",
    )
    require(
        state_generation["common_random_numbers_across_conditions"] is True,
        "IMDb good/base states must use common random numbers",
    )
    require(
        state_generation["shared_artifact_policy"]
        == "generate_once_per_model_and_reuse_exact_token_ids",
        "IMDb shared state artifact policy drift",
    )
    require(
        imdb["selector_preprocessing"]["rcm"]["coarse_score_rows"] == 300,
        "IMDb RCM coarse budget drift",
    )
    imdb_iti = imdb["selector_preprocessing"]["iti"]
    require(
        imdb_iti["pair_source"] == "same_shared_good_base_generation_artifact_as_rcm",
        "IMDb ITI shared state source drift",
    )
    require(imdb_iti["pair_groups"] == 300, "IMDb ITI group budget drift")
    require(
        imdb_iti["terminal_eos_policy"]
        == "exclude_terminal_eos_if_content_tokens_remain_to_match_official_prompt_last_token",
        "IMDb ITI terminal EOS policy drift",
    )
    require(
        imdb_iti["reward_ranking_used"] is False,
        "IMDb ITI reward ranking is forbidden",
    )
    require(
        imdb_iti["raw_imdb_labels_used"] is False,
        "IMDb ITI may not use raw source labels",
    )
    require(imdb["pair_preparation"]["target_train_pairs"] == 1024, "IMDb pair target drift")
    require(imdb["pair_preparation"]["target_val_pairs"] == 256, "IMDb pair target drift")
    require(imdb["downstream"]["train_rows"] == 1024, "IMDb train size drift")
    require(imdb["downstream"]["val_rows"] == 256, "IMDb val size drift")
    require(imdb["downstream"]["apply_mode"] == "all", "IMDb downstream timing drift")
    require(imdb["downstream"]["max_new_tokens"] == 256, "IMDb test generation window drift")

    confiqa = tasks["confiqa"]
    require(confiqa["states"]["rcm_zero_native"] == "official_rag", "ConFiQA zero native drift")
    require(confiqa["states"]["rcm_patch_recipient"] == "official_rag", "ConFiQA patch recipient drift")
    require(
        confiqa["states"]["rcm_patch_target_prototype"]
        == "trajectory_mean_over_official_rag_plus_all_context_answer_aliases",
        "ConFiQA target prototype drift",
    )
    require(
        confiqa["states"]["rcm_patch_target_prompt_key"] == "official_rag",
        "ConFiQA patch prompt must match the ITI pair prompt",
    )
    require(
        confiqa["score"]["name"] == "closed_context_over_prior_competition_margin",
        "ConFiQA selector score drift",
    )
    require(confiqa["score"]["open_generation_used_for_selector"] is False, "ConFiQA selector must be closed")
    require(confiqa["score"]["score_mode"] == "answer_rest_margin", "ConFiQA score-mode drift")
    require(confiqa["score"]["max_aliases_per_side"] == 0, "ConFiQA selector must use all aliases")
    require(
        confiqa["selector_preprocessing"]["rcm"]["prototype_rows_source"] == "head_refine_pair_groups",
        "ConFiQA prototype rows must be the head-refinement groups",
    )
    rcm_prep = confiqa["selector_preprocessing"]["rcm"]
    iti_prep = confiqa["selector_preprocessing"]["iti"]
    require(rcm_prep["candidate_group_source"] == "shared_admitted_pair_artifact", "RCM group source drift")
    require(rcm_prep["candidate_split_filter"] == "none", "RCM may not filter the common selector groups")
    require(rcm_prep["source_start"] == 0, "ConFiQA selector source start drift")
    require(rcm_prep["source_stop"] == 300, "ConFiQA initial selector window drift")
    require(
        rcm_prep["source_window_role"] == "initial_candidate_window_before_valid_group_refill",
        "ConFiQA selector window role drift",
    )
    require(rcm_prep["target_admitted_groups"] == 300, "ConFiQA admitted-group target drift")
    require(rcm_prep["fill_rejected_groups"] is True, "ConFiQA rejected groups must be refilled")
    require(
        rcm_prep["fill_order"] == "continue_official_file_order_after_source_stop",
        "ConFiQA refill order drift",
    )
    require(rcm_prep["admission_audit_required"] is True, "ConFiQA admission audit is required")
    require(
        rcm_prep["admission_audit_artifacts"]
        == [
            "sample_admission_manifest.json",
            "admitted_source_rows.csv",
            "rejected_source_rows.jsonl",
            "replacement_source_rows.jsonl",
        ],
        "ConFiQA admission audit artifact drift",
    )
    require(rcm_prep["coarse_pair_groups"] == 300, "ConFiQA coarse group budget drift")
    require(rcm_prep["head_refine_pair_groups"] == 300, "ConFiQA head group budget drift")
    require(iti_prep["pair_source"] == "same_shared_admitted_pair_artifact_as_rcm", "ITI group source drift")
    require(iti_prep["pair_groups"] == 300, "ITI group budget drift")
    require(confiqa["downstream"]["train_rows"] == 240, "ConFiQA train size drift")
    require(confiqa["downstream"]["val_rows"] == 60, "ConFiQA val size drift")
    require(confiqa["downstream"]["apply_mode"] == "decision_tokens", "ConFiQA timing drift")
    require(
        confiqa["downstream"]["alpha_dev"]
        == {
            "source_start": 301,
            "source_stop": 421,
            "eval_cap": 120,
            "selection_metric": "logic_score",
            "tie_break": "smallest_alpha",
            "source_start_policy": "max_registered_start_and_selector_scan_stop_plus_one",
            "raw_window_rows": 120,
            "fill_rejected_rows": False,
        },
        "ConFiQA alpha-dev drift",
    )
    require(
        confiqa["downstream"]["heldout_test"]
        == {"source_start": 1000, "source_stop": 1500, "eval_cap": 500, "fixed_window_no_fill": True},
        "ConFiQA held-out window drift",
    )

    for task_name, task in tasks.items():
        downstream = task["downstream"]
        grid = downstream["alpha_nonzero_grid"]
        require(0 not in grid and 0.0 not in grid, f"{task_name}: alpha zero must not be repeated")
        require(len(grid) == len(set(grid)), f"{task_name}: duplicate alpha values")
        require(
            str(downstream["alpha_zero"]).startswith("evaluate_once_and_reuse_per_dataset_model"),
            f"{task_name}: alpha-zero drift",
        )

    matrix = site.get("matrix", [])
    require(len(matrix) == 6, "Site matrix must contain six dataset-model entries")
    identities = {(row["dataset"], row["model_key"]) for row in matrix}
    require(len(identities) == len(matrix), "duplicate dataset-model matrix entry")
    require(sum(row["dataset"] == "confiqa" for row in matrix) == 3, "ConFiQA model count drift")
    require(sum(row["dataset"] == "imdb" for row in matrix) == 3, "IMDb model count drift")
    for row in matrix:
        require(row["dataset"] in tasks, f"unknown matrix dataset: {row['dataset']}")
        require(re.fullmatch(r"[0-9a-f]{40}", row.get("revision", "")) is not None, "matrix model unpinned")

    execution = site.get("execution", {})
    require(
        execution.get("run_root") == "/dev/shm/screscomp_runs/site_position_full_20260717_v8",
        "run root drift",
    )
    require(execution.get("serial_only") is True, "Site execution must remain serial")
    require(execution.get("empty_gpu_required") is False, "shared-GPU execution policy drift")
    require(execution.get("gpu_co_tenancy_allowed") is True, "GPU co-tenancy policy drift")
    require(execution.get("preferred_physical_gpu_indices") == [2, 3], "allowed GPU set drift")
    require(
        execution.get("gpu_selection_policy") == "lowest_observed_memory_used_at_matrix_launch",
        "GPU launch selection policy drift",
    )
    history = execution.get("historical_alpha_zero_reuse", {})
    require(history.get("allowed") is True, "registered alpha-zero reuse policy missing")
    require(
        history.get("source_config") == "configs/site_position_full_fixed_protocol_20260716_v6.json",
        "historical source config path drift",
    )
    require(
        history.get("source_config_sha256")
        == "96693c4448f468874dca38d9e250e715ba2d7e7a0ba4a7fcfae3dd8363a29d61",
        "historical source hash drift",
    )
    require(history.get("eligible_artifacts") == "alpha_zero_native_base_only", "historical reuse scope drift")

    native = data.get("native_plus_rlhf", {})
    native_ids = {row["id"] for row in native.get("methods", [])}
    require(native_ids == EXPECTED_NATIVE, "native + RLHF method set drift")
    require(native.get("site_count_policy") == "official_native_setting_not_forced_to_eight", "native size drift")
    require(
        native.get("gcm_policy") == "not_a_native_external_row",
        "GCM policy drift",
    )

    repositories = data.get("external_repositories", {})
    require(set(repositories) == {"honest_llama", "CAA", "BiPO", "axbench", "pyreft"}, "repo set drift")
    for name, spec in repositories.items():
        require(re.fullmatch(r"[0-9a-f]{40}", spec.get("commit", "")) is not None, f"{name}: unpinned")
        require(spec.get("url", "").startswith("https://github.com/"), f"{name}: non-GitHub source")
        require(
            str(spec.get("path", "")).startswith("LOCAL_HOME/RPEC/projects/baselines/"),
            f"{name}: path drift",
        )


def validate_superseded_scientific_lock(data: dict[str, Any], config_path: Path) -> dict[str, Any]:
    superseded = data["supersedes"]
    old_path = config_path.resolve().parent / "site_position_full_fixed_protocol_20260716_v7.json"
    require(old_path.is_file(), f"missing superseded Site config: {old_path}")
    old_hash = hashlib.sha256(old_path.read_bytes()).hexdigest()
    require(old_hash == superseded["config_sha256"], "superseded Site config hash drift")
    old = load_config(old_path)
    require(old.get("protocol_id") == superseded["protocol_id"], "superseded protocol id drift")
    expected = copy.deepcopy(old)
    for key in (
        "protocol_id", "version", "last_updated", "status", "remote_source_of_truth",
        "documentation", "preflight", "supersedes",
    ):
        expected[key] = copy.deepcopy(data[key])
    expected["site"]["tasks"]["imdb"]["states"] = copy.deepcopy(
        data["site"]["tasks"]["imdb"]["states"]
    )
    expected["site"]["tasks"]["imdb"]["selector_state_generation"] = copy.deepcopy(
        data["site"]["tasks"]["imdb"]["selector_state_generation"]
    )
    expected["site"]["tasks"]["imdb"]["selector_preprocessing"] = copy.deepcopy(
        data["site"]["tasks"]["imdb"]["selector_preprocessing"]
    )
    expected["site"]["execution"] = copy.deepcopy(data["site"]["execution"])
    require(expected == data, "version 8 changed fields outside the registered IMDb good/base state-alignment boundary")
    return {
        "superseded_config": str(old_path),
        "superseded_config_sha256": old_hash,
        "allowed_changes": [
            "protocol metadata and supersession boundary",
            "IMDb shared good/base selector-state generation contract",
            "IMDb ITI labels changed from reward-ranked downstream pairs to the shared good/base states",
            "IMDb RCM-patch recipient and prototype provenance bound to the exact shared state artifacts",
            "execution roots and superseded-result boundary",
        ],
    }


def check_repositories(data: dict[str, Any]) -> None:
    for name, spec in data["external_repositories"].items():
        path = Path(spec["path"])
        require(path.is_dir(), f"{name}: missing repository {path}")
        head = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        url = subprocess.check_output(["git", "-C", str(path), "remote", "get-url", "origin"], text=True).strip()
        require(head == spec["commit"], f"{name}: expected {spec['commit']}, found {head}")
        require(url == spec["url"], f"{name}: expected {spec['url']}, found {url}")


def compact(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def selector_replicates(selector: str, spec: dict[str, Any]) -> list[tuple[str, int]]:
    if selector == "random":
        return [(f"seed_{seed}", int(seed)) for seed in spec["replicate_seeds"]]
    seed = int(spec.get("seed", spec.get("random_state", 42)))
    return [("primary", seed)]


def emit_plan(data: dict[str, Any], config_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = config_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    (out_dir / "config.sha256").write_text(digest + "\n", encoding="ascii")
    (out_dir / "config.snapshot.json").write_bytes(raw)

    site = data["site"]
    tasks = site["tasks"]
    selectors = site["selectors"]
    fields = [
        "order",
        "protocol_version",
        "job_id",
        "dataset",
        "subset",
        "model_key",
        "model",
        "model_revision",
        "use_chat_template",
        "selector",
        "replicate",
        "seed",
        "selected_count",
        "candidate_space",
        "decision_trajectory",
        "rcm_common",
        "selector_preprocessing",
        "state_spec",
        "score_spec",
        "downstream_common",
        "downstream_spec",
        "output_dir",
    ]
    rows: list[dict[str, Any]] = []
    order = 0
    run_root = site["execution"]["run_root"]
    for matrix_row in site["matrix"]:
        dataset = matrix_row["dataset"]
        task = tasks[dataset]
        for subset in task["subsets"]:
            for selector, selector_spec in selectors.items():
                for replicate, seed in selector_replicates(selector, selector_spec):
                    order += 1
                    job_id = "__".join([dataset, subset, matrix_row["model_key"], selector, replicate])
                    prep_key = "rcm" if selector.startswith("rcm_") else selector
                    rows.append(
                        {
                            "order": order,
                            "protocol_version": data["version"],
                            "job_id": job_id,
                            "dataset": dataset,
                            "subset": subset,
                            "model_key": matrix_row["model_key"],
                            "model": matrix_row["model"],
                            "model_revision": matrix_row["revision"],
                            "use_chat_template": int(bool(matrix_row["use_chat_template"])),
                            "selector": selector,
                            "replicate": replicate,
                            "seed": seed,
                            "selected_count": site["selection_contract"]["selected_count"],
                            "candidate_space": site["selection_contract"]["candidate_space"],
                            "decision_trajectory": compact(site["decision_trajectory"]),
                            "rcm_common": compact(site["rcm_common"]),
                            "selector_preprocessing": compact(task["selector_preprocessing"][prep_key]),
                            "state_spec": compact(task["states"]),
                            "score_spec": compact(task["score"]),
                            "downstream_common": compact(site["downstream_common"]),
                            "downstream_spec": compact(task["downstream"]),
                            "output_dir": f"{run_root}/{job_id}",
                        }
                    )
    require(len(rows) == 72, f"full position plan must contain 72 jobs, got {len(rows)}")
    with (out_dir / "selector_jobs.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    native_fields = ["id", "kind", "repository", "tasks", "model", "revision", "configuration_policy", "retrain"]
    with (out_dir / "native_plus_rlhf.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=native_fields, delimiter="\t")
        writer.writeheader()
        for method in data["native_plus_rlhf"]["methods"]:
            writer.writerow(
                {
                    "id": method["id"],
                    "kind": method["kind"],
                    "repository": method.get("repository", ""),
                    "tasks": ",".join(method.get("tasks", [])),
                    "model": method.get("model", ""),
                    "revision": method.get("revision", ""),
                    "configuration_policy": method.get("configuration_policy", ""),
                    "retrain": method.get("retrain", ""),
                }
            )
    selector_counts: dict[str, int] = {}
    for row in rows:
        selector_counts[row["selector"]] = selector_counts.get(row["selector"], 0) + 1
    audit = {
        "protocol_id": data["protocol_id"],
        "protocol_version": data["version"],
        "config_sha256": digest,
        "status": data["status"],
        "job_count": len(rows),
        "stratum_count": 12,
        "selector_job_counts": selector_counts,
        "selected_count": site["selection_contract"]["selected_count"],
        "signed_partition": site["selection_contract"]["signed_partition"],
        "direction_by_selector": site["selection_contract"]["direction_by_selector"],
        "random_replicate_seeds": site["selectors"]["random"]["replicate_seeds"],
        "run_root": run_root,
        "ci_policy": site["selection_contract"]["ci_policy"],
        "superseded_boundary": data["supersedes"],
        **validate_superseded_scientific_lock(data, config_path),
    }
    (out_dir / "protocol_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"validated protocol={data['protocol_id']} sha256={digest}")
    print(f"wrote selector_jobs={len(rows)} out={out_dir}")


def validate_selector_output(path: Path, expected_selector: str) -> None:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 8, "selector output must contain exactly eight rows")
    require(expected_selector in EXPECTED_SELECTORS, "unknown expected selector")
    require({row.get("selector", "") for row in rows} == {expected_selector}, "selector mismatch")
    require([int(row["rank"]) for row in rows] == list(range(1, 9)), "ranks must be 1..8")
    components = [row["component_id"] for row in rows]
    require(len(set(components)) == 8, "selected heads must be distinct")
    require(all(HEAD_PATTERN.fullmatch(component) for component in components), "invalid head id")
    require(all(row.get("selection_role", "") for row in rows), "selection role is required")

    if expected_selector.startswith("rcm_"):
        require(all(row.get("state_constructor", "") for row in rows), "RCM output lacks state constructor")
        require(all(row.get("mean_score_delta", "") != "" for row in rows), "RCM output lacks mean effect")
        require(all(row.get("ci95_low", "") != "" and row.get("ci95_high", "") != "" for row in rows), "RCM output lacks CI audit")
        deltas = [float(row["mean_score_delta"]) for row in rows]
    if expected_selector == "rcm_zero_signed":
        positive, negative = rows[:4], rows[4:]
        positive_deltas, negative_deltas = deltas[:4], deltas[4:]
        require({row["selection_role"] for row in positive} == {"target_support"}, "signed positive role drift")
        require({row["selection_role"] for row in negative} == {"competitor_support"}, "signed negative role drift")
        require(all(value > 0 for value in positive_deltas), "signed target-support effect must be positive")
        require(all(value < 0 for value in negative_deltas), "signed competitor-support effect must be negative")
        require(positive_deltas == sorted(positive_deltas, reverse=True), "signed positive effects are not descending")
        require(negative_deltas == sorted(negative_deltas), "signed negative effects are not ascending")
        require(all(row.get("shared_scan_artifact", "") for row in rows), "signed zero lacks shared scan provenance")
    elif expected_selector == "rcm_patch_signed":
        positive, negative = rows[:4], rows[4:]
        positive_deltas, negative_deltas = deltas[:4], deltas[4:]
        require({row["selection_role"] for row in positive} == {"target_support"}, "patch positive role drift")
        require({row["selection_role"] for row in negative} == {"competitor_support"}, "patch negative role drift")
        require(all(value > 0 for value in positive_deltas), "patch target-support effect must be positive")
        require(all(value < 0 for value in negative_deltas), "patch competitor-support effect must be negative")
        require(positive_deltas == sorted(positive_deltas, reverse=True), "patch positive effects are not descending")
        require(negative_deltas == sorted(negative_deltas), "patch negative effects are not ascending")
        require(all(row.get("prototype_artifact", "") for row in rows), "patch output lacks prototype artifact")
    elif expected_selector in {"iti", "random"}:
        require({row["selection_role"] for row in rows} == {"unsigned"}, f"{expected_selector}: sign labels are forbidden")

    if expected_selector in {"rcm_zero_signed"}:
        require(
            {row["state_constructor"] for row in rows}
            == {"zero_at_every_generation_decision_state"},
            "RCM-zero output has the wrong state constructor",
        )
    if expected_selector == "rcm_patch_signed":
        require(
            {row["state_constructor"] for row in rows}
            == {"trajectory_mean_state_clamp_at_every_generation_decision_state"},
            "RCM-patch output has the wrong state constructor",
        )
    print(f"validated selector={expected_selector} rows=8 path={path}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and expand the locked full Site position protocol.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check-repositories", action="store_true")
    parser.add_argument("--emit-plan", type=Path)
    parser.add_argument("--validate-selector-output", type=Path)
    parser.add_argument("--expected-selector", choices=sorted(EXPECTED_SELECTORS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_config(args.config)
    validate_config(data)
    validate_superseded_scientific_lock(data, args.config)
    if args.check_repositories:
        check_repositories(data)
    if args.emit_plan is not None:
        emit_plan(data, args.config, args.emit_plan)
    if args.validate_selector_output is not None:
        require(args.expected_selector is not None, "--expected-selector is required with output validation")
        validate_selector_output(args.validate_selector_output, args.expected_selector)
    if args.emit_plan is None and args.validate_selector_output is None:
        print(f"validated protocol={data['protocol_id']}")


if __name__ == "__main__":
    main()
