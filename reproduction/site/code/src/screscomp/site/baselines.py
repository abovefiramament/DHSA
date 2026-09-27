from __future__ import annotations

import ast
import random
import subprocess
from pathlib import Path
from typing import Any, Callable

from screscomp.cecm.actuator import load_actuator_pairs
from screscomp.data import dump_csv, dump_json
from screscomp.site.imdb_states import ensure_imdb_selector_states
from screscomp.site.model import PreOInterface
from screscomp.site.protocol import SiteJob, SitePaths, SiteProtocol, require_files


def _load_official_train_probes(protocol: SiteProtocol) -> tuple[Callable[..., Any], dict[str, Any]]:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from tqdm import tqdm

    repo = protocol.data["external_repositories"]["honest_llama"]
    repo_path = Path(repo["path"])
    source_path = repo_path / "utils.py"
    if not source_path.is_file():
        raise RuntimeError(f"pinned honest_llama source is missing: {source_path}")
    commit = subprocess.check_output(
        ["git", "-C", str(repo_path), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != repo["commit"]:
        raise RuntimeError(f"honest_llama commit drift: expected {repo['commit']}, got {commit}")
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    matches = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "train_probes"
    ]
    if len(matches) != 1:
        raise RuntimeError("could not isolate official honest_llama train_probes")
    module = ast.Module(body=matches, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {
        "np": np,
        "tqdm": tqdm,
        "LogisticRegression": LogisticRegression,
        "accuracy_score": accuracy_score,
    }
    exec(compile(module, str(source_path), "exec"), namespace)
    defaults = LogisticRegression().get_params()
    locked = protocol.site["selectors"]["iti"]
    checks = {
        "penalty": locked["penalty"],
        "C": locked["C"],
        "solver": locked["solver"],
        "fit_intercept": locked["fit_intercept"],
        "class_weight": locked["class_weight"],
        "tol": locked["tol"],
        "max_iter": locked["max_iter"],
    }
    official_effective = {
        **defaults,
        "random_state": locked["random_state"],
        "max_iter": locked["max_iter"],
    }
    raw_penalty = official_effective["penalty"]
    if raw_penalty == "deprecated" and official_effective.get("l1_ratio") in {0, 0.0, None}:
        # sklearn >=1.8 renamed the API default; LogisticRegression.fit resolves
        # this exact combination to l2 before checking the solver.
        official_effective["penalty"] = "l2"
    for key, expected in checks.items():
        if official_effective[key] != expected:
            raise RuntimeError(
                f"pinned ITI LogisticRegression effective default drift for {key}: "
                f"expected {expected!r}, got {official_effective[key]!r}"
            )
    return namespace["train_probes"], {
        "repository": str(repo_path),
        "commit": commit,
        "source": str(source_path),
        "function": "train_probes",
        "classifier": checks,
        "classifier_api_penalty": raw_penalty,
        "classifier_effective_penalty": official_effective["penalty"],
        "random_state": locked["random_state"],
    }


def _confiqa_groups(job: SiteJob, paths: SitePaths, protocol: SiteProtocol) -> list[tuple[str, str, str, str]]:
    require_files([paths.pairs_csv])
    prep = protocol.selector_preprocessing(job)
    if prep["pair_source"] != "same_shared_admitted_pair_artifact_as_rcm":
        raise RuntimeError("ConFiQA ITI pair source drift")
    count = int(prep["pair_groups"])
    pairs = load_actuator_pairs(paths.pairs_csv, event=job.task["downstream"]["event"])
    if len(pairs) < count:
        raise RuntimeError(f"ConFiQA ITI requires {count} admitted groups, found {len(pairs)}")
    return [
        (pair.sample_id, pair.prompt, pair.y_plus, pair.y_minus)
        for pair in pairs[:count]
    ]


def _imdb_groups(
    job: SiteJob,
    paths: SitePaths,
    protocol: SiteProtocol,
    interface: PreOInterface,
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    prep = protocol.selector_preprocessing(job)
    if prep["pair_source"] != "same_shared_good_base_generation_artifact_as_rcm":
        raise RuntimeError("IMDb ITI state-pair source drift")
    good_rows, base_rows = ensure_imdb_selector_states(
        protocol, job, paths, interface
    )
    count = int(prep["pair_groups"])
    if len(good_rows) < count or len(base_rows) < count:
        raise RuntimeError(
            f"IMDb ITI requires {count} shared good/base groups, "
            f"found good={len(good_rows)} base={len(base_rows)}"
        )
    return [
        (str(good["sample_id"]), good, base)
        for good, base in zip(good_rows[:count], base_rows[:count], strict=True)
    ]


def run_iti(
    protocol: SiteProtocol,
    job: SiteJob,
    paths: SitePaths,
    interface: PreOInterface,
) -> None:
    import numpy as np

    train_probes, official = _load_official_train_probes(protocol)
    if job.dataset == "confiqa":
        groups = _confiqa_groups(job, paths, protocol)
        source_artifact = str(paths.pairs_csv)
    else:
        groups = _imdb_groups(job, paths, protocol, interface)
        source_artifact = str(paths.selector_state_pair_manifest)
        imdb_eos_policy = protocol.selector_preprocessing(job)["terminal_eos_policy"]
        if imdb_eos_policy != (
            "exclude_terminal_eos_if_content_tokens_remain_to_match_official_prompt_last_token"
        ):
            raise RuntimeError("IMDb ITI terminal EOS policy drift")
    if len(groups) < 2:
        raise RuntimeError("ITI requires at least two complete paired groups")
    separated_activations = []
    separated_labels = []
    for index, group in enumerate(groups, start=1):
        if job.dataset == "confiqa":
            group_id, prompt, positive, negative = group
            positive_activation = interface.capture_final_all_heads(prompt, positive)
            negative_activation = interface.capture_final_all_heads(prompt, negative)
        else:
            group_id, good, base = group
            positive_activation = interface.capture_final_all_heads_ids(
                str(good["prompt"]),
                [int(value) for value in good["generated_token_ids"]],
                exclude_terminal_eos=True,
            )
            negative_activation = interface.capture_final_all_heads_ids(
                str(base["prompt"]),
                [int(value) for value in base["generated_token_ids"]],
                exclude_terminal_eos=True,
            )
        separated_activations.append(np.stack([positive_activation, negative_activation], axis=0))
        separated_labels.append(np.asarray([1, 0], dtype=np.int64))
        if index % 25 == 0 or index == len(groups):
            print(f"[site-iti] captured_groups={index}/{len(groups)} last={group_id}", flush=True)
    num_layers, num_heads, _head_dim = separated_activations[0].shape[1:]
    first_fold, second_fold = np.array_split(np.arange(len(groups)), 2)
    if len(first_fold) == 0 or len(second_fold) == 0:
        raise RuntimeError("ITI grouped two-fold split is empty")
    _probes_ab, accuracy_ab = train_probes(
        int(job.selector_spec["random_state"]),
        first_fold,
        second_fold,
        separated_activations,
        separated_labels,
        num_layers,
        num_heads,
    )
    _probes_ba, accuracy_ba = train_probes(
        int(job.selector_spec["random_state"]),
        second_fold,
        first_fold,
        separated_activations,
        separated_labels,
        num_layers,
        num_heads,
    )
    mean_accuracy = (accuracy_ab + accuracy_ba) / 2.0
    scores = []
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            flat = layer_idx * num_heads + head_idx
            scores.append(
                {
                    "component_id": f"L{layer_idx}.attn.h{head_idx}",
                    "layer_idx": layer_idx,
                    "head_idx": head_idx,
                    "fold_A_to_B_accuracy": float(accuracy_ab[flat]),
                    "fold_B_to_A_accuracy": float(accuracy_ba[flat]),
                    "ranking_score": float(mean_accuracy[flat]),
                }
            )
    scores.sort(key=lambda row: (-row["ranking_score"], row["layer_idx"], row["head_idx"]))
    selected = scores[: int(protocol.site["selection_contract"]["selected_count"])]
    paths.selector_dir.mkdir(parents=True, exist_ok=True)
    dump_csv(paths.selector_dir / "iti_head_scores.csv", scores)
    rows = []
    for rank, row in enumerate(selected, start=1):
        rows.append(
            {
                "job_id": job.job_id,
                "selector": job.selector,
                "replicate": job.replicate,
                "rank": rank,
                "component_id": row["component_id"],
                "selection_role": "unsigned",
                "ranking_score": row["ranking_score"],
                "score_name": "mean_grouped_twofold_heldout_accuracy",
                "mean_score_delta": "",
                "ci95_low": "",
                "ci95_high": "",
                "ci_crosses_zero": "",
                "n": len(groups) * 2,
                "seed": job.seed,
                "state_constructor": "official_iti_final_nonpadding_token_probe",
                "prototype_artifact": "",
                "source_artifact": source_artifact,
                "shared_scan_artifact": "",
            }
        )
    dump_csv(paths.selected_heads, rows)
    dump_json(
        paths.selector_dir / "selector_manifest.json",
        {
            **protocol.snapshot_manifest(job),
            "selected_heads": [row["component_id"] for row in selected],
            "group_count": len(groups),
            "states": len(groups) * 2,
            "fold_A_group_indices": first_fold.tolist(),
            "fold_B_group_indices": second_fold.tolist(),
            "fold_construction": "numpy.array_split over deterministic admitted group order",
            "official_implementation": official,
            "source_artifact": source_artifact,
            "state_pair_artifacts": (
                {
                    "manifest": str(paths.selector_state_pair_manifest),
                    "good": str(paths.selector_good_generations),
                    "base": str(paths.selector_base_generations),
                }
                if job.dataset == "imdb"
                else {}
            ),
            "terminal_eos_policy": (
                protocol.selector_preprocessing(job).get("terminal_eos_policy", "")
                if job.dataset == "imdb"
                else "not_applicable"
            ),
            "status": "complete",
        },
    )


def run_random(
    protocol: SiteProtocol,
    job: SiteJob,
    paths: SitePaths,
    interface: PreOInterface,
) -> None:
    universe = interface.all_heads()
    count = int(protocol.site["selection_contract"]["selected_count"])
    selected = random.Random(job.seed).sample(universe, count)
    paths.selector_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for rank, component in enumerate(selected, start=1):
        rows.append(
            {
                "job_id": job.job_id,
                "selector": job.selector,
                "replicate": job.replicate,
                "rank": rank,
                "component_id": component.component_id,
                "selection_role": "unsigned",
                "ranking_score": "",
                "score_name": "none",
                "mean_score_delta": "",
                "ci95_low": "",
                "ci95_high": "",
                "ci_crosses_zero": "",
                "n": 0,
                "seed": job.seed,
                "state_constructor": "uniform_without_replacement",
                "prototype_artifact": "",
                "source_artifact": "",
                "shared_scan_artifact": "",
            }
        )
    dump_csv(paths.selected_heads, rows)
    dump_json(
        paths.selector_dir / "selector_manifest.json",
        {
            **protocol.snapshot_manifest(job),
            "selected_heads": [component.component_id for component in selected],
            "universe_size": len(universe),
            "replacement": False,
            "stratification": "none",
            "status": "complete",
        },
    )
