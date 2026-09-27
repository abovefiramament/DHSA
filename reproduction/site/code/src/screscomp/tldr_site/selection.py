from __future__ import annotations

import ast
import csv
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

from screscomp.cecm.actuator import ActuatorPair, load_actuator_pairs
from screscomp.data import dump_csv, dump_json, dump_jsonl, load_jsonl
from screscomp.site.model import PreOInterface, SiteComponent, StateAction
from screscomp.site.statistics import (
    CandidateResult,
    candidate_result,
    rank_negative,
    rank_positive,
    result_rows,
    role_oriented_ci95_lower_bound,
)
from screscomp.tldr_site.protocol import (
    CANDIDATE_IDS,
    SELECTORS,
    TldrSiteProtocol,
    require,
    sha256_file,
)


def prepare_selector_pairs(protocol: TldrSiteProtocol) -> dict[str, Any]:
    paths = protocol.paths
    if paths.selector_manifest.is_file() and paths.selector_pairs.is_file():
        manifest = json.loads(paths.selector_manifest.read_text(encoding="utf-8"))
        require(manifest.get("config_sha256") == protocol.config_sha256, "selector-pair config hash drift")
        require(manifest.get("selector_pairs_sha256") == sha256_file(paths.selector_pairs), "selector-pair hash drift")
        require(
            int(manifest.get("selected_unique_groups", 0))
            == int(protocol.data["inputs"]["selector_groups"]["target_unique_groups"]),
            "selector-pair count drift",
        )
        return manifest

    spec = protocol.data["inputs"]["selector_groups"]
    target = int(spec["target_unique_groups"])
    paths.input_dir.mkdir(parents=True, exist_ok=True)
    csv.field_size_limit(sys.maxsize)
    selected: list[dict[str, str]] = []
    audit: list[dict[str, Any]] = []
    seen_groups: set[str] = set()
    scanned = 0
    with protocol.pairs_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        require("prompt_id" in fieldnames, "TLDR pairs lack prompt_id")
        for source_csv_row, row in enumerate(reader):
            if len(selected) >= target:
                break
            scanned += 1
            split = str(row.get("split", ""))
            prompt_id = str(row.get("prompt_id", "")).strip()
            sample_id = str(row.get("sample_id", "")).strip()
            if split != spec["source_split"]:
                audit.append({
                    "source_csv_row": source_csv_row,
                    "sample_id": sample_id,
                    "prompt_id": prompt_id,
                    "status": "skipped_split",
                    "reason": f"split={split}",
                })
                continue
            require(bool(prompt_id), f"empty prompt_id at source CSV row {source_csv_row}")
            require(bool(sample_id), f"empty sample_id at source CSV row {source_csv_row}")
            if prompt_id in seen_groups:
                audit.append({
                    "source_csv_row": source_csv_row,
                    "sample_id": sample_id,
                    "prompt_id": prompt_id,
                    "status": "skipped_duplicate_group",
                    "reason": "prompt_id_already_admitted",
                })
                continue
            require(str(row.get("admitted", "1")) not in {"0", "false", "False"}, f"unadmitted pair at row {source_csv_row}")
            require(bool(str(row.get("prompt", ""))), f"empty prompt at row {source_csv_row}")
            require(bool(str(row.get("y_plus", ""))), f"empty y_plus at row {source_csv_row}")
            require(bool(str(row.get("y_minus", ""))), f"empty y_minus at row {source_csv_row}")
            seen_groups.add(prompt_id)
            selected.append(row)
            audit.append({
                "source_csv_row": source_csv_row,
                "selected_rank": len(selected),
                "sample_id": sample_id,
                "prompt_id": prompt_id,
                "status": "admitted",
                "reason": "first_train_pair_for_unique_prompt_id",
            })

    require(len(selected) == target, f"could not materialize {target} unique TLDR selector groups")
    with paths.selector_pairs.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)
    dump_jsonl(paths.selector_admission, audit)
    manifest = {
        "protocol_id": protocol.data["protocol_id"],
        "config_sha256": protocol.config_sha256,
        "source_pairs_csv": str(protocol.pairs_path),
        "source_pairs_bytes": protocol.pairs_path.stat().st_size,
        "source_pairs_sha256": sha256_file(protocol.pairs_path),
        "source_split": spec["source_split"],
        "source_order": spec["source_order"],
        "group_key": spec["unique_group_key"],
        "pair_choice_within_group": spec["pair_choice_within_group"],
        "scanned_csv_rows": scanned,
        "selected_unique_groups": len(selected),
        "skipped_split_rows": sum(row["status"] == "skipped_split" for row in audit),
        "skipped_duplicate_groups": sum(row["status"] == "skipped_duplicate_group" for row in audit),
        "selected_sample_ids": [str(row["sample_id"]) for row in selected],
        "selected_prompt_ids": [str(row["prompt_id"]) for row in selected],
        "selector_pairs": str(paths.selector_pairs),
        "selector_pairs_sha256": sha256_file(paths.selector_pairs),
        "admission_audit": str(paths.selector_admission),
        "admission_audit_sha256": sha256_file(paths.selector_admission),
    }
    dump_json(paths.selector_manifest, manifest)
    return manifest


class AvgLogpCompetitionRunner:
    def __init__(self, interface: PreOInterface) -> None:
        self.interface = interface
        self.torch = interface.torch
        self.model = interface.model
        self.model.eval()

    def candidate_score(self, prompt: str, continuation: str, action: StateAction | None = None) -> Any:
        input_ids, prompt_len, continuation_len = self.interface.tokenize_sequence(prompt, continuation)
        positions = self.interface.decision_positions(prompt_len, continuation_len, int(input_ids.shape[-1]))
        with self.interface.action_hook(action, fixed_positions=positions):
            logits = self.model(
                input_ids=input_ids,
                attention_mask=self.torch.ones_like(input_ids),
                use_cache=False,
            ).logits
        targets = input_ids[:, prompt_len : prompt_len + continuation_len]
        predictions = logits[:, prompt_len - 1 : prompt_len + continuation_len - 1, :].float()
        log_probs = self.torch.nn.functional.log_softmax(predictions, dim=-1)
        return log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).mean()

    def margin(self, pair: ActuatorPair, action: StateAction | None = None) -> float:
        require(len(pair.y_plus_options) == 1, f"selector pair {pair.sample_id} has multiple y_plus options")
        require(len(pair.y_minus_options) == 1, f"selector pair {pair.sample_id} has multiple y_minus options")
        with self.torch.no_grad():
            plus = self.candidate_score(pair.prompt, pair.y_plus_options[0], action)
            minus = self.candidate_score(pair.prompt, pair.y_minus_options[0], action)
        return float((plus - minus).detach().cpu().item())


def _candidate(protocol: TldrSiteProtocol, component: SiteComponent, deltas: Iterable[float]) -> CandidateResult:
    bootstrap = protocol.data["selection"]["bootstrap"]
    return candidate_result(
        component.component_id,
        component.layer_idx,
        component.head_idx,
        deltas,
        bootstrap_samples=int(bootstrap["samples"]),
        bootstrap_seed=int(bootstrap["seed"]),
        interval_percentiles=tuple(float(value) for value in bootstrap["percentiles"]),
    )


def _scan(
    protocol: TldrSiteProtocol,
    components: list[SiteComponent],
    pairs: list[ActuatorPair],
    baselines: dict[str, float],
    runner: AvgLogpCompetitionRunner,
    *,
    selector: str,
    prototypes: dict[int, Any],
) -> tuple[list[CandidateResult], list[dict[str, Any]]]:
    results: list[CandidateResult] = []
    sample_rows: list[dict[str, Any]] = []
    is_zero = selector == "rcm_zero_signed"
    for component_index, component in enumerate(components, start=1):
        if is_zero:
            action = StateAction(component=component, operation="zero")
        else:
            action = StateAction(component=component, operation="clamp", prototype=prototypes[component.layer_idx])
        deltas: list[float] = []
        for pair in pairs:
            changed = runner.margin(pair, action)
            native = baselines[pair.sample_id]
            delta = native - changed if is_zero else changed - native
            deltas.append(delta)
            sample_rows.append({
                "selector": selector,
                "component_id": component.component_id,
                "layer_idx": component.layer_idx,
                "head_idx": "" if component.head_idx is None else component.head_idx,
                "sample_id": pair.sample_id,
                "native_competition_score": native,
                "changed_competition_score": changed,
                "delta": delta,
            })
        result = _candidate(protocol, component, deltas)
        results.append(result)
        print(
            f"[tldr-site-rcm] selector={selector} candidate={component_index}/{len(components)} "
            f"id={component.component_id} mean={result.mean_score_delta:.8f} "
            f"oriented_ci_low={role_oriented_ci95_lower_bound(result):.8f}",
            flush=True,
        )
    return results, sample_rows


def _load_or_create_baselines(
    protocol: TldrSiteProtocol,
    pairs: list[ActuatorPair],
    runner: AvgLogpCompetitionRunner,
) -> dict[str, float]:
    paths = protocol.paths
    sample_ids = [pair.sample_id for pair in pairs]
    if paths.shared_native_margins.is_file() and paths.shared_native_manifest.is_file():
        manifest = json.loads(paths.shared_native_manifest.read_text(encoding="utf-8"))
        rows = load_jsonl(paths.shared_native_margins)
        require(manifest.get("config_sha256") == protocol.config_sha256, "native-margin config drift")
        require(manifest.get("selector_pairs_sha256") == sha256_file(paths.selector_pairs), "native-margin input drift")
        require([str(row["sample_id"]) for row in rows] == sample_ids, "native-margin sample order drift")
        return {str(row["sample_id"]): float(row["native_competition_score"]) for row in rows}
    paths.shared_native_margins.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, pair in enumerate(pairs, start=1):
        value = runner.margin(pair)
        rows.append({"sample_id": pair.sample_id, "native_competition_score": value})
        if index % 25 == 0 or index == len(pairs):
            print(f"[tldr-site-rcm] native_margins={index}/{len(pairs)}", flush=True)
    dump_jsonl(paths.shared_native_margins, rows)
    dump_json(paths.shared_native_manifest, {
        "protocol_id": protocol.data["protocol_id"],
        "config_sha256": protocol.config_sha256,
        "selector_pairs": str(paths.selector_pairs),
        "selector_pairs_sha256": sha256_file(paths.selector_pairs),
        "score": protocol.data["selection"]["score"],
        "actual_n": len(rows),
        "artifact": str(paths.shared_native_margins),
        "artifact_sha256": sha256_file(paths.shared_native_margins),
    })
    return {str(row["sample_id"]): float(row["native_competition_score"]) for row in rows}


def _target_prototypes(
    protocol: TldrSiteProtocol,
    pairs: list[ActuatorPair],
    interface: PreOInterface,
    out_dir: Path,
) -> tuple[dict[int, Any], Path]:
    torch = interface.torch
    sums: dict[int, Any] = {}
    token_counts: list[dict[str, Any]] = []
    for index, pair in enumerate(pairs, start=1):
        require(len(pair.y_plus_options) == 1, f"prototype pair {pair.sample_id} has multiple y_plus options")
        maps, tokens = interface.collect_layer_sequence_means(pair.prompt, pair.y_plus_options[0])
        for layer_idx, vector in maps.items():
            value = vector.float()
            sums[layer_idx] = value.clone() if layer_idx not in sums else sums[layer_idx] + value
        token_counts.append({"sample_id": pair.sample_id, "decision_states": tokens})
        if index % 25 == 0 or index == len(pairs):
            print(f"[tldr-site-patch] prototype_sequences={index}/{len(pairs)}", flush=True)
    require(len(sums) == interface.backend.num_layers, "incomplete TLDR patch prototypes")
    prototypes = {layer_idx: value / float(len(pairs)) for layer_idx, value in sums.items()}
    artifact = out_dir / "trajectory_mean_states.safetensors"
    try:
        from safetensors.torch import save_file
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("TLDR patch prototype storage requires safetensors") from exc
    save_file({f"layer_{layer_idx}": value.to(dtype=torch.float32).contiguous() for layer_idx, value in prototypes.items()}, str(artifact))
    dump_json(out_dir / "trajectory_mean_manifest.json", {
        "protocol_id": protocol.data["protocol_id"],
        "config_sha256": protocol.config_sha256,
        "source": "complete_y_plus_decision_trajectory_on_shared_300_groups",
        "aggregation": protocol.selectors["rcm_patch_signed"]["prototype_aggregation"],
        "dtype": "float32",
        "groups": len(pairs),
        "sequence_decision_states": token_counts,
        "artifact": str(artifact),
        "artifact_sha256": sha256_file(artifact),
    })
    return prototypes, artifact


def _ranked_sample_indices(size: int, count: int, seed: int, *, salt: int = 0) -> list[int]:
    require(size >= count, f"candidate reservoir {size} is smaller than requested sample {count}")
    picked = random.Random(seed + salt * 1_000_003).sample(range(size), count)
    return sorted(picked)


def _unique_alternative(
    pools: list[list[str]],
    counts: list[int],
    seed: int,
    existing: set[tuple[str, ...]],
) -> list[str]:
    for salt in range(100):
        selected: list[str] = []
        for pool_index, (pool, count) in enumerate(zip(pools, counts, strict=True)):
            indices = _ranked_sample_indices(len(pool), count, seed + pool_index * 97_531, salt=salt)
            selected.extend(pool[index] for index in indices)
        key = tuple(selected)
        if len(set(selected)) == sum(counts) and key not in existing:
            return selected
    raise RuntimeError("could not form three unique complete candidate configurations without fallback")


def _write_candidate_configurations(
    protocol: TldrSiteProtocol,
    selector: str,
    configurations: list[dict[str, Any]],
    score_lookup: dict[str, dict[str, Any]],
    *,
    reservoirs: dict[str, list[str]],
    selector_manifest: dict[str, Any],
) -> None:
    require([row["candidate_id"] for row in configurations] == list(CANDIDATE_IDS), "candidate ID drift")
    rows: list[dict[str, Any]] = []
    selected_count = int(protocol.data["selection"]["selected_count"])
    for config in configurations:
        heads = list(config["heads"])
        roles = list(config["roles"])
        require(
            len(heads) == selected_count and len(set(heads)) == selected_count,
            f"{selector} {config['candidate_id']} is not distinct K={selected_count}",
        )
        for rank, (head, role) in enumerate(zip(heads, roles, strict=True), start=1):
            protocol.validate_component_id(head)
            score = score_lookup.get(head, {})
            rows.append({
                "selector": selector,
                "candidate_id": config["candidate_id"],
                "candidate_rank": int(config["candidate_index"]),
                "head_rank": rank,
                "component_id": head,
                "selection_role": role,
                "construction": config["construction"],
                "seed": config.get("seed", ""),
                "ranking_score": score.get("ranking_score", ""),
                "mean_score_delta": score.get("mean_score_delta", ""),
                "ci95_low": score.get("ci95_low", ""),
                "ci95_high": score.get("ci95_high", ""),
                "ci_crosses_zero": score.get("ci_crosses_zero", ""),
                "probe_fold_A_to_B_accuracy": score.get("fold_A_to_B_accuracy", ""),
                "probe_fold_B_to_A_accuracy": score.get("fold_B_to_A_accuracy", ""),
            })
    out_dir = protocol.paths.selector_dir(selector)
    dump_csv(protocol.paths.candidates_csv(selector), rows)
    payload = {
        **protocol.snapshot(),
        "selector": selector,
        "selector_manifest": selector_manifest,
        "reservoirs": reservoirs,
        "configurations": configurations,
        "candidate_rows": str(protocol.paths.candidates_csv(selector)),
        "candidate_rows_sha256": sha256_file(protocol.paths.candidates_csv(selector)),
        "status": "complete",
    }
    dump_json(protocol.paths.candidates_json(selector), payload)
    dump_json(out_dir / "selector_manifest.json", payload)


def _rcm_configurations(
    protocol: TldrSiteProtocol,
    selector: str,
    head_results: list[CandidateResult],
    selector_manifest: dict[str, Any],
) -> None:
    reservoir_count = int(protocol.data["common_position_audit"]["ranked_reservoir_sizes"]["rcm_per_role"])
    positive = rank_positive(head_results, reservoir_count)
    negative = rank_negative(head_results, reservoir_count)
    require(len(positive) == reservoir_count, f"{selector} has fewer than {reservoir_count} target-support reservoir heads")
    require(len(negative) == reservoir_count, f"{selector} has fewer than {reservoir_count} competitor-support reservoir heads")
    positive_ids = [row.component_id for row in positive]
    negative_ids = [row.component_id for row in negative]
    partition = protocol.data["selection"]["signed_partition"]
    positive_count = int(partition["target_support_count"])
    negative_count = int(partition["competitor_support_count"])
    raw = [*positive_ids[:positive_count], *negative_ids[:negative_count]]
    configurations = [{
        "candidate_id": "candidate_00",
        "candidate_index": 0,
        "heads": raw,
        "roles": ["target_support"] * positive_count + ["competitor_support"] * negative_count,
        "construction": "direct_raw_selector_ranking",
        "seed": 42,
    }]
    existing = {tuple(raw)}
    for candidate_index, seed in enumerate(protocol.data["common_position_audit"]["alternative_seeds"], start=1):
        heads = _unique_alternative(
            [positive_ids, negative_ids], [positive_count, negative_count], int(seed), existing,
        )
        existing.add(tuple(heads))
        configurations.append({
            "candidate_id": f"candidate_{candidate_index:02d}",
            "candidate_index": candidate_index,
            "heads": heads,
            "roles": ["target_support"] * positive_count + ["competitor_support"] * negative_count,
            "construction": f"fixed_seed_sample_without_replacement_from_top{reservoir_count}_per_role",
            "seed": seed,
        })
    score_lookup = {
        row.component_id: {
            "ranking_score": role_oriented_ci95_lower_bound(row),
            "mean_score_delta": row.mean_score_delta,
            "ci95_low": row.ci95_low,
            "ci95_high": row.ci95_high,
            "ci_crosses_zero": row.ci95_low <= 0.0 <= row.ci95_high,
        }
        for row in head_results
    }
    _write_candidate_configurations(
        protocol,
        selector,
        configurations,
        score_lookup,
        reservoirs={"target_support": positive_ids, "competitor_support": negative_ids},
        selector_manifest=selector_manifest,
    )


def run_rcm_selector(
    protocol: TldrSiteProtocol,
    selector: str,
    interface: PreOInterface,
) -> None:
    require(selector in {"rcm_zero_signed", "rcm_patch_signed"}, f"not an RCM selector: {selector}")
    out_dir = protocol.paths.selector_dir(selector)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = load_actuator_pairs(protocol.paths.selector_pairs, event=protocol.data["inputs"]["event"])
    target_groups = int(protocol.data["inputs"]["selector_groups"]["target_unique_groups"])
    require(len(pairs) == target_groups, f"{selector} requires exactly {target_groups} shared pairs")
    runner = AvgLogpCompetitionRunner(interface)
    baselines = _load_or_create_baselines(protocol, pairs, runner)
    prototypes: dict[int, Any] = {}
    prototype_artifact = ""
    if selector == "rcm_patch_signed":
        prototypes, artifact = _target_prototypes(protocol, pairs, interface, out_dir)
        prototype_artifact = str(artifact)

    layers = [SiteComponent(layer_idx) for layer_idx in range(interface.backend.num_layers)]
    layer_results, layer_samples = _scan(
        protocol, layers, pairs, baselines, runner,
        selector=selector, prototypes=prototypes,
    )
    dump_csv(out_dir / "layer_results.csv", result_rows(layer_results))
    dump_jsonl(out_dir / "layer_delta_samples.jsonl", layer_samples)
    beam = int(protocol.data["selection"]["coarse_to_fine"]["coarse_beam_per_role"])
    positive_layers = rank_positive(layer_results, beam)
    negative_layers = rank_negative(layer_results, beam)
    require(len(positive_layers) == beam, f"{selector} cannot form four positive parent layers")
    require(len(negative_layers) == beam, f"{selector} cannot form four negative parent layers")
    layer_indices = [row.layer_idx for row in [*positive_layers, *negative_layers]]
    require(
        len(set(layer_indices)) == beam * 2,
        f"{selector} signed parent layer union is not {beam * 2} distinct layers",
    )
    dump_csv(out_dir / "positive_layer_beam.csv", result_rows(positive_layers))
    dump_csv(out_dir / "negative_layer_beam.csv", result_rows(negative_layers))

    heads: list[SiteComponent] = []
    for layer_idx in layer_indices:
        _hidden, num_heads, _head_dim = interface.geometry(layer_idx)
        heads.extend(SiteComponent(layer_idx, head_idx) for head_idx in range(num_heads))
    head_results, head_samples = _scan(
        protocol, heads, pairs, baselines, runner,
        selector=selector, prototypes=prototypes,
    )
    dump_csv(out_dir / "head_results.csv", result_rows(head_results))
    dump_jsonl(out_dir / "head_delta_samples.jsonl", head_samples)
    selector_manifest = {
        "protocol_id": protocol.data["protocol_id"],
        "config_sha256": protocol.config_sha256,
        "selector": selector,
        "source_pairs": str(protocol.paths.selector_pairs),
        "source_pairs_sha256": sha256_file(protocol.paths.selector_pairs),
        "actual_groups": len(pairs),
        "state_timing": protocol.data["selection"]["state_timing"],
        "score": protocol.data["selection"]["score"],
        "operation": protocol.selectors[selector]["operation"],
        "prototype_artifact": prototype_artifact,
        "positive_parent_layers": [row.component_id for row in positive_layers],
        "negative_parent_layers": [row.component_id for row in negative_layers],
        "heads_scanned": len(head_results),
        "head_role_source": "candidate_head_own_mean_paired_effect",
        "ci_admission_gate": False,
        "status": "scored",
    }
    _rcm_configurations(protocol, selector, head_results, selector_manifest)


def _load_official_train_probes(protocol: TldrSiteProtocol):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from tqdm import tqdm

    spec = protocol.selectors["iti"]
    repo = Path(spec["official_repository"])
    source = repo / "utils.py"
    require(source.is_file(), f"missing pinned honest_llama source: {source}")
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    require(commit == spec["official_commit"], f"honest_llama commit drift: {commit}")
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "train_probes"]
    require(len(matches) == 1, "could not isolate official honest_llama train_probes")
    module = ast.Module(body=matches, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {
        "np": np,
        "tqdm": tqdm,
        "LogisticRegression": LogisticRegression,
        "accuracy_score": accuracy_score,
    }
    exec(compile(module, str(source), "exec"), namespace)
    defaults = LogisticRegression().get_params()
    effective = {**defaults, "random_state": spec["random_state"], "max_iter": spec["max_iter"]}
    if effective.get("penalty") == "deprecated" and effective.get("l1_ratio") in {0, 0.0, None}:
        effective["penalty"] = "l2"
    for key in ("penalty", "C", "solver", "fit_intercept", "class_weight", "tol", "max_iter"):
        require(effective[key] == spec[key], f"official ITI classifier drift: {key}")
    return namespace["train_probes"], {
        "repository": str(repo),
        "commit": commit,
        "source": str(source),
        "source_sha256": sha256_file(source),
    }


def run_iti_selector(protocol: TldrSiteProtocol, interface: PreOInterface) -> None:
    import numpy as np

    selector = "iti"
    out_dir = protocol.paths.selector_dir(selector)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = load_actuator_pairs(protocol.paths.selector_pairs, event=protocol.data["inputs"]["event"])
    target_groups = int(protocol.data["inputs"]["selector_groups"]["target_unique_groups"])
    require(len(pairs) == target_groups, f"ITI requires exactly {target_groups} shared groups")
    train_probes, official = _load_official_train_probes(protocol)
    activations = []
    labels = []
    for index, pair in enumerate(pairs, start=1):
        require(len(pair.y_plus_options) == 1 and len(pair.y_minus_options) == 1, f"ITI pair shape drift: {pair.sample_id}")
        positive = interface.capture_final_all_heads(pair.prompt, pair.y_plus_options[0])
        negative = interface.capture_final_all_heads(pair.prompt, pair.y_minus_options[0])
        activations.append(np.stack([positive, negative], axis=0))
        labels.append(np.asarray([1, 0], dtype=np.int64))
        if index % 25 == 0 or index == len(pairs):
            print(f"[tldr-site-iti] captured_groups={index}/{len(pairs)}", flush=True)
    num_layers, num_heads, _head_dim = activations[0].shape[1:]
    require(target_groups % 2 == 0, "ITI grouped two-fold split requires an even group count")
    midpoint = target_groups // 2
    first_fold = np.arange(0, midpoint)
    second_fold = np.arange(midpoint, target_groups)
    spec = protocol.selectors["iti"]
    _probe_ab, accuracy_ab = train_probes(
        int(spec["random_state"]), first_fold, second_fold,
        activations, labels, num_layers, num_heads,
    )
    _probe_ba, accuracy_ba = train_probes(
        int(spec["random_state"]), second_fold, first_fold,
        activations, labels, num_layers, num_heads,
    )
    scores: list[dict[str, Any]] = []
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            flat = layer_idx * num_heads + head_idx
            scores.append({
                "component_id": f"L{layer_idx}.attn.h{head_idx}",
                "layer_idx": layer_idx,
                "head_idx": head_idx,
                "fold_A_to_B_accuracy": float(accuracy_ab[flat]),
                "fold_B_to_A_accuracy": float(accuracy_ba[flat]),
                "ranking_score": float((accuracy_ab[flat] + accuracy_ba[flat]) / 2.0),
            })
    scores.sort(key=lambda row: (-float(row["ranking_score"]), int(row["layer_idx"]), int(row["head_idx"])))
    dump_csv(out_dir / "iti_head_scores.csv", scores)
    reservoir_size = int(protocol.data["common_position_audit"]["ranked_reservoir_sizes"]["iti_unsigned"])
    reservoir = [str(row["component_id"]) for row in scores[:reservoir_size]]
    selected_count = int(protocol.data["selection"]["selected_count"])
    raw = reservoir[:selected_count]
    configurations = [{
        "candidate_id": "candidate_00",
        "candidate_index": 0,
        "heads": raw,
        "roles": ["unsigned"] * selected_count,
        "construction": "direct_raw_selector_ranking",
        "seed": int(spec["random_state"]),
    }]
    existing = {tuple(raw)}
    for candidate_index, seed in enumerate(protocol.data["common_position_audit"]["alternative_seeds"], start=1):
        heads = _unique_alternative([reservoir], [selected_count], int(seed), existing)
        existing.add(tuple(heads))
        configurations.append({
            "candidate_id": f"candidate_{candidate_index:02d}",
            "candidate_index": candidate_index,
            "heads": heads,
            "roles": ["unsigned"] * selected_count,
            "construction": f"fixed_seed_sample_without_replacement_from_top{reservoir_size}_probe_heads",
            "seed": seed,
        })
    score_lookup = {str(row["component_id"]): row for row in scores}
    selector_manifest = {
        "protocol_id": protocol.data["protocol_id"],
        "config_sha256": protocol.config_sha256,
        "selector": selector,
        "source_pairs": str(protocol.paths.selector_pairs),
        "source_pairs_sha256": sha256_file(protocol.paths.selector_pairs),
        "groups": len(pairs),
        "states": len(pairs) * 2,
        "positive_state": spec["positive_state"],
        "negative_state": spec["negative_state"],
        "fold_A_group_indices": first_fold.tolist(),
        "fold_B_group_indices": second_fold.tolist(),
        "official_implementation": official,
        "status": "scored",
    }
    _write_candidate_configurations(
        protocol, selector, configurations, score_lookup,
        reservoirs={"unsigned": reservoir}, selector_manifest=selector_manifest,
    )


def run_random_selector(protocol: TldrSiteProtocol) -> None:
    selector = "random"
    out_dir = protocol.paths.selector_dir(selector)
    out_dir.mkdir(parents=True, exist_ok=True)
    architecture = protocol.data["model"]["expected_architecture"]
    universe = [
        f"L{layer_idx}.attn.h{head_idx}"
        for layer_idx in range(int(architecture["attention_layers"]))
        for head_idx in range(int(architecture["attention_heads_per_layer"]))
    ]
    require(len(universe) == int(architecture["candidate_heads"]), "Random universe size drift")
    selected_count = int(protocol.data["selection"]["selected_count"])
    configurations = []
    for candidate_index, seed in enumerate(protocol.selectors["random"]["seeds"]):
        heads = random.Random(int(seed)).sample(universe, selected_count)
        configurations.append({
            "candidate_id": f"candidate_{candidate_index:02d}",
            "candidate_index": candidate_index,
            "heads": heads,
            "roles": ["unsigned"] * selected_count,
            "construction": "uniform_without_replacement_from_all_compatible_heads",
            "seed": int(seed),
        })
    selector_manifest = {
        "protocol_id": protocol.data["protocol_id"],
        "config_sha256": protocol.config_sha256,
        "selector": selector,
        "universe_size": len(universe),
        "replacement": False,
        "stratification": "none",
        "seeds": protocol.selectors["random"]["seeds"],
        "status": "scored",
    }
    _write_candidate_configurations(
        protocol, selector, configurations, {},
        reservoirs={"unsigned": universe}, selector_manifest=selector_manifest,
    )


def validate_model_geometry(protocol: TldrSiteProtocol, interface: PreOInterface) -> None:
    expected = protocol.data["model"]["expected_architecture"]
    require(interface.backend.num_layers == int(expected["attention_layers"]), "loaded model layer count drift")
    for layer_idx in range(interface.backend.num_layers):
        hidden, heads, head_dim = interface.geometry(layer_idx)
        require(hidden == int(expected["hidden_size"]), f"L{layer_idx} hidden size drift")
        require(heads == int(expected["attention_heads_per_layer"]), f"L{layer_idx} head count drift")
        require(head_dim == int(expected["head_dim"]), f"L{layer_idx} head dimension drift")


def run_selector(
    protocol: TldrSiteProtocol,
    selector: str,
    interface: PreOInterface | None,
) -> None:
    require(selector in SELECTORS, f"unknown TLDR Site selector: {selector}")
    prepare_selector_pairs(protocol)
    if selector == "random":
        run_random_selector(protocol)
        return
    require(interface is not None, f"{selector} requires a loaded model interface")
    validate_model_geometry(protocol, interface)
    if selector == "iti":
        run_iti_selector(protocol, interface)
    else:
        run_rcm_selector(protocol, selector, interface)
