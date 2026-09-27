from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from screscomp.cecm.actuator import ActuatorPair, load_actuator_pairs
from screscomp.data import dump_csv, dump_json, dump_jsonl, load_jsonl
from screscomp.site.imdb_states import (
    ensure_imdb_selector_states,
    generate_imdb_condition,
    imdb_condition_rows,
)
from screscomp.site.model import ClosedCompetitionRunner, PreOInterface, SiteComponent, StateAction
from screscomp.site.protocol import SiteJob, SitePaths, SiteProtocol, require_files
from screscomp.site.sentiment import FrozenSentimentScorer
from screscomp.site.statistics import (
    CandidateResult,
    candidate_result,
    rank_negative,
    rank_positive,
    result_rows,
    role_oriented_ci95_lower_bound,
)


ZERO_SELECTORS = frozenset({"rcm_zero_signed"})
PATCH_SELECTOR = "rcm_patch_signed"


def _is_zero(selector: str) -> bool:
    return selector in ZERO_SELECTORS


def _is_patch(selector: str) -> bool:
    return selector == PATCH_SELECTOR


def _state_constructor(selector: str) -> str:
    if _is_zero(selector):
        return "zero_at_every_generation_decision_state"
    if _is_patch(selector):
        return "trajectory_mean_state_clamp_at_every_generation_decision_state"
    raise ValueError(f"unknown RCM selector: {selector}")


def _action(selector: str, component: SiteComponent, prototypes: dict[int, Any]) -> StateAction:
    if _is_zero(selector):
        return StateAction(component=component, operation="zero")
    if _is_patch(selector):
        return StateAction(component=component, operation="clamp", prototype=prototypes[component.layer_idx])
    raise ValueError(f"unknown RCM selector: {selector}")

def _candidate(
    protocol: SiteProtocol,
    component: SiteComponent,
    deltas: list[float],
) -> CandidateResult:
    common = protocol.site["rcm_common"]
    return candidate_result(
        component.component_id,
        component.layer_idx,
        component.head_idx,
        deltas,
        bootstrap_samples=int(common["bootstrap_samples"]),
        bootstrap_seed=int(common["bootstrap_seed"]),
        interval_percentiles=tuple(float(value) for value in common["bootstrap_interval_percentiles"]),
    )


def _scan_components(
    protocol: SiteProtocol,
    components: list[SiteComponent],
    score_component: Callable[[SiteComponent], list[float]],
) -> list[CandidateResult]:
    results: list[CandidateResult] = []
    for index, component in enumerate(components, start=1):
        deltas = score_component(component)
        result = _candidate(protocol, component, deltas)
        results.append(result)
        print(
            f"[site-rcm] candidate={index}/{len(components)} id={component.component_id} "
            f"mean_delta={result.mean_score_delta:.8f} "
            f"oriented_ci95_lower={role_oriented_ci95_lower_bound(result):.8f} n={len(deltas)}",
            flush=True,
        )
    return results



def _head_pool(*groups: list[CandidateResult]) -> list[CandidateResult]:
    pool: list[CandidateResult] = []
    seen: set[str] = set()
    for group in groups:
        for result in group:
            if result.head_idx is None:
                raise ValueError(f"head refinement received a layer result: {result.component_id}")
            if result.component_id in seen:
                raise ValueError(f"duplicate head across signed layer beams: {result.component_id}")
            seen.add(result.component_id)
            pool.append(result)
    return pool


def _rank_heads_by_own_effect(
    positive_parent_results: list[CandidateResult],
    negative_parent_results: list[CandidateResult],
    positive_count: int,
    negative_count: int,
) -> tuple[list[CandidateResult], list[CandidateResult]]:
    pool = _head_pool(positive_parent_results, negative_parent_results)
    return rank_positive(pool, positive_count), rank_negative(pool, negative_count)


def _save_prototypes(
    path: Path,
    prototypes: dict[int, Any],
    manifest: dict[str, Any],
) -> None:
    try:
        from safetensors.torch import save_file
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("RCM-patch requires safetensors") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({f"L{layer}.attn": vector.float().contiguous() for layer, vector in prototypes.items()}, str(path))
    dump_json(path.with_name("trajectory_mean_manifest.json"), manifest)


def _mean_layer_maps(sample_maps: list[dict[int, Any]]) -> dict[int, Any]:
    if not sample_maps:
        raise ValueError("no target trajectories were admitted for RCM-patch")
    layers = sorted(sample_maps[0])
    return {
        layer: sum((sample[layer].float() for sample in sample_maps[1:]), sample_maps[0][layer].float().clone())
        / len(sample_maps)
        for layer in layers
    }


def _candidate_artifact_rows(results: list[CandidateResult]) -> list[dict[str, Any]]:
    return [
        {
            "component_id": row.component_id,
            "layer_idx": row.layer_idx,
            "head_idx": row.head_idx,
            "deltas": list(row.deltas),
            "mean_score_delta": row.mean_score_delta,
            "ci95_low": row.ci95_low,
            "ci95_high": row.ci95_high,
            "role_oriented_ci95_lower_bound": role_oriented_ci95_lower_bound(row),
            "n": len(row.deltas),
        }
        for row in results
    ]


def _load_candidate_artifact(path: Path) -> list[CandidateResult]:
    rows = load_jsonl(path)
    out: list[CandidateResult] = []
    for row in rows:
        deltas = tuple(float(value) for value in row["deltas"])
        if int(row["n"]) != len(deltas):
            raise RuntimeError(f"candidate n drift in {path}: {row['component_id']}")
        actual_mean = mean(deltas)
        if abs(actual_mean - float(row["mean_score_delta"])) > 1e-12:
            raise RuntimeError(f"candidate mean drift in {path}: {row['component_id']}")
        out.append(
            CandidateResult(
                component_id=str(row["component_id"]),
                layer_idx=int(row["layer_idx"]),
                head_idx=None if row.get("head_idx") is None else int(row["head_idx"]),
                deltas=deltas,
                mean_score_delta=float(row["mean_score_delta"]),
                ci95_low=float(row["ci95_low"]),
                ci95_high=float(row["ci95_high"]),
            )
        )
    return out


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _shared_zero_paths(paths: SitePaths) -> dict[str, Path]:
    root = paths.shared_zero_scan_dir
    return {
        "layer_results": root / "layer_results.jsonl",
        "positive_layer_beam": root / "positive_layer_beam.jsonl",
        "negative_layer_beam": root / "negative_layer_beam.jsonl",
        "positive_head_results": root / "positive_head_results.jsonl",
        "negative_head_results": root / "negative_head_results.jsonl",
        "manifest": root / "scan_manifest.json",
    }


def _shared_zero_contract(
    protocol: SiteProtocol,
    job: SiteJob,
    paths: SitePaths,
    *,
    source_artifact: str,
) -> dict[str, Any]:
    return {
        "protocol_id": protocol.data["protocol_id"],
        "protocol_version": protocol.data["version"],
        "config": str(protocol.config_path),
        "config_sha256": protocol.config_sha256,
        "scan_family": "rcm_zero",
        "dataset": job.dataset,
        "subset": job.subset,
        "model_key": job.model_key,
        "model": job.model,
        "model_revision": job.model_revision,
        "use_chat_template": job.use_chat_template,
        "selection_contract": protocol.site["selection_contract"],
        "decision_trajectory": protocol.site["decision_trajectory"],
        "rcm_common": protocol.site["rcm_common"],
        "selector_preprocessing": protocol.selector_preprocessing(job),
        "task_states": job.task["states"],
        "task_score": job.task["score"],
        "source_artifact": source_artifact,
    }


def _scan_sha256(artifact_sha256: dict[str, str]) -> str:
    payload = json.dumps(artifact_sha256, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_shared_zero_scan(
    protocol: SiteProtocol,
    job: SiteJob,
    paths: SitePaths,
    *,
    source_artifact: str,
) -> dict[str, Any] | None:
    artifacts = _shared_zero_paths(paths)
    manifest_path = artifacts["manifest"]
    if not manifest_path.is_file():
        partial = [str(path) for path in artifacts.values() if path.is_file()]
        incomplete = paths.shared_zero_scan_dir / "incomplete.json"
        if incomplete.is_file():
            partial.append(str(incomplete))
        if partial:
            raise RuntimeError(f"unregistered partial shared zero scan: {partial}")
        return None
    missing = [str(path) for key, path in artifacts.items() if key != "manifest" and not path.is_file()]
    if missing:
        raise RuntimeError(f"registered shared zero scan is incomplete: {missing}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = _shared_zero_contract(protocol, job, paths, source_artifact=source_artifact)
    if manifest.get("scan_contract") != expected:
        raise RuntimeError("shared zero scan contract drift")
    if manifest.get("status") != "complete":
        raise RuntimeError("shared zero scan is not complete")
    hashes = {
        key: _sha256(path)
        for key, path in artifacts.items()
        if key != "manifest"
    }
    if manifest.get("artifact_sha256") != hashes:
        raise RuntimeError("shared zero scan artifact hash drift")
    scan_sha = _scan_sha256(hashes)
    if manifest.get("scan_sha256") != scan_sha:
        raise RuntimeError("shared zero scan digest drift")
    return {
        **{key: _load_candidate_artifact(path) for key, path in artifacts.items() if key != "manifest"},
        "manifest": manifest,
        "manifest_path": manifest_path,
        "scan_sha256": scan_sha,
    }


def _record_shared_incomplete(
    protocol: SiteProtocol,
    job: SiteJob,
    paths: SitePaths,
    *,
    source_artifact: str,
    stage: str,
    counts: dict[str, int],
) -> None:
    paths.shared_zero_scan_dir.mkdir(parents=True, exist_ok=True)
    dump_json(
        paths.shared_zero_scan_dir / "incomplete.json",
        {
            "scan_contract": _shared_zero_contract(
                protocol, job, paths, source_artifact=source_artifact
            ),
            "stage": stage,
            "counts": counts,
            "status": "incomplete_no_fallback",
        },
    )


def _save_shared_zero_scan(
    protocol: SiteProtocol,
    job: SiteJob,
    paths: SitePaths,
    scan: dict[str, list[CandidateResult]],
    *,
    source_artifact: str,
) -> dict[str, Any]:
    artifacts = _shared_zero_paths(paths)
    paths.shared_zero_scan_dir.mkdir(parents=True, exist_ok=False)
    for key, results in scan.items():
        dump_jsonl(artifacts[key], _candidate_artifact_rows(results))
        dump_csv(paths.shared_zero_scan_dir / f"{key}.csv", result_rows(results))
    hashes = {key: _sha256(path) for key, path in artifacts.items() if key != "manifest"}
    scan_sha = _scan_sha256(hashes)
    repo_root = protocol.config_path.parent.parent
    commit = subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(repo_root), "status", "--short"], text=True
    ).splitlines()
    manifest = {
        "scan_contract": _shared_zero_contract(
            protocol, job, paths, source_artifact=source_artifact
        ),
        "producer_job_id": job.job_id,
        "artifact_sha256": hashes,
        "scan_sha256": scan_sha,
        "code_commit": commit,
        "dirty_tree_manifest": dirty,
        "ci_role": "report_only",
        "status": "complete",
    }
    dump_json(artifacts["manifest"], manifest)
    return {
        **scan,
        "manifest": manifest,
        "manifest_path": artifacts["manifest"],
        "scan_sha256": scan_sha,
    }


def _build_shared_zero_scan(
    protocol: SiteProtocol,
    job: SiteJob,
    paths: SitePaths,
    interface: PreOInterface,
    score_component: Callable[[SiteComponent], list[float]],
    *,
    source_artifact: str,
) -> dict[str, Any]:
    beam_width = int(protocol.site["rcm_common"]["coarse_beam_width"])
    partition = protocol.site["selection_contract"]["signed_partition"]
    positive_count = int(partition["target_support_count"])
    negative_count = int(partition["competitor_support_count"])
    layers = [SiteComponent(layer_idx) for layer_idx in range(interface.backend.num_layers)]
    layer_results = _scan_components(protocol, layers, score_component)
    positive_layers = rank_positive(layer_results, beam_width)
    negative_layers = rank_negative(layer_results, beam_width)
    if len(positive_layers) != beam_width or len(negative_layers) != beam_width:
        counts = {"positive_layers": len(positive_layers), "negative_layers": len(negative_layers)}
        _record_shared_incomplete(
            protocol, job, paths, source_artifact=source_artifact,
            stage="coarse_layer", counts=counts,
        )
        raise RuntimeError(f"shared zero layer scan cannot form the signed beams: {counts}")

    def heads(layers_to_refine: list[CandidateResult]) -> list[SiteComponent]:
        components: list[SiteComponent] = []
        for layer in layers_to_refine:
            _hidden, num_heads, _head_dim = interface.geometry(layer.layer_idx)
            components.extend(
                SiteComponent(layer.layer_idx, head_idx) for head_idx in range(num_heads)
            )
        return components

    positive_heads = _scan_components(protocol, heads(positive_layers), score_component)
    negative_heads = _scan_components(protocol, heads(negative_layers), score_component)
    admitted_positive, admitted_negative = _rank_heads_by_own_effect(
        positive_heads, negative_heads, positive_count, negative_count
    )
    if len(admitted_positive) != positive_count or len(admitted_negative) != negative_count:
        counts = {
            "positive_heads": len(admitted_positive),
            "negative_heads": len(admitted_negative),
        }
        _record_shared_incomplete(
            protocol, job, paths, source_artifact=source_artifact,
            stage="head_refine", counts=counts,
        )
        raise RuntimeError(f"shared zero head scan cannot form the registered selections: {counts}")
    return _save_shared_zero_scan(
        protocol,
        job,
        paths,
        {
            "layer_results": layer_results,
            "positive_layer_beam": positive_layers,
            "negative_layer_beam": negative_layers,
            "positive_head_results": positive_heads,
            "negative_head_results": negative_heads,
        },
        source_artifact=source_artifact,
    )


def _materialize_zero_selection(
    protocol: SiteProtocol,
    job: SiteJob,
    paths: SitePaths,
    scan: dict[str, Any],
    *,
    source_artifact: str,
) -> None:
    selected_count = int(protocol.site["selection_contract"]["selected_count"])
    partition = protocol.site["selection_contract"]["signed_partition"]
    if job.selector != "rcm_zero_signed":
        raise ValueError(f"not a zero selector: {job.selector}")
    positive, negative = _rank_heads_by_own_effect(
        scan["positive_head_results"], scan["negative_head_results"],
        int(partition["target_support_count"]), int(partition["competitor_support_count"]),
    )
    selected = [
        *((row, "target_support") for row in positive),
        *((row, "competitor_support") for row in negative),
    ]
    if len(selected) != selected_count:
        raise RuntimeError(f"{job.selector} did not materialize exactly {selected_count} heads")
    _write_rcm_outputs(
        protocol,
        job,
        paths,
        scan["layer_results"],
        {
            "target_support": scan["positive_layer_beam"],
            "competitor_support": scan["negative_layer_beam"],
        },
        {
            "target_support": scan["positive_head_results"],
            "competitor_support": scan["negative_head_results"],
        },
        selected,
        prototype_artifact="",
        source_artifact=source_artifact,
        shared_scan_artifact=str(scan["manifest_path"]),
        shared_scan_sha256=str(scan["scan_sha256"]),
    )


def _run_patch_signed_selection(
    protocol: SiteProtocol,
    job: SiteJob,
    paths: SitePaths,
    interface: PreOInterface,
    score_component: Callable[[SiteComponent], list[float]],
    *,
    prototype_artifact: str,
    source_artifact: str,
) -> None:
    beam_width = int(protocol.site["rcm_common"]["coarse_beam_width"])
    partition = protocol.site["selection_contract"]["signed_partition"]
    positive_count = int(partition["target_support_count"])
    negative_count = int(partition["competitor_support_count"])
    selected_count = int(protocol.site["selection_contract"]["selected_count"])

    layers = [SiteComponent(layer_idx) for layer_idx in range(interface.backend.num_layers)]
    layer_results = _scan_components(protocol, layers, score_component)
    positive_layers = rank_positive(layer_results, beam_width)
    negative_layers = rank_negative(layer_results, beam_width)
    if len(positive_layers) != beam_width or len(negative_layers) != beam_width:
        counts = {
            "positive_layers": len(positive_layers),
            "negative_layers": len(negative_layers),
        }
        dump_json(
            paths.selector_dir / "incomplete.json",
            {"stage": "coarse_layer", "counts": counts, "status": "incomplete_no_fallback"},
        )
        raise RuntimeError(f"RCM-patch cannot form the signed layer beams: {counts}")

    def heads(layers_to_refine: list[CandidateResult]) -> list[SiteComponent]:
        components: list[SiteComponent] = []
        for layer in layers_to_refine:
            _hidden, num_heads, _head_dim = interface.geometry(layer.layer_idx)
            components.extend(
                SiteComponent(layer.layer_idx, head_idx) for head_idx in range(num_heads)
            )
        return components

    positive_heads = _scan_components(protocol, heads(positive_layers), score_component)
    negative_heads = _scan_components(protocol, heads(negative_layers), score_component)
    positive, negative = _rank_heads_by_own_effect(
        positive_heads, negative_heads, positive_count, negative_count
    )
    if len(positive) != positive_count or len(negative) != negative_count:
        counts = {"positive_heads": len(positive), "negative_heads": len(negative)}
        dump_json(
            paths.selector_dir / "incomplete.json",
            {"stage": "head_refine", "counts": counts, "status": "incomplete_no_fallback"},
        )
        raise RuntimeError(f"RCM-patch cannot form the signed head selection: {counts}")
    selected = [
        *((row, "target_support") for row in positive),
        *((row, "competitor_support") for row in negative),
    ]
    if len(selected) != selected_count:
        raise RuntimeError(f"RCM-patch did not materialize exactly {selected_count} heads")
    _write_rcm_outputs(
        protocol,
        job,
        paths,
        layer_results,
        {
            "target_support": positive_layers,
            "competitor_support": negative_layers,
        },
        {
            "target_support": positive_heads,
            "competitor_support": negative_heads,
        },
        selected,
        prototype_artifact=prototype_artifact,
        source_artifact=source_artifact,
    )


def _confiqa_matched_prototype_prompt(
    job: SiteJob,
    source: dict[str, Any],
    pair: ActuatorPair,
) -> tuple[str, str]:
    prompt_key = str(job.task["states"]["rcm_patch_target_prompt_key"])
    if prompt_key != "official_rag":
        raise ValueError(f"ConFiQA matched-prompt patch requires official_rag, got {prompt_key}")
    prompt = str(source["prompts"][prompt_key])
    if prompt != pair.prompt:
        raise ValueError(
            f"ConFiQA patch/ITI prompt mismatch for admitted pair {pair.sample_id}"
        )
    return prompt, prompt_key


def _confiqa_prototypes(
    protocol: SiteProtocol,
    job: SiteJob,
    paths: SitePaths,
    interface: PreOInterface,
    pairs: list[ActuatorPair],
) -> tuple[dict[int, Any], Path]:
    open_rows = {str(row["sample_id"]): row for row in load_jsonl(paths.open_rows)}
    sample_maps: list[dict[int, Any]] = []
    sample_records: list[dict[str, Any]] = []
    prompt_key = ""
    for pair in pairs:
        source = open_rows.get(pair.sample_id)
        if source is None:
            raise ValueError(f"missing ConFiQA open row for admitted pair {pair.sample_id}")
        prompt, prompt_key = _confiqa_matched_prototype_prompt(job, source, pair)
        alias_maps: list[dict[int, Any]] = []
        state_counts: list[int] = []
        for continuation in pair.y_plus_options:
            layer_map, states = interface.collect_layer_sequence_means(prompt, continuation)
            alias_maps.append(layer_map)
            state_counts.append(states)
        sample_maps.append(_mean_layer_maps(alias_maps))
        sample_records.append(
            {
                "sample_id": pair.sample_id,
                "row_index": pair.row_index,
                "alias_count": len(alias_maps),
                "decision_state_counts": state_counts,
            }
        )
    prototypes = _mean_layer_maps(sample_maps)
    artifact = paths.selector_dir / "trajectory_mean_states.safetensors"
    _save_prototypes(
        artifact,
        prototypes,
        {
            **protocol.snapshot_manifest(job),
            "component_geometry": {
                f"L{layer}.attn": list(vector.shape) for layer, vector in prototypes.items()
            },
            "source_artifact": str(paths.open_rows),
            "prompt_key": prompt_key,
            "prompt_match_contract": "same_official_rag_prompt_as_iti_pair_source",
            "target_sequence_ids": sample_records,
            "sample_count": len(sample_records),
            "total_alias_sequences": sum(row["alias_count"] for row in sample_records),
            "total_decision_states": sum(
                sum(row["decision_state_counts"]) for row in sample_records
            ),
            "averaging_order": protocol.site["rcm_common"]["prototype_aggregation_order"],
            "accumulation_dtype": "float32",
            "saved_dtype": "float32",
        },
    )
    return prototypes, artifact


def run_confiqa_rcm(
    protocol: SiteProtocol,
    job: SiteJob,
    paths: SitePaths,
    interface: PreOInterface,
) -> None:
    require_files([paths.open_rows, paths.pairs_csv])
    source_artifact = str(paths.pairs_csv)
    if _is_zero(job.selector):
        shared = _load_shared_zero_scan(
            protocol, job, paths, source_artifact=source_artifact
        )
        if shared is not None:
            _materialize_zero_selection(
                protocol, job, paths, shared, source_artifact=source_artifact
            )
            return

    prep = protocol.selector_preprocessing(job)
    score = job.task["score"]
    all_pairs = load_actuator_pairs(paths.pairs_csv, event=job.task["downstream"]["event"])
    if prep["candidate_group_source"] != "shared_admitted_pair_artifact":
        raise ValueError("ConFiQA RCM candidate group source drift")
    if prep["candidate_split_filter"] != "none":
        raise ValueError("ConFiQA RCM must use every admitted selector group before truncation")
    coarse_n = int(prep["coarse_pair_groups"])
    head_n = int(prep["head_refine_pair_groups"])
    if len(all_pairs) < max(coarse_n, head_n):
        raise ValueError(
            f"ConFiQA RCM requires {max(coarse_n, head_n)} admitted groups, found {len(all_pairs)}"
        )
    coarse_pairs = all_pairs[:coarse_n]
    head_pairs = all_pairs[:head_n]
    if [pair.sample_id for pair in coarse_pairs] != [pair.sample_id for pair in head_pairs]:
        raise ValueError("locked ConFiQA coarse and head groups must be identical")
    runner = ClosedCompetitionRunner(
        interface,
        score_mode=score["score_mode"],
        option_selection_mode=score["option_selection_mode"],
        max_aliases_per_side=int(score["max_aliases_per_side"]),
    )
    prototypes: dict[int, Any] = {}
    prototype_artifact = ""
    if _is_patch(job.selector):
        prototypes, artifact = _confiqa_prototypes(
            protocol, job, paths, interface, head_pairs
        )
        prototype_artifact = str(artifact)

    baseline_by_id = {pair.sample_id: runner.margin(pair) for pair in coarse_pairs}

    def score_pairs(component: SiteComponent) -> list[float]:
        action = _action(job.selector, component, prototypes)
        deltas: list[float] = []
        for pair in coarse_pairs:
            changed = runner.margin(pair, action)
            baseline = baseline_by_id[pair.sample_id]
            deltas.append(baseline - changed if _is_zero(job.selector) else changed - baseline)
        return deltas

    if _is_zero(job.selector):
        shared = _build_shared_zero_scan(
            protocol,
            job,
            paths,
            interface,
            score_pairs,
            source_artifact=source_artifact,
        )
        _materialize_zero_selection(
            protocol, job, paths, shared, source_artifact=source_artifact
        )
        return

    _run_patch_signed_selection(
        protocol,
        job,
        paths,
        interface,
        score_pairs,
        prototype_artifact=prototype_artifact,
        source_artifact=source_artifact,
    )

def _sentiment_scores(texts: list[str], job: SiteJob, device: str) -> list[float]:
    from transformers import pipeline

    score = job.task["score"]
    pipeline_device = -1
    if str(device).startswith("cuda"):
        pipeline_device = int(str(device).split(":", 1)[1]) if ":" in str(device) else 0
    classifier = pipeline(
        "text-classification",
        model=score["scorer_model"],
        revision=score["scorer_revision"],
        device=pipeline_device,
    )
    outputs = classifier(
        texts,
        batch_size=int(score["reward_batch_size"]),
        truncation=True,
        max_length=512,
        top_k=None,
    )
    margins: list[float] = []
    for output in outputs:
        by_label = {str(row["label"]).upper(): float(row["score"]) for row in output}
        positive = by_label.get(str(score["positive_label"]).upper(), 0.0)
        negative = by_label.get("NEGATIVE", 0.0)
        margins.append(positive - negative)
    del classifier
    return margins


def _imdb_prototypes(
    protocol: SiteProtocol,
    job: SiteJob,
    paths: SitePaths,
    interface: PreOInterface,
    target_rows: list[dict[str, Any]],
) -> tuple[dict[int, Any], Path]:
    sample_maps: list[dict[int, Any]] = []
    records = []
    for row in target_rows:
        layer_map, states = interface.collect_layer_sequence_means_ids(
            str(row["prompt"]),
            [int(value) for value in row["generated_token_ids"]],
        )
        sample_maps.append(layer_map)
        records.append(
            {
                "sample_id": row["sample_id"],
                "source_row_index": row.get("source_row_index", ""),
                "decision_state_count": states,
            }
        )
    prototypes = _mean_layer_maps(sample_maps)
    artifact = paths.selector_dir / "trajectory_mean_states.safetensors"
    _save_prototypes(
        artifact,
        prototypes,
        {
            **protocol.snapshot_manifest(job),
            "component_geometry": {
                f"L{layer}.attn": list(vector.shape) for layer, vector in prototypes.items()
            },
            "source_artifact": str(paths.selector_good_generations),
            "prompt_key": "good_template",
            "target_sequence_ids": records,
            "sample_count": len(records),
            "total_decision_states": sum(row["decision_state_count"] for row in records),
            "averaging_order": protocol.site["rcm_common"]["prototype_aggregation_order"],
            "accumulation_dtype": "float32",
            "saved_dtype": "float32",
        },
    )
    return prototypes, artifact


def run_imdb_rcm(
    protocol: SiteProtocol,
    job: SiteJob,
    paths: SitePaths,
    interface: PreOInterface,
) -> None:
    require_files([paths.selector_prompts])
    good_rows, base_rows = ensure_imdb_selector_states(
        protocol, job, paths, interface
    )
    source_artifact = str(paths.selector_state_pair_manifest)
    if _is_zero(job.selector):
        shared = _load_shared_zero_scan(
            protocol, job, paths, source_artifact=source_artifact
        )
        if shared is not None:
            _materialize_zero_selection(
                protocol, job, paths, shared, source_artifact=source_artifact
            )
            return

    state_spec = protocol.selector_state_generation(job)
    if state_spec is None:
        raise RuntimeError("IMDb selector state-generation contract is missing")
    source_rows = [
        row for row in load_jsonl(paths.selector_prompts)
        if str(row.get("admitted", "1")) not in {"0", "false", "False"}
    ][: int(state_spec["prompt_rows"])]
    good_prompts = imdb_condition_rows(job, source_rows, "good")
    base_prompts = imdb_condition_rows(job, source_rows, "base")

    prototypes: dict[int, Any] = {}
    prototype_artifact = ""
    if _is_patch(job.selector):
        prototypes, artifact = _imdb_prototypes(
            protocol, job, paths, interface, good_rows
        )
        prototype_artifact = str(artifact)
        recipient_prompts = base_prompts
        baseline_rows = base_rows
    else:
        recipient_prompts = good_prompts
        baseline_rows = good_rows
    sentiment_scorer = FrozenSentimentScorer(job.task["score"], device=interface.device)
    baseline_scores = [
        row["competition_score"]
        for row in sentiment_scorer.score(
            [str(item["completion"]) for item in baseline_rows]
        )
    ]
    dump_jsonl(
        paths.selector_dir / "native_baseline.jsonl",
        (
            {**row, "competition_score": score}
            for row, score in zip(baseline_rows, baseline_scores, strict=True)
        ),
    )

    def score_component(component: SiteComponent) -> list[float]:
        action = _action(job.selector, component, prototypes)
        changed_rows = generate_imdb_condition(
            interface,
            recipient_prompts,
            action=action,
            state_spec=state_spec,
            seed=int(state_spec["generation_seed"]),
        )
        changed_scores = [
            row["competition_score"]
            for row in sentiment_scorer.score(
                [str(item["completion"]) for item in changed_rows]
            )
        ]
        if _is_zero(job.selector):
            return [
                base - changed
                for base, changed in zip(baseline_scores, changed_scores, strict=True)
            ]
        return [
            changed - base
            for base, changed in zip(baseline_scores, changed_scores, strict=True)
        ]

    if _is_zero(job.selector):
        shared = _build_shared_zero_scan(
            protocol,
            job,
            paths,
            interface,
            score_component,
            source_artifact=source_artifact,
        )
        _materialize_zero_selection(
            protocol, job, paths, shared, source_artifact=source_artifact
        )
        return

    _run_patch_signed_selection(
        protocol,
        job,
        paths,
        interface,
        score_component,
        prototype_artifact=prototype_artifact,
        source_artifact=source_artifact,
    )

def _role_rows(groups: dict[str, list[CandidateResult]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for role, results in groups.items():
        for row in result_rows(results):
            rows.append({**row, "selection_role": role})
    return rows


def _head_effect_role(result: CandidateResult) -> str:
    if result.mean_score_delta > 0:
        return "target_support"
    if result.mean_score_delta < 0:
        return "competitor_support"
    return "zero_effect"


def _head_scan_rows(groups: dict[str, list[CandidateResult]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for parent_role, results in groups.items():
        for result, row in zip(results, result_rows(results), strict=True):
            rows.append({
                **row,
                "parent_layer_beam": parent_role,
                "selection_role": _head_effect_role(result),
            })
    return rows


def _write_rcm_outputs(
    protocol: SiteProtocol,
    job: SiteJob,
    paths: SitePaths,
    layer_results: list[CandidateResult],
    layer_beams: dict[str, list[CandidateResult]],
    head_results: dict[str, list[CandidateResult]],
    selected: list[tuple[CandidateResult, str]],
    *,
    prototype_artifact: str,
    source_artifact: str,
    shared_scan_artifact: str = "",
    shared_scan_sha256: str = "",
) -> None:
    paths.selector_dir.mkdir(parents=True, exist_ok=True)
    dump_csv(paths.selector_dir / "layer_scan.csv", result_rows(layer_results))
    dump_csv(paths.selector_dir / "layer_beam.csv", _role_rows(layer_beams))
    dump_csv(paths.selector_dir / "head_scan.csv", _head_scan_rows(head_results))
    rows = []
    for rank, (result, role) in enumerate(selected, start=1):
        rows.append(
            {
                "job_id": job.job_id,
                "selector": job.selector,
                "replicate": job.replicate,
                "rank": rank,
                "component_id": result.component_id,
                "selection_role": role,
                "ranking_score": role_oriented_ci95_lower_bound(result),
                "ranking_score_name": "role_oriented_ci95_lower_bound",
                "score_name": job.task["score"]["name"],
                "mean_score_delta": result.mean_score_delta,
                "ci95_low": result.ci95_low,
                "ci95_high": result.ci95_high,
                "role_oriented_ci95_lower_bound": role_oriented_ci95_lower_bound(result),
                "ci_crosses_zero": result.ci95_low <= 0.0 <= result.ci95_high,
                "n": len(result.deltas),
                "seed": job.seed,
                "state_constructor": _state_constructor(job.selector),
                "prototype_artifact": prototype_artifact,
                "source_artifact": source_artifact,
                "shared_scan_artifact": shared_scan_artifact,
            }
        )
    dump_csv(paths.selected_heads, rows)
    dump_json(
        paths.selector_dir / "selector_manifest.json",
        {
            **protocol.snapshot_manifest(job),
            "selected_heads": [result.component_id for result, _role in selected],
            "selection_roles": [role for _result, role in selected],
            "component_effect_audit": {
                "selected_count": len(selected),
                "ci_crosses_zero_count": sum(
                    result.ci95_low <= 0.0 <= result.ci95_high for result, _role in selected
                ),
                "ci_policy": protocol.site["selection_contract"]["ci_policy"],
                "ranking_score_name": "role_oriented_ci95_lower_bound",
                "ranking_scores": [
                    role_oriented_ci95_lower_bound(result) for result, _role in selected
                ],
            },
            "layer_beam": {
                role: [row.component_id for row in results]
                for role, results in layer_beams.items()
            },
            "state_constructor": _state_constructor(job.selector),
            "prototype_artifact": prototype_artifact,
            "source_artifact": source_artifact,
            "shared_scan_artifact": shared_scan_artifact,
            "shared_scan_sha256": shared_scan_sha256,
            "head_effect_role_source": protocol.site["rcm_common"]["head_effect_role_source"],
            "parent_layer_sign_inheritance": False,
            "ci_role": "selection_statistic_and_audit",
            "status": "complete",
        },
    )
