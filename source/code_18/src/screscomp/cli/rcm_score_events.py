from __future__ import annotations

import argparse
from pathlib import Path
from math import prod
from statistics import mean
from typing import Any

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, desc=None):
        return iterable

from screscomp.cli.score_ckplug_factor_component_scan import _target_and_contrast_prompts
from screscomp.data import dump_csv, dump_jsonl, load_csv, load_jsonl
from screscomp.modeling import TransformersABBackend
from screscomp.rcm.graph import RCMGraph, load_graph
from screscomp.rcm.prediction import summarize_prediction_scores


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Score natural generations with a fixed RCM graph. This is a readout-only pass: "
            "one current-prompt forward capture per sample plus offline target/contrast directions."
        )
    )
    p.add_argument("--eval_jsonl", type=Path, required=True)
    p.add_argument("--generations_jsonl", type=Path, required=True)
    p.add_argument("--graph_json", type=Path, required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--out_predictions_jsonl", type=Path, required=True)
    p.add_argument("--out_summary_csv", type=Path, required=True)
    p.add_argument("--target_method", type=str, default="strong_rag")
    p.add_argument("--prompt_key", type=str, default="strong_rag")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--use_chat_template", action="store_true")
    p.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
    )
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--max_rows", type=int, default=None)
    p.add_argument(
        "--source_contrast_kind",
        choices=["strong_no_rag", "objective_prior"],
        default="strong_no_rag",
        help="Source-axis contrast attractor used when constructing prediction readout directions.",
    )
    p.add_argument("--normalize", action="store_true", help="Use cosine-style normalized projection scores.")
    p.add_argument(
        "--score_mode",
        choices=["dot", "cosine", "midpoint"],
        default=None,
        help=(
            "Projection score. Defaults to dot unless --normalize is set. "
            "midpoint uses dot(current - (target+contrast)/2, target-contrast) / ||target-contrast||^2. "
            "--normalize is kept for backward compatibility."
        ),
    )
    p.add_argument(
        "--edge_stats_csv",
        type=Path,
        default=None,
        help="Optional per-edge robust normalization stats from rcm_calibrate_readout --out_edge_stats_csv.",
    )
    p.add_argument(
        "--edge_normalization",
        choices=["none", "robust"],
        default="none",
        help="Normalize each edge score before aggregation. robust uses median/MAD edge stats.",
    )
    p.add_argument("--conflict_gamma", type=float, default=1.0)
    p.add_argument("--cross_conflict_gamma", type=float, default=0.0)
    p.add_argument(
        "--readout_calibration_csv",
        type=Path,
        default=None,
        help="Optional frozen readout calibration from screscomp.cli.rcm_calibrate_readout.",
    )
    p.add_argument(
        "--include_native_baselines",
        action="store_true",
        help="Include output-derived sanity baselines such as length. Omit for the main white-box prediction table.",
    )
    return p.parse_args()


def _generation_labels(row: dict[str, Any]) -> dict[str, Any]:
    outcome = str(row.get("outcome", ""))
    first_cf = bool(row.get("first_answer_cf_em", row.get("cf_em", False)))
    short = bool(row.get("short_output", False))
    context_only = outcome == "context_only"
    prior_only = outcome == "prior_only"
    both = outcome == "both"
    neither = outcome == "neither"
    single_commitment = context_only or prior_only
    composite_success = context_only and first_cf and single_commitment and not bool(row.get("orig_hit", False))
    return {
        "label_context_only": context_only,
        "label_prior_only": prior_only,
        "label_both": both,
        "label_neither": neither,
        "label_failure_not_context_only": not context_only,
        "label_first_answer_cf_em": first_cf,
        "label_short_output": short,
        "label_single_commitment": single_commitment,
        "label_composite_success": composite_success,
        "outcome": outcome,
    }


def _read_generations(path: Path, target_method: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        if str(row.get("method")) == target_method:
            output[str(row["sample_id"])] = row
    if not output:
        raise ValueError(f"No generation rows found for method={target_method!r}.")
    return output


def _component_specs(graph: RCMGraph) -> list[tuple[int, str]]:
    seen: set[tuple[int, str]] = set()
    specs: list[tuple[int, str]] = []
    for node in graph.components:
        spec = (node.layer_idx, node.component_type)
        if spec not in seen:
            seen.add(spec)
            specs.append(spec)
    return specs


def _target_contrast_direction_cache(
    backend: TransformersABBackend,
    graph: RCMGraph,
    row: dict[str, Any],
    source_contrast_kind: str,
) -> dict[tuple[str, int, str], tuple[Any, Any, Any]]:
    cache: dict[tuple[str, int, str], tuple[Any, Any, Any]] = {}
    for axis in sorted(graph.axes):
        target_prompt, contrast_prompt, _pair_name, _, _ = _target_and_contrast_prompts(
            row,
            axis,
            "forward",
            source_contrast_kind=source_contrast_kind,
        )
        axis_specs = [(node.layer_idx, node.component_type) for node in graph.by_axis(axis)]
        target_acts = backend.capture_component_last_token(target_prompt, axis_specs)
        contrast_acts = backend.capture_component_last_token(contrast_prompt, axis_specs)
        for spec in axis_specs:
            direction = target_acts[spec] - contrast_acts[spec]
            cache[(axis, spec[0], spec[1])] = (target_acts[spec], contrast_acts[spec], direction)
    return cache


def _projection_score(current, target, contrast, direction, score_mode: str) -> float:
    if score_mode == "midpoint":
        midpoint = (target + contrast) * 0.5
        centered = current - midpoint
        denom = direction.flatten().float().dot(direction.flatten().float())
        if float(denom.item()) <= 1e-12:
            return 0.0
        return float((centered.flatten().float().dot(direction.flatten().float()) / denom).item())
    vector = current - contrast
    if score_mode == "cosine":
        denom = vector.flatten().float().norm() * direction.flatten().float().norm()
        if float(denom.item()) <= 1e-12:
            return 0.0
        return float((vector.flatten().float().dot(direction.flatten().float()) / denom).item())
    if score_mode != "dot":
        raise ValueError(f"Unsupported score_mode: {score_mode}")
    return float(vector.flatten().float().dot(direction.flatten().float()).item())


def _signed(node_sign: str) -> float:
    if node_sign == "up":
        return 1.0
    if node_sign == "down":
        return -1.0
    raise ValueError(f"Unsupported node sign: {node_sign}")


def _float(value: Any, default: float = 0.0) -> float:
    if value in ("", None):
        return default
    return float(value)


def _load_readout_calibration(path: Path | None) -> dict[str, list[dict[str, Any]]]:
    if path is None:
        return {}
    by_target: dict[str, list[dict[str, Any]]] = {}
    for row in load_csv(path):
        target = str(row.get("target", "")).strip()
        edge_id = str(row.get("edge_id", "")).strip()
        if not target or not edge_id:
            continue
        by_target.setdefault(target, []).append(
            {
                "edge_id": edge_id,
                "readout_sign": _float(row.get("readout_sign"), 1.0),
                "weight": _float(row.get("weight"), 0.0),
            }
        )
    return by_target


def _load_edge_stats(path: Path | None) -> dict[str, dict[str, float]]:
    if path is None:
        return {}
    output: dict[str, dict[str, float]] = {}
    for row in load_csv(path):
        edge_id = str(row.get("edge_id", "")).strip()
        if not edge_id:
            continue
        output[edge_id] = {
            "median": _float(row.get("median"), 0.0),
            "mad": _float(row.get("mad"), 0.0),
            "scale": _float(row.get("robust_scale"), 1.0),
        }
    return output


def _normalize_edge_score(
    score: float,
    edge_id: str,
    *,
    edge_stats: dict[str, dict[str, float]],
    edge_normalization: str,
) -> float:
    if edge_normalization == "none":
        return score
    if edge_normalization != "robust":
        raise ValueError(f"Unsupported edge_normalization: {edge_normalization}")
    stats = edge_stats.get(edge_id)
    if not stats:
        return score
    scale = stats.get("scale", 1.0)
    if abs(scale) <= 1e-12:
        return 0.0
    return (score - stats.get("median", 0.0)) / scale


def _bounded(value: float) -> float:
    return value / (1.0 + abs(value))


def _score_row(
    backend: TransformersABBackend,
    graph: RCMGraph,
    row: dict[str, Any],
    *,
    prompt_key: str,
    score_mode: str,
    edge_stats: dict[str, dict[str, float]] | None = None,
    edge_normalization: str = "none",
    source_contrast_kind: str = "strong_no_rag",
    conflict_gamma: float = 1.0,
    cross_conflict_gamma: float = 0.0,
    readout_calibration: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    current_prompt = row["prompts"][prompt_key]
    current_acts = backend.capture_component_last_token(current_prompt, _component_specs(graph))
    direction_cache = _target_contrast_direction_cache(backend, graph, row, source_contrast_kind)

    axis_num: dict[str, float] = {axis: 0.0 for axis in graph.axes}
    axis_den: dict[str, float] = {axis: 0.0 for axis in graph.axes}
    axis_up_num: dict[str, float] = {axis: 0.0 for axis in graph.axes}
    axis_up_den: dict[str, float] = {axis: 0.0 for axis in graph.axes}
    axis_down_num: dict[str, float] = {axis: 0.0 for axis in graph.axes}
    axis_down_den: dict[str, float] = {axis: 0.0 for axis in graph.axes}
    edge_scores: list[dict[str, Any]] = []
    raw_by_edge_id: dict[str, float] = {}
    for node in graph.components:
        key = (node.axis, node.layer_idx, node.component_type)
        target, contrast, direction = direction_cache[key]
        edge_id = node.metadata.get("edge_id", f"{node.axis}:{node.component_id}:{node.sign}:{node.timing}")
        projection = _projection_score(
            current=current_acts[(node.layer_idx, node.component_type)],
            target=target,
            contrast=contrast,
            direction=direction,
            score_mode=score_mode,
        )
        raw = _normalize_edge_score(
            projection,
            str(edge_id),
            edge_stats=edge_stats or {},
            edge_normalization=edge_normalization,
        )
        raw_by_edge_id[str(edge_id)] = raw
        contribution = _signed(node.sign) * node.weight * raw
        axis_num[node.axis] += contribution
        axis_den[node.axis] += abs(node.weight)
        if node.sign == "up":
            axis_up_num[node.axis] += node.weight * raw
            axis_up_den[node.axis] += abs(node.weight)
        elif node.sign == "down":
            axis_down_num[node.axis] += node.weight * raw
            axis_down_den[node.axis] += abs(node.weight)
        edge_scores.append(
            {
                "edge_id": edge_id,
                "axis": node.axis,
                "component_id": node.component_id,
                "sign": node.sign,
                "timing": node.timing,
                "weight": node.weight,
                "projection_score": projection,
                "raw_score": raw,
                "contribution": contribution,
            }
        )

    axis_scores = {
        f"{axis}_score": (axis_num[axis] / axis_den[axis] if axis_den[axis] > 0 else 0.0)
        for axis in sorted(graph.axes)
    }
    for axis in sorted(graph.axes):
        up_score = axis_up_num[axis] / axis_up_den[axis] if axis_up_den[axis] > 0 else 0.0
        down_score = axis_down_num[axis] / axis_down_den[axis] if axis_down_den[axis] > 0 else 0.0
        conflict = min(max(up_score, 0.0), max(down_score, 0.0))
        axis_scores[f"{axis}_up_score"] = up_score
        axis_scores[f"{axis}_down_score"] = down_score
        axis_scores[f"{axis}_margin_score"] = up_score - down_score
        axis_scores[f"{axis}_intensity_score"] = abs(up_score) + abs(down_score)
        axis_scores[f"{axis}_momentum_score"] = (up_score - down_score) * (abs(up_score) + abs(down_score))
        axis_scores[f"{axis}_balance_ratio_score"] = (
            (up_score - down_score) / (abs(up_score) + abs(down_score) + 1e-12)
        )
        axis_scores[f"{axis}_conflict_score"] = conflict
        axis_scores[f"{axis}_competition_score"] = conflict
        axis_scores[f"{axis}_potential_score"] = (up_score - down_score) - conflict_gamma * conflict
        axis_scores[f"{axis}_up_only_score"] = up_score
        axis_scores[f"{axis}_down_only_score"] = -down_score
        axis_scores[f"{axis}_up_down_score"] = up_score - down_score
    source = axis_scores.get("source_identity_score", 0.0)
    form = axis_scores.get("form_score", 0.0)
    commitment = axis_scores.get("commitment_score", 0.0)
    prior_suppression = axis_scores.get("prior_suppression_score", 0.0)
    source_margin = axis_scores.get("source_identity_margin_score", 0.0)
    form_potential = axis_scores.get("form_potential_score", 0.0)
    commitment_potential = axis_scores.get("commitment_potential_score", 0.0)
    prior_suppression_margin = axis_scores.get("prior_suppression_margin_score", 0.0)
    source_prior_consistency = source_margin + prior_suppression_margin
    source_prior_conflict = max(0.0, source_margin) * max(0.0, -prior_suppression_margin)
    prior_failure_conjunction = _bounded(-source_margin) * _bounded(-prior_suppression_margin)
    prior_failure_energy = -source_prior_consistency + source_prior_conflict
    axis_scores["source_prior_consistency_score"] = source_prior_consistency
    axis_scores["source_prior_conflict_score"] = source_prior_conflict
    axis_scores["prior_failure_conjunction_score"] = prior_failure_conjunction
    axis_scores["prior_failure_energy_score"] = prior_failure_energy
    axis_scores["composite_score"] = source + form + commitment + prior_suppression
    axis_scores["composite_margin_score"] = (
        axis_scores.get("source_identity_margin_score", 0.0)
        + axis_scores.get("form_margin_score", 0.0)
        + axis_scores.get("commitment_margin_score", 0.0)
        + axis_scores.get("prior_suppression_margin_score", 0.0)
    )
    axis_scores["composite_conflict_score"] = (
        axis_scores.get("source_identity_competition_score", 0.0)
        + axis_scores.get("form_competition_score", 0.0)
        + axis_scores.get("commitment_competition_score", 0.0)
        + axis_scores.get("prior_suppression_competition_score", 0.0)
    )
    axis_scores["composite_competition_score"] = axis_scores["composite_conflict_score"]
    axis_scores["composite_intensity_score"] = (
        axis_scores.get("source_identity_intensity_score", 0.0)
        + axis_scores.get("form_intensity_score", 0.0)
        + axis_scores.get("commitment_intensity_score", 0.0)
        + axis_scores.get("prior_suppression_intensity_score", 0.0)
    )
    cross_axis_conflict = (
        max(0.0, axis_scores.get("source_identity_margin_score", 0.0))
        * max(0.0, -axis_scores.get("prior_suppression_margin_score", 0.0))
        + max(0.0, axis_scores.get("source_identity_margin_score", 0.0))
        * max(0.0, -axis_scores.get("commitment_margin_score", 0.0))
    )
    axis_scores["cross_axis_conflict_score"] = cross_axis_conflict
    axis_scores["composite_potential_score"] = (
        axis_scores.get("source_identity_potential_score", 0.0)
        + axis_scores.get("form_potential_score", 0.0)
        + axis_scores.get("commitment_potential_score", 0.0)
        + axis_scores.get("prior_suppression_potential_score", 0.0)
        - cross_conflict_gamma * cross_axis_conflict
    )
    composite_parts = [source_prior_consistency, form_potential, commitment_potential]
    axis_scores["composite_gate_score"] = min(composite_parts)
    axis_scores["composite_soft_product_score"] = prod(
        _bounded(part) for part in composite_parts
    )
    axis_scores["source_form_commitment_conflict_score"] = (
        axis_scores.get("source_identity_conflict_score", 0.0)
        + axis_scores.get("form_conflict_score", 0.0)
        + axis_scores.get("commitment_conflict_score", 0.0)
        + cross_axis_conflict
    )
    for held_out_axis in ("source_identity", "prior_suppression", "form", "commitment"):
        axis_scores[f"composite_without_{held_out_axis}_potential_score"] = sum(
            axis_scores.get(f"{axis}_potential_score", 0.0)
            for axis in ("source_identity", "prior_suppression", "form", "commitment")
            if axis != held_out_axis
        )
    axis_scores["residual_arbitration_energy_score"] = axis_scores["composite_potential_score"]
    axis_scores["anti_source_identity_score"] = -source
    axis_scores["anti_form_score"] = -form
    axis_scores["anti_commitment_score"] = -commitment
    axis_scores["anti_prior_suppression_score"] = -prior_suppression
    axis_scores["anti_composite_score"] = -axis_scores["composite_score"]
    axis_scores["anti_composite_margin_score"] = -axis_scores["composite_margin_score"]
    axis_scores["anti_composite_potential_score"] = -axis_scores["composite_potential_score"]

    if readout_calibration:
        for target, calibration_edges in readout_calibration.items():
            num = 0.0
            den = 0.0
            for item in calibration_edges:
                edge_id = str(item["edge_id"])
                if edge_id not in raw_by_edge_id:
                    continue
                weight = float(item["weight"])
                num += float(item["readout_sign"]) * weight * raw_by_edge_id[edge_id]
                den += abs(weight)
            axis_scores[f"calibrated_{target}_score"] = num / den if den > 0 else 0.0
    return {**axis_scores, "edge_scores": edge_scores}


def _native_baseline_scores(gen: dict[str, Any]) -> dict[str, float]:
    output_chars = float(gen.get("output_chars") or len(str(gen.get("prediction", ""))))
    tokens_approx = max(1.0, output_chars / 4.0)
    return {
        "output_chars_score": output_chars,
        "shortness_score": -output_chars,
        "tokens_approx_score": tokens_approx,
    }


def _target_scores_for_graph(
    graph: RCMGraph,
    calibrated_targets: list[str] | None = None,
    include_native_baselines: bool = False,
) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {
        "label_context_only": [
            "source_identity_score",
            "source_identity_margin_score",
            "source_identity_momentum_score",
            "source_identity_balance_ratio_score",
            "source_identity_up_score",
            "source_identity_down_only_score",
            "source_identity_up_down_score",
            "source_prior_consistency_score",
            "source_prior_conflict_score",
            "source_identity_competition_score",
            "source_identity_potential_score",
            "composite_score",
            "composite_margin_score",
            "composite_competition_score",
            "composite_potential_score",
            "residual_arbitration_energy_score",
        ],
        "label_prior_only": [
            "anti_source_identity_score",
            "prior_failure_conjunction_score",
            "prior_failure_energy_score",
            "source_prior_conflict_score",
        ],
        "label_failure_not_context_only": [
            "anti_source_identity_score",
            "anti_composite_score",
            "anti_composite_margin_score",
            "anti_composite_potential_score",
        ],
        "label_composite_success": [
            "composite_score",
            "composite_margin_score",
            "composite_competition_score",
            "composite_potential_score",
            "composite_gate_score",
            "composite_soft_product_score",
            "source_prior_consistency_score",
            "composite_without_source_identity_potential_score",
            "composite_without_prior_suppression_potential_score",
            "composite_without_form_potential_score",
            "composite_without_commitment_potential_score",
            "residual_arbitration_energy_score",
        ],
    }
    if "form" in graph.axes:
        output["label_first_answer_cf_em"] = [
            "form_score",
            "form_margin_score",
            "form_momentum_score",
            "form_balance_ratio_score",
            "form_up_score",
            "form_down_only_score",
            "form_up_down_score",
            "form_competition_score",
            "form_potential_score",
            "composite_score",
            "composite_margin_score",
            "composite_competition_score",
            "composite_potential_score",
        ]
        output["label_short_output"] = [
            "form_score",
            "form_margin_score",
            "form_momentum_score",
            "form_balance_ratio_score",
            "form_up_score",
            "form_down_only_score",
            "form_up_down_score",
            "form_competition_score",
            "form_potential_score",
        ]
        if include_native_baselines:
            output["label_short_output"].append("shortness_score")
    if "commitment" in graph.axes:
        output["label_single_commitment"] = [
            "commitment_score",
            "commitment_margin_score",
            "commitment_momentum_score",
            "commitment_balance_ratio_score",
            "commitment_up_score",
            "commitment_down_only_score",
            "commitment_up_down_score",
            "commitment_conflict_score",
            "commitment_potential_score",
            "composite_score",
            "composite_margin_score",
            "composite_potential_score",
        ]
    if "prior_suppression" in graph.axes:
        output["label_prior_only"].append("anti_prior_suppression_score")
    for target in calibrated_targets or []:
        output.setdefault(target, []).append(f"calibrated_{target}_score")
    return output


def _mean_or_none(values: list[float]) -> float | None:
    return mean(values) if values else None


def main() -> None:
    args = parse_args()
    graph = load_graph(args.graph_json)
    eval_rows = load_jsonl(args.eval_jsonl)
    if args.start:
        eval_rows = eval_rows[args.start :]
    if args.max_rows is not None:
        eval_rows = eval_rows[: args.max_rows]
    generations = _read_generations(args.generations_jsonl, args.target_method)
    eval_rows = [row for row in eval_rows if str(row["sample_id"]) in generations]
    if not eval_rows:
        raise ValueError("No eval rows matched generation rows.")

    backend = TransformersABBackend(
        model_name_or_path=args.model,
        device=args.device,
        use_chat_template=args.use_chat_template,
        torch_dtype=args.torch_dtype,
    )
    score_mode = args.score_mode or ("cosine" if args.normalize else "dot")
    readout_calibration = _load_readout_calibration(args.readout_calibration_csv)
    edge_stats = _load_edge_stats(args.edge_stats_csv)

    prediction_rows: list[dict[str, Any]] = []
    for row in tqdm(eval_rows, desc="rcm event scoring"):
        gen = generations[str(row["sample_id"])]
        labels = _generation_labels(gen)
        native_scores = _native_baseline_scores(gen) if args.include_native_baselines else {}
        scores = _score_row(
            backend=backend,
            graph=graph,
            row=row,
            prompt_key=args.prompt_key,
            score_mode=score_mode,
            edge_stats=edge_stats,
            edge_normalization=args.edge_normalization,
            source_contrast_kind=args.source_contrast_kind,
            conflict_gamma=args.conflict_gamma,
            cross_conflict_gamma=args.cross_conflict_gamma,
            readout_calibration=readout_calibration,
        )
        prediction_rows.append(
            {
                "sample_id": row["sample_id"],
                "target_method": args.target_method,
                "graph": graph.name,
                "prediction": gen.get("prediction", ""),
                **labels,
                **native_scores,
                **{key: value for key, value in scores.items() if key != "edge_scores"},
                "edge_scores": scores["edge_scores"],
            }
        )

    dump_jsonl(args.out_predictions_jsonl, prediction_rows)
    summary = [metric.to_dict() for metric in summarize_prediction_scores(
        prediction_rows,
        target_scores=_target_scores_for_graph(
            graph,
            sorted(readout_calibration),
            include_native_baselines=args.include_native_baselines,
        ),
    )]
    dump_csv(args.out_summary_csv, summary)
    print(
        f"[rcm-score-events] rows={len(prediction_rows)} graph={graph.name} "
        f"axes={','.join(sorted(graph.axes))} out={args.out_summary_csv}"
    )


if __name__ == "__main__":
    main()
