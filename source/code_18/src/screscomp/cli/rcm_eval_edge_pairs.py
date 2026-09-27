from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, desc=None):
        return iterable

from screscomp.cli.rcm_eval_candidate_edges import _operation_alpha, _parse_float_map
from screscomp.cli.score_ckplug_factor_component_scan import _addition, _generation_prompt, _output_metrics, _score_row
from screscomp.cli.score_ckplug_generation import _classify_generation, _parse_stop_strings
from screscomp.data import dump_csv, dump_jsonl, load_csv, load_jsonl
from screscomp.modeling import TransformersABBackend
from screscomp.rcm.defaults import objective_preset
from screscomp.rcm.objective import weighted_objective


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value in ("", None):
        return default
    return float(value)


def _int(row: dict[str, Any], key: str, default: int = 0) -> int:
    value = row.get(key, default)
    if value in ("", None):
        return default
    return int(float(value))


def _load_edges(path: Path, max_edges: int | None) -> list[dict[str, Any]]:
    rows = load_csv(path)
    if max_edges is not None:
        rows = rows[:max_edges]
    return rows


def _mean_float(rows: list[dict[str, Any]], key: str) -> float:
    return mean(float(row.get(key, 0.0) or 0.0) for row in rows) if rows else 0.0


def _summary_metric(rows: list[dict[str, Any]], objective_terms: dict[str, float]) -> dict[str, float]:
    summary = {
        "context_only": _mean_float(rows, "context_only"),
        "prior_only": _mean_float(rows, "prior_only"),
        "both": _mean_float(rows, "both"),
        "neither": _mean_float(rows, "neither"),
        "cf_hit": _mean_float(rows, "cf_hit"),
        "orig_hit": _mean_float(rows, "orig_hit"),
        "first_answer_cf_em": _mean_float(rows, "first_answer_cf_em"),
        "short_output": _mean_float(rows, "short_output"),
        "output_chars": _mean_float(rows, "output_chars"),
        "source_score": _mean_float(rows, "source_score"),
        "form_score": _mean_float(rows, "form_score"),
        "commitment_score": _mean_float(rows, "commitment_score"),
        "prior_suppression_score": _mean_float(rows, "prior_suppression_score"),
        "joint_score": _mean_float(rows, "joint_score"),
    }
    summary["objective_score"] = weighted_objective(summary, objective_terms) if objective_terms else summary["joint_score"]
    return summary


def _edge_addition(backend: TransformersABBackend, row: dict[str, Any], edge: dict[str, Any], operation: str) -> dict[str, Any] | None:
    sign = str(edge["sign"])
    alpha = _operation_alpha(sign, operation, _float(edge, "alpha", 1.0))
    if alpha is None:
        return None
    component_spec = (_int(edge, "layer_idx"), str(edge["component_type"]))
    addition = _addition(backend=backend, row=row, component_spec=component_spec, axis=str(edge["axis"]), alpha=alpha)
    return {**addition, "apply_mode": str(edge["timing"])}


def _evaluate_edges(
    backend: TransformersABBackend,
    rows: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    *,
    operation: str,
    max_new_tokens: int,
    stop_strings: list[str],
    generation_axis: str | None = None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    if generation_axis is None:
        if not edges:
            raise ValueError("generation_axis is required when evaluating an empty edge set.")
        generation_axis = str(edges[0]["axis"])
    for row in rows:
        additions = []
        for edge in edges:
            addition = _edge_addition(backend, row, edge, operation)
            if addition is not None:
                additions.append(addition)
        prediction = backend.generate_with_component_last_token_add_many(
            _generation_prompt(row, generation_axis),
            additions=additions,
            max_new_tokens=max_new_tokens,
            stop_strings=stop_strings,
            apply_mode="all",
        )
        classified = _classify_generation(prediction, row["orig_answers"], row["cf_answers"])
        metrics = {**classified, **_output_metrics(prediction, row["cf_answers"])}
        output.append(
            {
                "sample_id": row["sample_id"],
                "prediction": prediction,
                "context_only": float(metrics.get("outcome") == "context_only"),
                "prior_only": float(metrics.get("outcome") == "prior_only"),
                "both": float(metrics.get("outcome") == "both"),
                "neither": float(metrics.get("outcome") == "neither"),
                **metrics,
                **_score_row(metrics),
            }
        )
    return output


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate pairwise RCM edge synergy/conflict on validation generations.")
    p.add_argument("--eval_jsonl", type=Path, required=True)
    p.add_argument("--selected_edges_csv", type=Path, required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--out_rows_jsonl", type=Path, required=True)
    p.add_argument("--out_summary_csv", type=Path, required=True)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--use_chat_template", action="store_true")
    p.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
    )
    p.add_argument("--operation", default="target")
    p.add_argument("--objective_terms", default="joint_score=1")
    p.add_argument("--objective_preset", default="", help="Objective preset used when --objective_terms is empty.")
    p.add_argument("--max_rows", type=int, default=None)
    p.add_argument("--max_edges", type=int, default=12)
    p.add_argument("--max_pairs", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=48)
    p.add_argument("--stop_strings", type=str, default="Q:")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    eval_rows = load_jsonl(args.eval_jsonl)
    if args.max_rows is not None:
        eval_rows = eval_rows[: args.max_rows]
    edges = _load_edges(args.selected_edges_csv, args.max_edges)
    edge_pairs = list(combinations(edges, 2))
    if args.max_pairs is not None:
        edge_pairs = edge_pairs[: args.max_pairs]
    objective_terms = _parse_float_map(args.objective_terms) or objective_preset(args.objective_preset)
    stop_strings = _parse_stop_strings(args.stop_strings)
    backend = TransformersABBackend(
        model_name_or_path=args.model,
        device=args.device,
        use_chat_template=args.use_chat_template,
        torch_dtype=args.torch_dtype,
    )

    output_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    single_cache: dict[str, dict[str, float]] = {}
    base_cache: dict[str, float] = {}

    for edge_a, edge_b in tqdm(edge_pairs, desc="edge pair eval"):
        generation_axis = str(edge_a["axis"])
        if generation_axis not in base_cache:
            base_rows = _evaluate_edges(
                backend,
                eval_rows,
                [],
                operation=args.operation,
                max_new_tokens=args.max_new_tokens,
                stop_strings=stop_strings,
                generation_axis=generation_axis,
            )
            base_cache[generation_axis] = _summary_metric(base_rows, objective_terms)["objective_score"]
        base_score = base_cache[generation_axis]
        for edge in (edge_a, edge_b):
            if edge["edge_id"] not in single_cache:
                single_rows = _evaluate_edges(
                    backend,
                    eval_rows,
                    [edge],
                    operation=args.operation,
                    max_new_tokens=args.max_new_tokens,
                    stop_strings=stop_strings,
                )
                single_cache[str(edge["edge_id"])] = _summary_metric(single_rows, objective_terms)
        pair_rows = _evaluate_edges(
            backend,
            eval_rows,
            [edge_a, edge_b],
            operation=args.operation,
            max_new_tokens=args.max_new_tokens,
            stop_strings=stop_strings,
        )
        pair_summary = _summary_metric(pair_rows, objective_terms)
        score_a = single_cache[str(edge_a["edge_id"])]["objective_score"]
        score_b = single_cache[str(edge_b["edge_id"])]["objective_score"]
        pair_score = pair_summary["objective_score"]
        synergy = pair_score - score_a - score_b + base_score
        conflict = max(0.0, -synergy)
        summary_rows.append(
            {
                "edge_id_a": edge_a["edge_id"],
                "edge_id_b": edge_b["edge_id"],
                "operation": args.operation,
                "objective_a": score_a,
                "objective_b": score_b,
                "objective_baseline": base_score,
                "objective_pair": pair_score,
                "synergy": synergy,
                "conflict": conflict,
                **{f"pair_{key}": value for key, value in pair_summary.items()},
            }
        )
        for row in pair_rows:
            output_rows.append(
                {
                    "edge_id_a": edge_a["edge_id"],
                    "edge_id_b": edge_b["edge_id"],
                    "operation": args.operation,
                    **row,
                }
            )

    dump_jsonl(args.out_rows_jsonl, output_rows)
    dump_csv(args.out_summary_csv, summary_rows)
    print(
        f"[rcm-eval-edge-pairs] edges={len(edges)} pairs={len(edge_pairs)} eval_rows={len(eval_rows)} "
        f"out={args.out_summary_csv}"
    )


if __name__ == "__main__":
    main()
