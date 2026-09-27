from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, desc=None):
        return iterable

from screscomp.cli.score_component_competition import (
    _component_id,
    _score_b,
    _score_component_patch_b,
)
from screscomp.cli.score_steering_utility import _load_summary_rows, _rank_components
from screscomp.data import dump_csv, dump_jsonl, load_jsonl
from screscomp.modeling import TransformersABBackend


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compute fixed pre-generation prediction scores for CK-PLUG-style open generations. "
            "This does not train a classifier or tune thresholds."
        )
    )
    p.add_argument("--eval_jsonl", type=Path, required=True)
    p.add_argument("--generations_jsonl", type=Path, required=True)
    p.add_argument("--component_summary_csv", type=Path, required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--out_predictions_jsonl", type=Path, required=True)
    p.add_argument("--out_summary_csv", type=Path, required=True)
    p.add_argument("--target_method", type=str, default="strong_rag")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--use_chat_template", action="store_true")
    p.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
    )
    p.add_argument("--selection_pairs", type=str, default="e1_content,e1_content_swapped")
    p.add_argument("--top_k_components", type=int, default=4)
    p.add_argument("--max_rows", type=int, default=None)
    return p.parse_args()


def _ab_block(option_a: str, option_b: str) -> str:
    return (
        f"A. {option_a}\n"
        f"B. {option_b}\n\n"
        "Answer with A or B only.\n"
        "Answer:"
    )


def _qa_prompt(question: str, option_a: str, option_b: str) -> str:
    return f"Q: {question}\n\n{_ab_block(option_a, option_b)}"


def _context_prompt(context: str, question: str, option_a: str, option_b: str) -> str:
    return f"{context}\n\nQ: {question}\n\n{_ab_block(option_a, option_b)}"


def _prior_context(question: str, answer: str) -> str:
    return f"Background:\nThe answer to the question is {answer}."


def _make_ab_prompts(row: dict, swapped: bool) -> tuple[str, str, str]:
    question = row["question"]
    context = row["context"]
    orig = row["orig_answer"]
    cf = row["cf_answer"]
    if swapped:
        option_a, option_b = cf, orig
        prior_label = "B"
    else:
        option_a, option_b = orig, cf
        prior_label = "A"
    high_prompt = _context_prompt(_prior_context(question, orig), question, option_a, option_b)
    low_prompt = _context_prompt(context, question, option_a, option_b)
    return high_prompt, low_prompt, prior_label


def _read_generations(path: Path, target_method: str) -> dict[str, dict]:
    rows = load_jsonl(path)
    selected: dict[str, dict] = {}
    for row in rows:
        if row.get("method") == target_method:
            selected[row["sample_id"]] = row
    if not selected:
        raise ValueError(f"No generation rows found for target_method={target_method!r}.")
    return selected


def _mean_dict(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = rows[0].keys()
    return {key: mean(float(row[key]) for row in rows) for key in keys}


def _score_one_label_order(
    backend: TransformersABBackend,
    row: dict,
    component_specs: list[tuple[int, str]],
    *,
    swapped: bool,
) -> dict[str, float]:
    high_prompt, low_prompt, prior_label = _make_ab_prompts(row, swapped=swapped)
    high_b, high_logits = _score_b(backend, high_prompt, prior_label)
    low_b, low_logits = _score_b(backend, low_prompt, prior_label)
    high_acts = backend.capture_component_last_token(high_prompt, component_specs)
    low_acts = backend.capture_component_last_token(low_prompt, component_specs)

    c_vals: list[float] = []
    i_vals: list[float] = []
    min_vals: list[float] = []
    both_positive = 0
    for layer_idx, component_type in component_specs:
        low_with_high_b, _ = _score_component_patch_b(
            backend=backend,
            prompt=low_prompt,
            prior_label=prior_label,
            layer_idx=layer_idx,
            component_type=component_type,
            replacement=high_acts[(layer_idx, component_type)],
        )
        high_with_low_b, _ = _score_component_patch_b(
            backend=backend,
            prompt=high_prompt,
            prior_label=prior_label,
            layer_idx=layer_idx,
            component_type=component_type,
            replacement=low_acts[(layer_idx, component_type)],
        )
        c_t = float(low_with_high_b - low_b)
        i_t = float(high_b - high_with_low_b)
        c_vals.append(c_t)
        i_vals.append(i_t)
        min_vals.append(min(c_t, i_t))
        if c_t > 0 and i_t > 0:
            both_positive += 1

    prior_logit_low = low_logits[prior_label]
    cf_label = "B" if prior_label == "A" else "A"
    context_logit_low = low_logits[cf_label]
    max_answer_logit = max(prior_logit_low, context_logit_low)
    return {
        "B_prior_margin_context": float(low_b),
        "B_prior_margin_prior_context": float(high_b),
        "B_context_shift": float(high_b - low_b),
        "answer_logit_max": float(max_answer_logit),
        "C_prior_restore": mean(c_vals),
        "I_context_overwrite": mean(i_vals),
        "min_CI": mean(min_vals),
        "frac_both_positive": both_positive / len(component_specs),
    }


def _score_prediction_row(
    backend: TransformersABBackend,
    row: dict,
    component_specs: list[tuple[int, str]],
) -> dict[str, float]:
    normal = _score_one_label_order(backend, row, component_specs, swapped=False)
    swapped = _score_one_label_order(backend, row, component_specs, swapped=True)
    base = _mean_dict([normal, swapped])
    c_prior = base["C_prior_restore"]
    i_context = base["I_context_overwrite"]
    b_context = base["B_prior_margin_context"]
    max_logit = base["answer_logit_max"]

    base.update(
        {
            "margin_score_context": -b_context,
            "margin_score_prior": b_context,
            "margin_score_both_or_uncertain": -abs(b_context),
            "margin_score_neither": -max_logit,
            "ci_score_context": i_context - c_prior,
            "ci_score_prior": c_prior - i_context,
            "ci_score_both": min(c_prior, i_context),
            "ci_score_neither": -max(c_prior, i_context),
        }
    )
    return base


def _binary_labels(row: dict) -> dict[str, bool]:
    outcome = row["outcome"]
    return {
        "context_only": outcome == "context_only",
        "prior_only": outcome == "prior_only",
        "both": outcome == "both",
        "neither": outcome == "neither",
        "cf_hit": bool(row["cf_hit"]),
        "orig_hit": bool(row["orig_hit"]),
        "failure_not_context_only": outcome != "context_only",
    }


def _average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def _auroc(labels: list[bool], scores: list[float]) -> float | None:
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = _average_ranks(scores)
    pos_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _average_precision(labels: list[bool], scores: list[float]) -> float | None:
    n_pos = sum(labels)
    if n_pos == 0:
        return None
    ordered = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    hits = 0
    precisions: list[float] = []
    for rank, (_score, label) in enumerate(ordered, start=1):
        if label:
            hits += 1
            precisions.append(hits / rank)
    return sum(precisions) / n_pos


def _summarize(pred_rows: list[dict]) -> list[dict[str, Any]]:
    score_map = {
        "context_only": ["ci_score_context", "margin_score_context"],
        "prior_only": ["ci_score_prior", "margin_score_prior"],
        "both": ["ci_score_both", "margin_score_both_or_uncertain"],
        "neither": ["ci_score_neither", "margin_score_neither"],
        "cf_hit": ["ci_score_context", "margin_score_context"],
        "orig_hit": ["ci_score_prior", "margin_score_prior"],
        "failure_not_context_only": ["ci_score_prior", "ci_score_both", "ci_score_neither", "margin_score_prior"],
    }
    rows: list[dict[str, Any]] = []
    for target, score_names in score_map.items():
        labels = [bool(row[target]) for row in pred_rows]
        positives = sum(labels)
        for score_name in score_names:
            scores = [float(row[score_name]) for row in pred_rows]
            rows.append(
                {
                    "target_method": pred_rows[0]["target_method"],
                    "target": target,
                    "score_name": score_name,
                    "n": len(pred_rows),
                    "positives": positives,
                    "positive_rate": positives / len(pred_rows) if pred_rows else 0.0,
                    "auroc": _auroc(labels, scores),
                    "average_precision": _average_precision(labels, scores),
                    "mean_score_positive": mean(s for s, label in zip(scores, labels) if label) if positives else None,
                    "mean_score_negative": mean(s for s, label in zip(scores, labels) if not label)
                    if positives < len(pred_rows)
                    else None,
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    ranked = _rank_components(
        summary_rows=_load_summary_rows(args.component_summary_csv),
        selection_pairs=[x.strip() for x in args.selection_pairs.split(",") if x.strip()],
    )
    selected = ranked[: args.top_k_components]
    component_specs = [(int(row["layer_idx"]), str(row["component_type"])) for row in selected]

    eval_rows = load_jsonl(args.eval_jsonl)
    if args.max_rows is not None:
        eval_rows = eval_rows[: args.max_rows]
    gen_by_id = _read_generations(args.generations_jsonl, args.target_method)
    eval_rows = [row for row in eval_rows if row["sample_id"] in gen_by_id]
    if not eval_rows:
        raise ValueError("No eval rows matched generation rows.")

    backend = TransformersABBackend(
        model_name_or_path=args.model,
        device=args.device,
        use_chat_template=args.use_chat_template,
        torch_dtype=args.torch_dtype,
    )

    output_rows: list[dict[str, Any]] = []
    for row in tqdm(eval_rows, desc="ckplug prediction"):
        generation_row = gen_by_id[row["sample_id"]]
        scores = _score_prediction_row(backend, row, component_specs)
        labels = _binary_labels(generation_row)
        output_rows.append(
            {
                "sample_id": row["sample_id"],
                "dataset": row["dataset"],
                "source_index": row["source_index"],
                "target_method": args.target_method,
                "outcome": generation_row["outcome"],
                "prediction": generation_row["prediction"],
                "orig_answer": row["orig_answer"],
                "cf_answer": row["cf_answer"],
                "top_k_components": args.top_k_components,
                "component_ids": ",".join(_component_id(*spec) for spec in component_specs),
                **labels,
                **scores,
            }
        )

    dump_jsonl(args.out_predictions_jsonl, output_rows)
    dump_csv(args.out_summary_csv, _summarize(output_rows))
    print(
        f"[score-ckplug-prediction] rows={len(output_rows)} target_method={args.target_method} "
        f"components={','.join(_component_id(*spec) for spec in component_specs)}"
    )


if __name__ == "__main__":
    main()
