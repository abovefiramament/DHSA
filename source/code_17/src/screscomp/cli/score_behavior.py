from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean

from screscomp.data import dump_csv, dump_jsonl, load_jsonl
from screscomp.modeling import TransformersABBackend


NORMAL_PROMPT_KEYS = [
    "prior_only",
    "c_p",
    "c_e",
    "ix_source_prior_according",
    "ix_source_counter_according",
    "ix_source_prior_actual",
    "ix_source_counter_actual",
    "e2_c1_doc_according",
    "e2_c1_doc_actual",
    "e2_c2_doc_actual",
    "e2_c2_user_actual",
    "format_only",
]

SWAPPED_PROMPT_KEYS = [
    "swap_prior_only",
    "swap_c_p",
    "swap_c_e",
    "swap_ix_source_prior_according",
    "swap_ix_source_counter_according",
    "swap_ix_source_prior_actual",
    "swap_ix_source_counter_actual",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score rendered A/B prompts and summarize behavioral effects.")
    p.add_argument("--rendered_jsonl", type=Path, required=True, help="Input 09_rendered_samples.jsonl.")
    p.add_argument("--model", type=str, required=True, help="Transformers model id/path.")
    p.add_argument("--out_scores_jsonl", type=Path, required=True, help="Output per-rendered-sample scores JSONL.")
    p.add_argument("--out_summary_csv", type=Path, required=True, help="Output summary CSV.")
    p.add_argument("--device", type=str, default="auto", help="cpu|cuda|auto")
    p.add_argument(
        "--use_chat_template",
        action="store_true",
        help="Wrap each prompt as one user message with tokenizer.apply_chat_template(..., add_generation_prompt=True).",
    )
    p.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
        help="Torch dtype for model loading.",
    )
    p.add_argument("--max_rows", type=int, default=None, help="Optional limit for quick smoke tests.")
    return p.parse_args()


def _opposite_label(label: str) -> str:
    if label == "A":
        return "B"
    if label == "B":
        return "A"
    raise ValueError(f"Unsupported label: {label}")


def _semantic_b(logit_a: float, logit_b: float, prior_label: str) -> float:
    if prior_label == "A":
        return logit_a - logit_b
    if prior_label == "B":
        return logit_b - logit_a
    raise ValueError(f"Unsupported prior label: {prior_label}")


def _score_prompt_map(backend: TransformersABBackend, row: dict) -> dict:
    prompts = row["prompts"]
    prior_label = row["prior_label"]
    swapped_prior_label = _opposite_label(prior_label)

    b_scores: dict[str, float] = {}
    logits: dict[str, dict[str, float]] = {}

    for key in NORMAL_PROMPT_KEYS:
        logit_a, logit_b = backend.score_ab(prompts[key])
        logits[key] = {"A": logit_a, "B": logit_b}
        b_scores[key] = _semantic_b(logit_a=logit_a, logit_b=logit_b, prior_label=prior_label)

    for key in SWAPPED_PROMPT_KEYS:
        logit_a, logit_b = backend.score_ab(prompts[key])
        logits[key] = {"A": logit_a, "B": logit_b}
        b_scores[key] = _semantic_b(logit_a=logit_a, logit_b=logit_b, prior_label=swapped_prior_label)

    interaction_content_according = b_scores["ix_source_prior_according"] - b_scores["ix_source_counter_according"]
    interaction_content_actual = b_scores["ix_source_prior_actual"] - b_scores["ix_source_counter_actual"]
    interaction_objective_prior = b_scores["ix_source_prior_actual"] - b_scores["ix_source_prior_according"]
    interaction_objective_counter = b_scores["ix_source_counter_actual"] - b_scores["ix_source_counter_according"]
    interaction_did = interaction_content_according - interaction_content_actual

    swap_interaction_content_according = (
        b_scores["swap_ix_source_prior_according"] - b_scores["swap_ix_source_counter_according"]
    )
    swap_interaction_content_actual = (
        b_scores["swap_ix_source_prior_actual"] - b_scores["swap_ix_source_counter_actual"]
    )
    swap_interaction_objective_prior = (
        b_scores["swap_ix_source_prior_actual"] - b_scores["swap_ix_source_prior_according"]
    )
    swap_interaction_objective_counter = (
        b_scores["swap_ix_source_counter_actual"] - b_scores["swap_ix_source_counter_according"]
    )
    swap_interaction_did = swap_interaction_content_according - swap_interaction_content_actual

    effects = {
        "doc_counter_effect": b_scores["c_e"] - b_scores["c_p"],
        "doc_counter_shift_from_prior": b_scores["c_e"] - b_scores["prior_only"],
        "doc_prior_shift_from_prior": b_scores["c_p"] - b_scores["prior_only"],
        "doc_support_gap": b_scores["c_p"] - b_scores["c_e"],
        "counter_reduces_prior_vs_cp": b_scores["c_e"] < b_scores["c_p"],
        "counter_reduces_prior_vs_prior_only": b_scores["c_e"] < b_scores["prior_only"],
        "counter_flips_prior": b_scores["c_e"] < 0,
        "e2_task_delta_actual_minus_according": b_scores["e2_c1_doc_actual"] - b_scores["e2_c1_doc_according"],
        "e2_source_delta_user_minus_doc": b_scores["e2_c2_user_actual"] - b_scores["e2_c2_doc_actual"],
        "swap_doc_support_gap": b_scores["swap_c_p"] - b_scores["swap_c_e"],
        "interaction_content_according": interaction_content_according,
        "interaction_content_actual": interaction_content_actual,
        "interaction_objective_prior": interaction_objective_prior,
        "interaction_objective_counter": interaction_objective_counter,
        "interaction_did": interaction_did,
        "interaction_did_positive": interaction_did > 0,
        "interaction_actual_smaller_than_according": abs(interaction_content_actual) < abs(interaction_content_according),
        "interaction_counter_objective_exceeds_prior_objective": (
            interaction_objective_counter > interaction_objective_prior
        ),
        "swap_interaction_content_according": swap_interaction_content_according,
        "swap_interaction_content_actual": swap_interaction_content_actual,
        "swap_interaction_objective_prior": swap_interaction_objective_prior,
        "swap_interaction_objective_counter": swap_interaction_objective_counter,
        "swap_interaction_did": swap_interaction_did,
        "swap_interaction_did_positive": swap_interaction_did > 0,
        "swap_interaction_actual_smaller_than_according": (
            abs(swap_interaction_content_actual) < abs(swap_interaction_content_according)
        ),
        "swap_interaction_counter_objective_exceeds_prior_objective": (
            swap_interaction_objective_counter > swap_interaction_objective_prior
        ),
    }

    return {
        "render_id": row["render_id"],
        "sample_id": row["sample_id"],
        "fact_id": row["fact_id"],
        "subject_id": row["subject_id"],
        "relation": row["relation"],
        "domain": row["domain"],
        "split": row["split"],
        "template_family": row["template_family"],
        "prior_answer": row["prior_answer"],
        "counter_prior_answer": row["counter_prior_answer"],
        "prior_label": row["prior_label"],
        "counter_prior_label": row["counter_prior_label"],
        "b_scores": b_scores,
        "effects": effects,
        "logits": logits,
    }


def _rate(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row["effects"][key]) / len(rows)


def _mean_effect(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    return mean(float(row["effects"][key]) for row in rows)


def _mean_b(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    return mean(float(row["b_scores"][key]) for row in rows)


def _summarize_group(name: str, rows: list[dict]) -> dict[str, object]:
    return {
        "group": name,
        "n": len(rows),
        "mean_B_prior_only": _mean_b(rows, "prior_only"),
        "mean_B_c_p": _mean_b(rows, "c_p"),
        "mean_B_c_e": _mean_b(rows, "c_e"),
        "mean_doc_support_gap": _mean_effect(rows, "doc_support_gap"),
        "mean_doc_counter_effect": _mean_effect(rows, "doc_counter_effect"),
        "mean_doc_counter_shift_from_prior": _mean_effect(rows, "doc_counter_shift_from_prior"),
        "frac_counter_reduces_prior_vs_cp": _rate(rows, "counter_reduces_prior_vs_cp"),
        "frac_counter_reduces_prior_vs_prior_only": _rate(rows, "counter_reduces_prior_vs_prior_only"),
        "frac_counter_flips_prior": _rate(rows, "counter_flips_prior"),
        "mean_e2_task_delta_actual_minus_according": _mean_effect(rows, "e2_task_delta_actual_minus_according"),
        "mean_e2_source_delta_user_minus_doc": _mean_effect(rows, "e2_source_delta_user_minus_doc"),
        "mean_swap_doc_support_gap": _mean_effect(rows, "swap_doc_support_gap"),
        "mean_interaction_content_according": _mean_effect(rows, "interaction_content_according"),
        "mean_interaction_content_actual": _mean_effect(rows, "interaction_content_actual"),
        "mean_interaction_objective_prior": _mean_effect(rows, "interaction_objective_prior"),
        "mean_interaction_objective_counter": _mean_effect(rows, "interaction_objective_counter"),
        "mean_interaction_did": _mean_effect(rows, "interaction_did"),
        "frac_interaction_did_positive": _rate(rows, "interaction_did_positive"),
        "frac_interaction_actual_smaller_than_according": _rate(rows, "interaction_actual_smaller_than_according"),
        "frac_interaction_counter_objective_exceeds_prior_objective": _rate(
            rows,
            "interaction_counter_objective_exceeds_prior_objective",
        ),
        "mean_swap_interaction_content_according": _mean_effect(rows, "swap_interaction_content_according"),
        "mean_swap_interaction_content_actual": _mean_effect(rows, "swap_interaction_content_actual"),
        "mean_swap_interaction_objective_prior": _mean_effect(rows, "swap_interaction_objective_prior"),
        "mean_swap_interaction_objective_counter": _mean_effect(rows, "swap_interaction_objective_counter"),
        "mean_swap_interaction_did": _mean_effect(rows, "swap_interaction_did"),
        "frac_swap_interaction_did_positive": _rate(rows, "swap_interaction_did_positive"),
        "frac_swap_interaction_actual_smaller_than_according": _rate(
            rows,
            "swap_interaction_actual_smaller_than_according",
        ),
        "frac_swap_interaction_counter_objective_exceeds_prior_objective": _rate(
            rows,
            "swap_interaction_counter_objective_exceeds_prior_objective",
        ),
    }


def _summarize(rows: list[dict]) -> list[dict[str, object]]:
    summary = [_summarize_group("all", rows)]

    by_split: dict[str, list[dict]] = defaultdict(list)
    by_template: dict[str, list[dict]] = defaultdict(list)
    by_relation: dict[str, list[dict]] = defaultdict(list)
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row)
        by_template[row["template_family"]].append(row)
        by_relation[row["relation"]].append(row)
        by_domain[row["domain"]].append(row)

    for split in sorted(by_split):
        summary.append(_summarize_group(f"split:{split}", by_split[split]))
    for template_family in sorted(by_template):
        summary.append(_summarize_group(f"template:{template_family}", by_template[template_family]))
    for relation in sorted(by_relation):
        summary.append(_summarize_group(f"relation:{relation}", by_relation[relation]))
    for domain in sorted(by_domain):
        summary.append(_summarize_group(f"domain:{domain}", by_domain[domain]))

    return summary


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.rendered_jsonl)
    if args.max_rows is not None:
        rows = rows[: args.max_rows]

    backend = TransformersABBackend(
        model_name_or_path=args.model,
        device=args.device,
        use_chat_template=args.use_chat_template,
        torch_dtype=args.torch_dtype,
    )

    scored_rows = [_score_prompt_map(backend=backend, row=row) for row in rows]
    dump_jsonl(args.out_scores_jsonl, scored_rows)
    summary = _summarize(scored_rows)
    dump_csv(args.out_summary_csv, summary)
    print(f"[score-behavior] rows={len(scored_rows)} summary_rows={len(summary)}")


if __name__ == "__main__":
    main()
