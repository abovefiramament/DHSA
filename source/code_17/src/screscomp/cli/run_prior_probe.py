from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from screscomp.data import dump_csv, dump_jsonl, load_jsonl
from screscomp.modeling import TransformersABBackend
from screscomp.prompts import SUPPORTED_TEMPLATE_FAMILIES, render_prior_only


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run prior probe and filter unstable/biased pairs.")
    p.add_argument("--in_jsonl", type=Path, required=True, help="Input prompt-ready pairs JSONL.")
    p.add_argument("--model", type=str, required=True, help="Transformers model id/path.")
    p.add_argument("--out_retained", type=Path, required=True, help="Output retained JSONL.")
    p.add_argument("--out_excluded", type=Path, required=True, help="Output excluded JSONL.")
    p.add_argument("--out_funnel_csv", type=Path, default=None, help="Optional sample funnel CSV path.")
    p.add_argument(
        "--out_exclusion_stats_csv",
        type=Path,
        default=None,
        help="Optional exclusion-reason stats CSV path.",
    )
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
    p.add_argument("--tie_eps", type=float, default=1e-6, help="Tie threshold on A/B logits.")
    p.add_argument(
        "--stability_template_families",
        type=str,
        default="",
        help="Optional comma-separated template families for template-level prior stability probing.",
    )
    p.add_argument(
        "--require_template_stability",
        action="store_true",
        help="Exclude pairs whose prior semantic answer changes across stability templates.",
    )
    p.add_argument(
        "--require_prior_equals_true",
        action="store_true",
        help="Exclude pairs unless the model's semantic prior answer matches the verified true answer.",
    )
    p.add_argument(
        "--out_template_scores_jsonl",
        type=Path,
        default=None,
        help="Optional JSONL path with per-template prior probe scores.",
    )
    return p.parse_args()


def _append_exclusion(excluded: list[dict], sample_id: str, fact_id: str, reason: str) -> None:
    excluded.append({"sample_id": sample_id, "fact_id": fact_id, "exclusion_reason": reason})


def _parse_family_csv(raw: str) -> list[str]:
    parts = [x.strip() for x in raw.split(",")]
    families = [x for x in parts if x]
    unknown = sorted(set(families) - SUPPORTED_TEMPLATE_FAMILIES)
    if unknown:
        raise ValueError(f"Unsupported template families: {unknown}")
    return families


def _score_prior_pair(
    backend: TransformersABBackend,
    prior_prompt: str,
    swapped_prior_prompt: str,
    candidate_1: str,
    candidate_2: str,
    tie_eps: float,
) -> tuple[dict, str | None]:
    try:
        a, b = backend.score_ab(prior_prompt)
        sa, sb = backend.score_ab(swapped_prior_prompt)
    except ValueError:
        return {}, "tokenization_invalid"
    except Exception:
        return {}, "generation_error"

    if abs(a - b) <= tie_eps:
        return {}, "prior_tie"

    if abs(sa - sb) <= tie_eps:
        return {}, "prior_unstable"

    prior_semantic = candidate_1 if a > b else candidate_2
    swap_prior_semantic = candidate_2 if sa > sb else candidate_1
    if prior_semantic != swap_prior_semantic:
        return {}, "swap_unstable"

    if prior_semantic == candidate_1:
        prior_label = "A"
        counter_label = "B"
        b_prior = a - b
        b_prior_swapped = sb - sa
        counter = candidate_2
    else:
        prior_label = "B"
        counter_label = "A"
        b_prior = b - a
        b_prior_swapped = sa - sb
        counter = candidate_1

    return (
        {
            "prior_answer": prior_semantic,
            "counter_prior_answer": counter,
            "prior_label": prior_label,
            "counter_prior_label": counter_label,
            "B_prior": b_prior,
            "B_prior_swapped": b_prior_swapped,
            "logits": {
                "prior_only": {"A": a, "B": b},
                "swapped_prior_only": {"A": sa, "B": sb},
            },
        },
        None,
    )


def _emit_reports(
    raw_count: int,
    retained: list[dict],
    excluded: list[dict],
    out_funnel_csv: Path | None,
    out_exclusion_stats_csv: Path | None,
) -> None:
    reason_counts = Counter(row["exclusion_reason"] for row in excluded)
    template_dirty_count = sum(count for reason, count in reason_counts.items() if reason.startswith("template_"))
    prior_stable_count = raw_count - reason_counts["tokenization_invalid"] - reason_counts["prior_tie"] - reason_counts[
        "generation_error"
    ]
    swap_stable_count = prior_stable_count - reason_counts["prior_unstable"] - reason_counts["swap_unstable"]
    template_clean_count = swap_stable_count - template_dirty_count
    final_count = len(retained)

    if out_funnel_csv is not None:
        dump_csv(
            out_funnel_csv,
            [
                {
                    "raw_count": raw_count,
                    "prior_stable_count": max(prior_stable_count, 0),
                    "swap_stable_count": max(swap_stable_count, 0),
                    "template_clean_count": max(template_clean_count, 0),
                    "final_count": final_count,
                }
            ],
        )

    if out_exclusion_stats_csv is not None:
        rows: list[dict[str, float | int | str]] = []
        excluded_total = len(excluded)
        for reason in sorted(reason_counts.keys()):
            count = reason_counts[reason]
            rows.append(
                {
                    "exclusion_reason": reason,
                    "count": count,
                    "ratio_of_excluded": (count / excluded_total) if excluded_total else 0.0,
                    "ratio_of_raw": (count / raw_count) if raw_count else 0.0,
                }
            )
        dump_csv(out_exclusion_stats_csv, rows)


def main() -> None:
    args = parse_args()
    backend = TransformersABBackend(
        model_name_or_path=args.model,
        device=args.device,
        use_chat_template=args.use_chat_template,
        torch_dtype=args.torch_dtype,
    )
    rows = load_jsonl(args.in_jsonl)
    stability_families = _parse_family_csv(args.stability_template_families)

    retained: list[dict] = []
    excluded: list[dict] = []
    template_scores: list[dict] = []

    for row in rows:
        sample_id = row["sample_id"]
        fact_id = row["fact_id"]
        c1 = row["candidate_1"]
        c2 = row["candidate_2"]

        base_score, reason = _score_prior_pair(
            backend=backend,
            prior_prompt=row["prior_only_prompt"],
            swapped_prior_prompt=row["swapped_prior_only_prompt"],
            candidate_1=c1,
            candidate_2=c2,
            tie_eps=args.tie_eps,
        )
        if reason is not None:
            _append_exclusion(excluded, sample_id, fact_id, reason)
            continue

        b_prior_by_template: dict[str, float] = {}
        b_prior_swapped_by_template: dict[str, float] = {}
        prior_answer_by_template: dict[str, str] = {}
        template_logits: dict[str, dict] = {}
        template_failure_reason: str | None = None

        for family in stability_families:
            prior_prompt = render_prior_only(row["question"], c1, c2, family=family)
            swapped_prior_prompt = render_prior_only(row["question"], c2, c1, family=family)
            score, template_reason = _score_prior_pair(
                backend=backend,
                prior_prompt=prior_prompt,
                swapped_prior_prompt=swapped_prior_prompt,
                candidate_1=c1,
                candidate_2=c2,
                tie_eps=args.tie_eps,
            )
            if template_reason is not None:
                template_failure_reason = f"template_{template_reason}"
                break
            b_prior_by_template[family] = score["B_prior"]
            b_prior_swapped_by_template[family] = score["B_prior_swapped"]
            prior_answer_by_template[family] = score["prior_answer"]
            template_logits[family] = score["logits"]
            template_scores.append(
                {
                    "sample_id": sample_id,
                    "fact_id": fact_id,
                    "template_family": family,
                    "prior_answer": score["prior_answer"],
                    "counter_prior_answer": score["counter_prior_answer"],
                    "B_prior": score["B_prior"],
                    "B_prior_swapped": score["B_prior_swapped"],
                    "probe_model": backend.model_id,
                    "probe_logits": score["logits"],
                }
            )

        if template_failure_reason is not None:
            _append_exclusion(excluded, sample_id, fact_id, template_failure_reason)
            continue

        if args.require_template_stability and prior_answer_by_template:
            answers = set(prior_answer_by_template.values())
            answers.add(base_score["prior_answer"])
            if len(answers) > 1:
                _append_exclusion(excluded, sample_id, fact_id, "template_unstable")
                continue

        if args.require_prior_equals_true:
            true_answer = row["true_answer"]
            if base_score["prior_answer"] != true_answer:
                _append_exclusion(excluded, sample_id, fact_id, "prior_not_true")
                continue
            if prior_answer_by_template:
                template_truth_mismatch = any(answer != true_answer for answer in prior_answer_by_template.values())
                if template_truth_mismatch:
                    _append_exclusion(excluded, sample_id, fact_id, "template_prior_not_true")
                    continue

        retained.append(
            {
                **row,
                "option_A": c1,
                "option_B": c2,
                "prior_answer": base_score["prior_answer"],
                "counter_prior_answer": base_score["counter_prior_answer"],
                "prior_label": base_score["prior_label"],
                "counter_prior_label": base_score["counter_prior_label"],
                "B_prior": base_score["B_prior"],
                "B_prior_swapped": base_score["B_prior_swapped"],
                "B_prior_by_template": b_prior_by_template,
                "B_prior_swapped_by_template": b_prior_swapped_by_template,
                "prior_answer_by_template": prior_answer_by_template,
                "probe_model": backend.model_id,
                "probe_logits": base_score["logits"],
                "probe_logits_by_template": template_logits,
            }
        )

    dump_jsonl(args.out_retained, retained)
    dump_jsonl(args.out_excluded, excluded)
    if args.out_template_scores_jsonl is not None:
        dump_jsonl(args.out_template_scores_jsonl, template_scores)
    _emit_reports(
        raw_count=len(rows),
        retained=retained,
        excluded=excluded,
        out_funnel_csv=args.out_funnel_csv,
        out_exclusion_stats_csv=args.out_exclusion_stats_csv,
    )
    print(f"[probe] retained={len(retained)} excluded={len(excluded)}")


if __name__ == "__main__":
    main()
