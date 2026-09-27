from __future__ import annotations

import argparse
import json
from pathlib import Path

from screscomp.data import dump_jsonl, load_jsonl
from screscomp.prompts import (
    render_doc_prompt,
    render_entity_answer_prompt,
    render_format_only_prompt,
    render_prior_only,
    render_user_prompt,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render E1/E2/control prompts from retained pairs.")
    p.add_argument("--retained_jsonl", type=Path, required=True)
    p.add_argument("--split_manifest", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, default=Path("data/02_prompts"))
    p.add_argument(
        "--template_mode",
        type=str,
        default="assigned",
        choices=["assigned", "all_by_split"],
        help=(
            "`assigned` renders the single sampled template per sample. "
            "`all_by_split` renders every template family allowed by that sample's split."
        ),
    )
    return p.parse_args()


def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_template_family(sample_id: str, split: str, manifest: dict) -> str:
    sample_to_family = manifest.get("sample_to_template_family")
    if isinstance(sample_to_family, dict) and sample_id in sample_to_family:
        return sample_to_family[sample_id]

    by_split_multi = manifest.get("template_families_by_split")
    if isinstance(by_split_multi, dict):
        families = by_split_multi.get(split) or []
        if families:
            return families[0]

    by_split_single = manifest.get("template_family_by_split")
    if isinstance(by_split_single, dict):
        family = by_split_single.get(split)
        if family:
            return family

    return "main_v1"


def _resolve_template_families(sample_id: str, split: str, manifest: dict, mode: str) -> list[str]:
    if mode == "assigned":
        return [_resolve_template_family(sample_id=sample_id, split=split, manifest=manifest)]

    sample_to_families = manifest.get("sample_to_template_families")
    if isinstance(sample_to_families, dict):
        families = sample_to_families.get(sample_id)
        if isinstance(families, list) and families:
            return [str(f) for f in families]

    by_split_multi = manifest.get("template_families_by_split")
    if isinstance(by_split_multi, dict):
        families = by_split_multi.get(split) or []
        if families:
            return [str(f) for f in families]

    return [_resolve_template_family(sample_id=sample_id, split=split, manifest=manifest)]


def _render_prompt_bundle(row: dict, family: str) -> dict[str, str]:
    subject = row["subject"]
    relation = row["relation"]
    question = row["question"]
    option_a = row["option_A"]
    option_b = row["option_B"]
    prior = row["prior_answer"]
    counter = row["counter_prior_answer"]

    source_prior_according = render_doc_prompt(
        subject=subject,
        relation=relation,
        claim_answer=prior,
        question=question,
        option_a=option_a,
        option_b=option_b,
        objective="according",
        family=family,
    )
    source_counter_according = render_doc_prompt(
        subject=subject,
        relation=relation,
        claim_answer=counter,
        question=question,
        option_a=option_a,
        option_b=option_b,
        objective="according",
        family=family,
    )
    source_prior_actual = render_doc_prompt(
        subject=subject,
        relation=relation,
        claim_answer=prior,
        question=question,
        option_a=option_a,
        option_b=option_b,
        objective="actual",
        family=family,
    )
    source_counter_actual = render_doc_prompt(
        subject=subject,
        relation=relation,
        claim_answer=counter,
        question=question,
        option_a=option_a,
        option_b=option_b,
        objective="actual",
        family=family,
    )
    swap_source_prior_according = render_doc_prompt(
        subject=subject,
        relation=relation,
        claim_answer=prior,
        question=question,
        option_a=option_b,
        option_b=option_a,
        objective="according",
        family=family,
    )
    swap_source_counter_according = render_doc_prompt(
        subject=subject,
        relation=relation,
        claim_answer=counter,
        question=question,
        option_a=option_b,
        option_b=option_a,
        objective="according",
        family=family,
    )
    swap_source_prior_actual = render_doc_prompt(
        subject=subject,
        relation=relation,
        claim_answer=prior,
        question=question,
        option_a=option_b,
        option_b=option_a,
        objective="actual",
        family=family,
    )
    swap_source_counter_actual = render_doc_prompt(
        subject=subject,
        relation=relation,
        claim_answer=counter,
        question=question,
        option_a=option_b,
        option_b=option_a,
        objective="actual",
        family=family,
    )

    return {
        "prior_only": render_prior_only(question, option_a, option_b, family=family),
        "c_p": source_prior_according,
        "c_e": source_counter_according,
        "swap_prior_only": render_prior_only(question, option_b, option_a, family=family),
        "swap_c_p": swap_source_prior_according,
        "swap_c_e": swap_source_counter_according,
        "ix_source_prior_according": source_prior_according,
        "ix_source_counter_according": source_counter_according,
        "ix_source_prior_actual": source_prior_actual,
        "ix_source_counter_actual": source_counter_actual,
        "swap_ix_source_prior_according": swap_source_prior_according,
        "swap_ix_source_counter_according": swap_source_counter_according,
        "swap_ix_source_prior_actual": swap_source_prior_actual,
        "swap_ix_source_counter_actual": swap_source_counter_actual,
        "e2_c1_doc_according": source_counter_according,
        "e2_c1_doc_actual": source_counter_actual,
        "e2_c2_doc_actual": source_counter_actual,
        "e2_c2_user_actual": render_user_prompt(
            subject=subject,
            relation=relation,
            claim_answer=counter,
            option_a=option_a,
            option_b=option_b,
            family=family,
        ),
        "swap_e2_c1_doc_according": swap_source_counter_according,
        "swap_e2_c1_doc_actual": swap_source_counter_actual,
        "swap_e2_c2_doc_actual": swap_source_counter_actual,
        "swap_e2_c2_user_actual": render_user_prompt(
            subject=subject,
            relation=relation,
            claim_answer=counter,
            option_a=option_b,
            option_b=option_a,
            family=family,
        ),
        "format_only": render_format_only_prompt(
            option_a=option_a,
            option_b=option_b,
            label=row.get("prior_label", "A"),
        ),
        "swap_format_only": render_format_only_prompt(
            option_a=option_b,
            option_b=option_a,
            label=row.get("counter_prior_label", "B"),
        ),
        "entity_answer": render_entity_answer_prompt(
            subject=subject,
            relation=relation,
            claim_answer=counter,
            question=question,
            answer_type=row["answer_type"],
            family=family,
        ),
    }


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.retained_jsonl)
    manifest = _load_manifest(args.split_manifest)
    sample_to_split = manifest["sample_to_split"]

    e1_rows: list[dict] = []
    e2_rows: list[dict] = []
    control_rows: list[dict] = []
    merged_rows: list[dict] = []

    for row in rows:
        base_sample_id = row["sample_id"]
        split = sample_to_split[base_sample_id]
        families = _resolve_template_families(
            sample_id=base_sample_id,
            split=split,
            manifest=manifest,
            mode=args.template_mode,
        )

        for family in families:
            render_id = f"{base_sample_id}::{family}"
            prompts = _render_prompt_bundle(row=row, family=family)

            common = {
                "render_id": render_id,
                "sample_id": base_sample_id,
                "split": split,
                "template_family": family,
            }

            e1_rows.append(
                {
                    **common,
                    "prior_only_prompt": prompts["prior_only"],
                    "c_p_prompt": prompts["c_p"],
                    "c_e_prompt": prompts["c_e"],
                    "swapped_prior_only_prompt": prompts["swap_prior_only"],
                    "swapped_c_p_prompt": prompts["swap_c_p"],
                    "swapped_c_e_prompt": prompts["swap_c_e"],
                }
            )

            e2_rows.append(
                {
                    **common,
                    "c1_doc_according_prompt": prompts["e2_c1_doc_according"],
                    "c1_doc_actual_prompt": prompts["e2_c1_doc_actual"],
                    "c2_document_actual_prompt": prompts["e2_c2_doc_actual"],
                    "c2_user_actual_prompt": prompts["e2_c2_user_actual"],
                    "swapped_c1_doc_according_prompt": prompts["swap_e2_c1_doc_according"],
                    "swapped_c1_doc_actual_prompt": prompts["swap_e2_c1_doc_actual"],
                    "swapped_c2_document_actual_prompt": prompts["swap_e2_c2_doc_actual"],
                    "swapped_c2_user_actual_prompt": prompts["swap_e2_c2_user_actual"],
                }
            )

            control_rows.append(
                {
                    **common,
                    "format_only_prompt": prompts["format_only"],
                    "swapped_format_only_prompt": prompts["swap_format_only"],
                    "entity_answer_prompt": prompts["entity_answer"],
                }
            )

            merged_rows.append(
                {
                    **common,
                    "fact_id": row["fact_id"],
                    "subject_id": row["subject_id"],
                    "subject": row["subject"],
                    "domain": row["domain"],
                    "relation": row["relation"],
                    "question": row["question"],
                    "answer_type": row["answer_type"],
                    "true_answer": row["true_answer"],
                    "true_answer_id": row.get("true_answer_id", row["true_answer"]),
                    "split_group_id": row.get("split_group_id", row["subject_id"]),
                    "option_A": row["option_A"],
                    "option_B": row["option_B"],
                    "prior_answer": row["prior_answer"],
                    "counter_prior_answer": row["counter_prior_answer"],
                    "counter_answer_id": row.get("counter_answer_id", row["counter_prior_answer"]),
                    "counter_difficulty": row.get("counter_difficulty", ""),
                    "counter_rationale": row.get("counter_rationale", ""),
                    "prior_label": row["prior_label"],
                    "counter_prior_label": row["counter_prior_label"],
                    "B_prior": row["B_prior"],
                    "B_prior_swapped": row["B_prior_swapped"],
                    "prompts": prompts,
                    "exclusion_reason": None,
                }
            )

    dump_jsonl(args.out_dir / "06_e1_prompts.jsonl", e1_rows)
    dump_jsonl(args.out_dir / "07_e2_prompts.jsonl", e2_rows)
    dump_jsonl(args.out_dir / "08_control_prompts.jsonl", control_rows)
    dump_jsonl(args.out_dir / "09_rendered_samples.jsonl", merged_rows)
    print(
        f"[render] mode={args.template_mode} e1={len(e1_rows)} e2={len(e2_rows)} "
        f"control={len(control_rows)} merged={len(merged_rows)}"
    )


if __name__ == "__main__":
    main()
