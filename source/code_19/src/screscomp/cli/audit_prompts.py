from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from screscomp.data import dump_json, load_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit rendered prompts and split/template isolation.")
    p.add_argument("--rendered_jsonl", type=Path, required=True, help="Input 09_rendered_samples.jsonl.")
    p.add_argument("--split_manifest", type=Path, default=None, help="Optional split manifest for leakage checks.")
    p.add_argument("--out_json", type=Path, default=None, help="Optional JSON report path.")
    p.add_argument("--strict", action="store_true", help="Exit with code 1 if any audit check fails.")
    return p.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mask_first(text: str, needle: str) -> str | None:
    if not needle:
        return None
    idx = text.find(needle)
    if idx < 0:
        return None
    return f"{text[:idx]}{{ANSWER}}{text[idx + len(needle):]}"


def _source_block(prompt: str) -> tuple[str, str] | None:
    marker = "\n\nQuestion:"
    idx = prompt.find(marker)
    if idx >= 0:
        return prompt[:idx], prompt[idx:]
    marker = "Question:"
    idx = prompt.find(marker)
    if idx >= 0:
        return prompt[:idx], prompt[idx:]
    return None


def _mask_claim_in_source(prompt: str, needle: str) -> str | None:
    if not needle:
        return None
    blocks = _source_block(prompt)
    if blocks is None:
        return None
    source, suffix = blocks
    contextual_patterns = [
        (" is ", "."),
        (" gives ", " as "),
        (" reports ", " for "),
        (" names ", " as "),
        (" as ", "."),
    ]
    for prefix, tail in contextual_patterns:
        token = f"{prefix}{needle}{tail}"
        idx = source.find(token)
        if idx >= 0:
            answer_idx = idx + len(prefix)
            masked_source = f"{source[:answer_idx]}{{ANSWER}}{source[answer_idx + len(needle):]}"
            return masked_source + suffix

    matches = [idx for idx in range(len(source)) if source.startswith(needle, idx)]
    if len(matches) != 1:
        return None
    idx = matches[0]
    masked_source = f"{source[:idx]}{{ANSWER}}{source[idx + len(needle):]}"
    return masked_source + suffix


def _question_suffix(prompt: str) -> str | None:
    marker = "\n\nQuestion:"
    idx = prompt.find(marker)
    if idx >= 0:
        return prompt[idx + 2 :]
    idx = prompt.find("Question:")
    if idx >= 0:
        return prompt[idx:]
    return None


def _option_block(prompt: str) -> str | None:
    idx = prompt.rfind("\nA. ")
    if idx >= 0:
        return prompt[idx + 1 :]
    idx = prompt.find("A. ")
    if idx >= 0:
        return prompt[idx:]
    return None


def _add_failure(failures: list[dict], row: dict, check: str, detail: str) -> None:
    failures.append(
        {
            "check": check,
            "detail": detail,
            "render_id": row.get("render_id"),
            "sample_id": row.get("sample_id"),
            "fact_id": row.get("fact_id"),
            "template_family": row.get("template_family"),
            "split": row.get("split"),
        }
    )


def _audit_prompt_row(row: dict, failures: list[dict]) -> None:
    prompts = row["prompts"]
    prior = row["prior_answer"]
    counter = row["counter_prior_answer"]

    c_p_masked = _mask_claim_in_source(prompts["c_p"], prior)
    c_e_masked = _mask_claim_in_source(prompts["c_e"], counter)
    if c_p_masked is None or c_e_masked is None or c_p_masked != c_e_masked:
        _add_failure(failures, row, "e1_claim_entity_only", "c_p/c_e differ by more than the first claim answer.")

    swap_c_p_masked = _mask_claim_in_source(prompts["swap_c_p"], prior)
    swap_c_e_masked = _mask_claim_in_source(prompts["swap_c_e"], counter)
    if swap_c_p_masked is None or swap_c_e_masked is None or swap_c_p_masked != swap_c_e_masked:
        _add_failure(
            failures,
            row,
            "e1_swapped_claim_entity_only",
            "swap_c_p/swap_c_e differ by more than the first claim answer.",
        )

    c1_according = prompts["e2_c1_doc_according"]
    c1_actual = prompts["e2_c1_doc_actual"]
    if counter not in c1_according or counter not in c1_actual:
        _add_failure(failures, row, "e2_c1_same_claim", "C1 prompts do not preserve the counter claim answer.")
    if _option_block(c1_according) != _option_block(c1_actual):
        _add_failure(failures, row, "e2_c1_same_options", "C1 prompts changed the A/B option block.")

    ix_sp_acc = prompts.get("ix_source_prior_according")
    ix_sc_acc = prompts.get("ix_source_counter_according")
    ix_sp_act = prompts.get("ix_source_prior_actual")
    ix_sc_act = prompts.get("ix_source_counter_actual")
    if not all(isinstance(x, str) for x in (ix_sp_acc, ix_sc_acc, ix_sp_act, ix_sc_act)):
        _add_failure(failures, row, "interaction_four_cell_missing", "Missing one or more source/objective four-cell prompts.")
    else:
        if ix_sp_acc != prompts["c_p"] or ix_sc_acc != prompts["c_e"] or ix_sc_act != prompts["e2_c1_doc_actual"]:
            _add_failure(
                failures,
                row,
                "interaction_alias_mismatch",
                "Interaction aliases do not match the existing c_p/c_e/e2 counter-actual prompts.",
            )
        sp_act_masked = _mask_claim_in_source(ix_sp_act, prior)
        sc_act_masked = _mask_claim_in_source(ix_sc_act, counter)
        if sp_act_masked is None or sc_act_masked is None or sp_act_masked != sc_act_masked:
            _add_failure(
                failures,
                row,
                "interaction_actual_claim_entity_only",
                "source-prior/source-counter actual prompts differ by more than the first claim answer.",
            )
        if _option_block(ix_sp_acc) != _option_block(ix_sp_act):
            _add_failure(failures, row, "interaction_prior_same_options", "Prior-source prompts changed options.")
        if _option_block(ix_sc_acc) != _option_block(ix_sc_act):
            _add_failure(failures, row, "interaction_counter_same_options", "Counter-source prompts changed options.")

    swap_ix_sp_acc = prompts.get("swap_ix_source_prior_according")
    swap_ix_sc_acc = prompts.get("swap_ix_source_counter_according")
    swap_ix_sp_act = prompts.get("swap_ix_source_prior_actual")
    swap_ix_sc_act = prompts.get("swap_ix_source_counter_actual")
    if not all(isinstance(x, str) for x in (swap_ix_sp_acc, swap_ix_sc_acc, swap_ix_sp_act, swap_ix_sc_act)):
        _add_failure(
            failures,
            row,
            "swap_interaction_four_cell_missing",
            "Missing one or more swapped source/objective four-cell prompts.",
        )
    else:
        if (
            swap_ix_sp_acc != prompts["swap_c_p"]
            or swap_ix_sc_acc != prompts["swap_c_e"]
            or swap_ix_sc_act != prompts["swap_e2_c1_doc_actual"]
        ):
            _add_failure(
                failures,
                row,
                "swap_interaction_alias_mismatch",
                "Swapped interaction aliases do not match existing swapped prompts.",
            )
        swap_sp_act_masked = _mask_claim_in_source(swap_ix_sp_act, prior)
        swap_sc_act_masked = _mask_claim_in_source(swap_ix_sc_act, counter)
        if swap_sp_act_masked is None or swap_sc_act_masked is None or swap_sp_act_masked != swap_sc_act_masked:
            _add_failure(
                failures,
                row,
                "swap_interaction_actual_claim_entity_only",
                "Swapped source-prior/source-counter actual prompts differ by more than the first claim answer.",
            )
        if _option_block(swap_ix_sp_acc) != _option_block(swap_ix_sp_act):
            _add_failure(failures, row, "swap_interaction_prior_same_options", "Swapped prior-source prompts changed options.")
        if _option_block(swap_ix_sc_acc) != _option_block(swap_ix_sc_act):
            _add_failure(
                failures,
                row,
                "swap_interaction_counter_same_options",
                "Swapped counter-source prompts changed options.",
            )

    c2_doc = prompts["e2_c2_doc_actual"]
    c2_user = prompts["e2_c2_user_actual"]
    if counter not in c2_doc or counter not in c2_user:
        _add_failure(failures, row, "e2_c2_same_claim", "C2 prompts do not preserve the counter claim answer.")
    if _question_suffix(c2_doc) != _question_suffix(c2_user):
        _add_failure(failures, row, "e2_c2_same_task", "C2 prompts changed task wording while changing source label.")
    if _option_block(c2_doc) != _option_block(c2_user):
        _add_failure(failures, row, "e2_c2_same_options", "C2 prompts changed the A/B option block.")

    if "swap_e2_c2_doc_actual" in prompts and "swap_e2_c2_user_actual" in prompts:
        swap_c2_doc = prompts["swap_e2_c2_doc_actual"]
        swap_c2_user = prompts["swap_e2_c2_user_actual"]
        if counter not in swap_c2_doc or counter not in swap_c2_user:
            _add_failure(
                failures,
                row,
                "swap_e2_c2_same_claim",
                "Swapped C2 prompts do not preserve the counter claim answer.",
            )
        if _question_suffix(swap_c2_doc) != _question_suffix(swap_c2_user):
            _add_failure(
                failures,
                row,
                "swap_e2_c2_same_task",
                "Swapped C2 prompts changed task wording while changing source label.",
            )
        if _option_block(swap_c2_doc) != _option_block(swap_c2_user):
            _add_failure(failures, row, "swap_e2_c2_same_options", "Swapped C2 prompts changed the A/B option block.")


def _audit_manifest(rows: list[dict], manifest: dict, failures: list[dict]) -> None:
    cross_split_subjects = manifest.get("cross_split_subjects") or []
    if cross_split_subjects:
        failures.append(
            {
                "check": "split_subject_leakage",
                "detail": f"Subjects cross splits: {cross_split_subjects[:10]}",
            }
        )

    buckets = manifest.get("template_bucket_by_family") or {}
    by_split = manifest.get("template_families_by_split") or {}
    discovery_templates = set(by_split.get("discovery") or [])
    heldout_templates = set(by_split.get("test") or [])
    overlap = discovery_templates & heldout_templates
    if overlap:
        failures.append(
            {
                "check": "template_bucket_overlap",
                "detail": f"Discovery/test template overlap: {sorted(overlap)}",
            }
        )

    sample_to_split = manifest.get("sample_to_split") or {}
    for row in rows:
        sample_id = row.get("sample_id")
        split = row.get("split")
        family = row.get("template_family")

        expected_split = sample_to_split.get(sample_id)
        if expected_split is not None and expected_split != split:
            _add_failure(
                failures,
                row,
                "sample_split_mismatch",
                f"Rendered split {split} does not match manifest split {expected_split}.",
            )

        allowed = set(by_split.get(split) or [])
        if allowed and family not in allowed:
            _add_failure(
                failures,
                row,
                "template_not_allowed_for_split",
                f"Template {family} is not allowed for split {split}.",
            )

        bucket = buckets.get(family)
        if split == "discovery" and bucket != "discovery":
            _add_failure(failures, row, "heldout_template_in_discovery", f"Template bucket is {bucket}.")
        if split == "test" and bucket != "heldout":
            _add_failure(failures, row, "discovery_template_in_test", f"Template bucket is {bucket}.")


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.rendered_jsonl)
    failures: list[dict] = []

    for row in rows:
        _audit_prompt_row(row, failures)

    if args.split_manifest is not None:
        _audit_manifest(rows=rows, manifest=_load_json(args.split_manifest), failures=failures)

    counts = Counter(f["check"] for f in failures)
    report = {
        "rendered_rows": len(rows),
        "failure_count": len(failures),
        "failures_by_check": dict(sorted(counts.items())),
        "failures": failures,
    }

    if args.out_json is not None:
        dump_json(args.out_json, report)

    print(f"[audit] rows={len(rows)} failures={len(failures)} checks={dict(sorted(counts.items()))}")
    if args.strict and failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
