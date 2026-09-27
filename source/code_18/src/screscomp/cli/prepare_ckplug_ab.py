from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from screscomp.data import dump_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Convert CK-PLUG knowledge-reliance datasets into the screscomp A/B "
            "factual arbitration format. This is an adapter for our diagnostic, "
            "not a reimplementation of CK-PLUG decoding."
        )
    )
    p.add_argument("--dataset", choices=["nq", "confiqa", "mquake"], required=True)
    p.add_argument("--out_jsonl", type=Path, required=True)
    p.add_argument("--max_rows", type=int, default=None)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--split", type=str, default="external_test")
    p.add_argument("--template_family", type=str, default="ckplug_base")
    p.add_argument("--data_json", type=Path, default=None, help="ConFiQA or MQuAKE JSON file.")
    p.add_argument("--orig_json", type=Path, default=None, help="NQ original-answer JSON file.")
    p.add_argument("--counter_json", type=Path, default=None, help="NQ counterfactual-context JSON file.")
    return p.parse_args()


def _norm(text: Any) -> str:
    return " ".join(str(text).strip().split())


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _first_answer(value: Any) -> str:
    if isinstance(value, list):
        for item in value:
            text = _norm(item)
            if text:
                return text
        return ""
    return _norm(value)


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


def _simple_fact_context(question: str, answer: str) -> str:
    return f"Background:\nThe answer to the question is {answer}."


def _mk_prompts(question: str, context: str, prior_answer: str, context_answer: str) -> dict[str, str]:
    prior_context = _simple_fact_context(question=question, answer=prior_answer)
    normal_prior = _qa_prompt(question, prior_answer, context_answer)
    normal_cp = _context_prompt(prior_context, question, prior_answer, context_answer)
    normal_ce = _context_prompt(context, question, prior_answer, context_answer)
    swap_prior = _qa_prompt(question, context_answer, prior_answer)
    swap_cp = _context_prompt(prior_context, question, context_answer, prior_answer)
    swap_ce = _context_prompt(context, question, context_answer, prior_answer)

    return {
        "prior_only": normal_prior,
        "c_p": normal_cp,
        "c_e": normal_ce,
        "swap_prior_only": swap_prior,
        "swap_c_p": swap_cp,
        "swap_c_e": swap_ce,
        "ix_source_prior_according": normal_cp,
        "ix_source_counter_according": normal_ce,
        "ix_source_prior_actual": normal_cp,
        "ix_source_counter_actual": normal_ce,
        "swap_ix_source_prior_according": swap_cp,
        "swap_ix_source_counter_according": swap_ce,
        "swap_ix_source_prior_actual": swap_cp,
        "swap_ix_source_counter_actual": swap_ce,
        "e2_c1_doc_according": normal_ce,
        "e2_c1_doc_actual": normal_ce,
        "e2_c2_doc_actual": normal_ce,
        "e2_c2_user_actual": normal_ce,
        "swap_e2_c1_doc_according": swap_ce,
        "swap_e2_c1_doc_actual": swap_ce,
        "swap_e2_c2_doc_actual": swap_ce,
        "swap_e2_c2_user_actual": swap_ce,
        "format_only": normal_prior,
        "swap_format_only": swap_prior,
        "entity_answer": f"{context}\n\nQ: {question}\n\nAnswer with the answer only.\nAnswer:",
    }


def _mk_row(
    *,
    dataset: str,
    idx: int,
    split: str,
    template_family: str,
    question: str,
    context: str,
    prior_answer: str,
    context_answer: str,
    source_id: str = "",
) -> dict[str, Any] | None:
    question = _norm(question)
    context = _norm(context)
    prior_answer = _first_answer(prior_answer)
    context_answer = _first_answer(context_answer)
    if not question or not context or not prior_answer or not context_answer:
        return None
    if prior_answer.casefold() == context_answer.casefold():
        return None

    sample_id = f"{dataset}_{idx:06d}"
    prompts = _mk_prompts(
        question=question,
        context=context,
        prior_answer=prior_answer,
        context_answer=context_answer,
    )
    return {
        "render_id": f"{sample_id}::{template_family}",
        "sample_id": sample_id,
        "fact_id": source_id or sample_id,
        "subject_id": source_id or sample_id,
        "subject": source_id or sample_id,
        "domain": f"ckplug_{dataset}",
        "relation": f"ckplug_{dataset}",
        "question": question,
        "answer_type": "entity",
        "true_answer": prior_answer,
        "true_answer_id": prior_answer,
        "split_group_id": source_id or sample_id,
        "option_A": prior_answer,
        "option_B": context_answer,
        "prior_answer": prior_answer,
        "counter_prior_answer": context_answer,
        "counter_answer_id": context_answer,
        "counter_difficulty": "ckplug_conflict_context",
        "counter_rationale": "counterfactual context answer from CK-PLUG evaluation data",
        "prior_label": "A",
        "counter_prior_label": "B",
        "B_prior": 0.0,
        "B_prior_swapped": 0.0,
        "split": split,
        "template_family": template_family,
        "prompts": prompts,
        "exclusion_reason": None,
    }


def _slice(rows: list[Any], start: int, max_rows: int | None) -> list[Any]:
    rows = rows[start:]
    if max_rows is not None:
        rows = rows[:max_rows]
    return rows


def _convert_confiqa(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.data_json is None:
        raise ValueError("--data_json is required for --dataset confiqa")
    data = _slice(_load_json(args.data_json), args.start, args.max_rows)
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(data, start=args.start):
        row = _mk_row(
            dataset="confiqa",
            idx=idx,
            split=args.split,
            template_family=args.template_family,
            question=item.get("question", ""),
            context=item.get("cf_context", ""),
            prior_answer=item.get("orig_answer", ""),
            context_answer=item.get("cf_answer", ""),
            source_id=_norm(item.get("id", "")),
        )
        if row is not None:
            rows.append(row)
    return rows


def _convert_nq(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.orig_json is None or args.counter_json is None:
        raise ValueError("--orig_json and --counter_json are required for --dataset nq")
    orig = _load_json(args.orig_json)
    counter = _load_json(args.counter_json)
    paired = list(zip(orig, counter))
    paired = _slice(paired, args.start, args.max_rows)
    rows: list[dict[str, Any]] = []
    for idx, (orig_item, counter_item) in enumerate(paired, start=args.start):
        row = _mk_row(
            dataset="nq",
            idx=idx,
            split=args.split,
            template_family=args.template_family,
            question=counter_item.get("question", ""),
            context=counter_item.get("context", ""),
            prior_answer=orig_item.get("answer", ""),
            context_answer=counter_item.get("answer", ""),
            source_id=_norm(counter_item.get("id", "")),
        )
        if row is not None:
            rows.append(row)
    return rows


def _convert_mquake(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.data_json is None:
        raise ValueError("--data_json is required for --dataset mquake")
    data = _slice(_load_json(args.data_json), args.start, args.max_rows)
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(data, start=args.start):
        facts: list[str] = []
        for rewrite in item.get("requested_rewrite", []):
            prompt = _norm(rewrite.get("prompt", ""))
            subject = _norm(rewrite.get("subject", ""))
            target = _norm((rewrite.get("target_new") or {}).get("str", ""))
            if prompt and subject and target:
                facts.append(f'{prompt.format(subject)} {target}.')
        context = "Edit Knowledge: " + " ".join(facts)
        questions = item.get("questions") or []
        question = questions[0] if questions else ""
        row = _mk_row(
            dataset="mquake",
            idx=idx,
            split=args.split,
            template_family=args.template_family,
            question=question,
            context=context,
            prior_answer=item.get("answer", ""),
            context_answer=item.get("new_answer", ""),
            source_id=_norm(item.get("case_id", "")),
        )
        if row is not None:
            rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    if args.dataset == "confiqa":
        rows = _convert_confiqa(args)
    elif args.dataset == "nq":
        rows = _convert_nq(args)
    elif args.dataset == "mquake":
        rows = _convert_mquake(args)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    dump_jsonl(args.out_jsonl, rows)
    print(f"[prepare-ckplug-ab] dataset={args.dataset} rows={len(rows)} out={args.out_jsonl}")


if __name__ == "__main__":
    main()
