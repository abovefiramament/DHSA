from __future__ import annotations

import argparse
import glob
import json
import re
import string
from pathlib import Path
from typing import Any

from screscomp.data import dump_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare CK-PLUG-style open-generation rows with curated aliases.")
    p.add_argument("--dataset", choices=["confiqa"], required=True)
    p.add_argument("--data_json", type=Path, required=True)
    p.add_argument("--out_jsonl", type=Path, required=True)
    p.add_argument("--alias_jsonl", type=Path, action="append", default=[])
    p.add_argument("--alias_glob", type=str, default="")
    p.add_argument("--max_rows", type=int, default=None)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--schema", type=str, default="base", choices=["base", "instr"])
    p.add_argument("--alias_policy", type=str, default="raw", choices=["qwen", "raw", "answer_only"])
    p.add_argument("--include_medium_added", action="store_true")
    return p.parse_args()


def _norm_text(value: Any) -> str:
    return " ".join(str(value).strip().split())


def _normalize_answer(text: str) -> str:
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _dedup(items: list[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = _norm_text(item)
        key = _normalize_answer(text)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_alias_rows(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    paths = list(args.alias_jsonl)
    if args.alias_glob:
        paths.extend(Path(p) for p in sorted(glob.glob(args.alias_glob)))
    aliases: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in _load_jsonl(path):
            sample_id = str(row.get("sample_id", ""))
            if sample_id:
                aliases[sample_id] = row
    return aliases


def _qa_to_prompt_baseline(question: str, context: str, schema: str) -> str:
    if schema == "instr":
        return (
            "Instruction: read the given information and answer the corresponding question.\n\n"
            f"{context}\nQ:{question}\nA:"
        )
    return f"{context}\nQ:{question}\nA:"


def _base_prompt(question: str) -> str:
    return f"Q: {question}\nA:"


def _strong_no_rag_prompt(question: str) -> str:
    return (
        "Answer the question. Return only the short answer.\n\n"
        f"Q: {question}\nA:"
    )


def _strong_prompt(question: str, context: str) -> str:
    return (
        "Read the given information and answer the question using the given information. "
        "Return only the short answer.\n\n"
        f"{context}\n\nQ: {question}\nA:"
    )


def _verbose_no_rag_prompt(question: str) -> str:
    return (
        "Answer the question with a full sentence and a brief explanation.\n\n"
        f"Q: {question}\nA:"
    )


def _verbose_prompt(question: str, context: str) -> str:
    return (
        "Read the given information and answer the question using the given information. "
        "Answer with a full sentence and a brief explanation.\n\n"
        f"{context}\n\nQ: {question}\nA:"
    )


def _official_no_rag_prompt(question: str) -> str:
    return f"Q: {question}\nA: "


def _official_rag_prompt(question: str, context: str, schema: str) -> str:
    query = _official_no_rag_prompt(question)
    if schema == "base":
        return f"{context}\nQ:{query}\nA:"
    if schema == "instr":
        return (
            "Instruction: read the given information and answer the corresponding question.\n\n"
            f"{context}\nQ:{query}\nA:"
        )
    return _qa_to_prompt_baseline(question, context, schema)


def _answers_with_aliases(
    *,
    answer: str,
    raw_aliases: list[Any],
    qwen_row: dict[str, Any] | None,
    clean_key: str,
    added_key: str,
    include_medium_added: bool,
) -> list[str]:
    if qwen_row is None:
        return _dedup([answer, *raw_aliases])
    risk = str(qwen_row.get("alias_risk", "low"))
    added_allowed = risk == "low" or (risk == "medium" and include_medium_added)
    added = qwen_row.get(added_key, []) if added_allowed else []
    return _dedup([answer, *qwen_row.get(clean_key, []), *added])


def _remove_cross_overlaps(orig_answers: list[str], cf_answers: list[str]) -> tuple[list[str], list[str], list[str]]:
    orig_main = _normalize_answer(orig_answers[0]) if orig_answers else ""
    cf_main = _normalize_answer(cf_answers[0]) if cf_answers else ""
    orig_norms = {_normalize_answer(x) for x in orig_answers}
    cf_norms = {_normalize_answer(x) for x in cf_answers}
    overlaps = sorted((orig_norms & cf_norms) - {orig_main, cf_main})
    if not overlaps:
        return orig_answers, cf_answers, []
    orig_clean = [x for x in orig_answers if _normalize_answer(x) not in overlaps]
    cf_clean = [x for x in cf_answers if _normalize_answer(x) not in overlaps]
    return orig_clean, cf_clean, overlaps


def _convert_confiqa(args: argparse.Namespace, alias_rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    data = _load_json(args.data_json)
    data = data[args.start :]
    if args.max_rows is not None:
        data = data[: args.max_rows]

    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(data, start=args.start):
        sample_id = f"confiqa_{idx:06d}"
        question = _norm_text(item.get("question", ""))
        context = _norm_text(item.get("cf_context", ""))
        orig_answer = _norm_text(item.get("orig_answer", ""))
        cf_answer = _norm_text(item.get("cf_answer", ""))
        if not question or not context or not orig_answer or not cf_answer:
            continue
        if _normalize_answer(orig_answer) == _normalize_answer(cf_answer):
            continue

        qwen_row = alias_rows.get(sample_id)
        if args.alias_policy == "answer_only":
            orig_answers = [orig_answer]
            cf_answers = [cf_answer]
            alias_source = "answer_only"
            alias_risk = "none"
        elif args.alias_policy == "raw":
            orig_answers = _dedup([orig_answer, *(item.get("orig_alias", []) or [])])
            cf_answers = _dedup([cf_answer, *(item.get("cf_alias", []) or [])])
            alias_source = "raw"
            alias_risk = "raw"
        else:
            orig_answers = _answers_with_aliases(
                answer=orig_answer,
                raw_aliases=item.get("orig_alias", []) or [],
                qwen_row=qwen_row,
                clean_key="orig_alias_clean",
                added_key="orig_alias_added",
                include_medium_added=args.include_medium_added,
            )
            cf_answers = _answers_with_aliases(
                answer=cf_answer,
                raw_aliases=item.get("cf_alias", []) or [],
                qwen_row=qwen_row,
                clean_key="cf_alias_clean",
                added_key="cf_alias_added",
                include_medium_added=args.include_medium_added,
            )
            alias_source = "qwen" if qwen_row else "raw"
            alias_risk = qwen_row.get("alias_risk", "raw") if qwen_row else "raw"
        orig_answers, cf_answers, removed_overlaps = _remove_cross_overlaps(orig_answers, cf_answers)
        if not orig_answers or not cf_answers:
            continue

        rows.append(
            {
                "sample_id": sample_id,
                "dataset": "confiqa",
                "source_index": idx,
                "question": question,
                "context": context,
                "orig_answer": orig_answer,
                "cf_answer": cf_answer,
                "orig_answers": orig_answers,
                "cf_answers": cf_answers,
                "alias_policy": args.alias_policy,
                "alias_source": alias_source,
                "alias_risk": alias_risk,
                "removed_alias_overlaps": removed_overlaps,
                "prompts": {
                    "base_no_rag": _base_prompt(question),
                    "strong_no_rag": _strong_no_rag_prompt(question),
                    "verbose_no_rag": _verbose_no_rag_prompt(question),
                    "official_no_rag": _official_no_rag_prompt(question),
                    "base_rag": _qa_to_prompt_baseline(question, context, args.schema),
                    "official_rag": _official_rag_prompt(question, context, args.schema),
                    "strong_rag": _strong_prompt(question, context),
                    "verbose_rag": _verbose_prompt(question, context),
                },
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    alias_rows = _load_alias_rows(args)
    if args.dataset == "confiqa":
        rows = _convert_confiqa(args, alias_rows)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    dump_jsonl(args.out_jsonl, rows)
    print(f"[prepare-ckplug-open] dataset={args.dataset} rows={len(rows)} aliases={len(alias_rows)} out={args.out_jsonl}")


if __name__ == "__main__":
    main()
