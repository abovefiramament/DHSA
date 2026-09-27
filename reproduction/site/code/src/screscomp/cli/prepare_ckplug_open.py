from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import string
from pathlib import Path
from typing import Any

from screscomp.data import dump_csv, dump_json, dump_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare CK-PLUG-style open-generation rows with curated aliases.")
    p.add_argument("--dataset", choices=["confiqa"], required=True)
    p.add_argument("--data_json", type=Path, required=True)
    p.add_argument("--out_jsonl", type=Path, required=True)
    p.add_argument("--alias_jsonl", type=Path, action="append", default=[])
    p.add_argument("--alias_glob", type=str, default="")
    p.add_argument("--max_rows", type=int, default=None)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--target_rows", type=int, default=None)
    p.add_argument("--audit_dir", type=Path, default=None)
    p.add_argument("--schema", type=str, default="base", choices=["base", "instr"])
    p.add_argument("--alias_policy", type=str, default="raw", choices=["qwen", "raw", "answer_only"])
    p.add_argument(
        "--official_prompt_contract",
        choices=["ckplug_nested_v1", "context_dpo_official_v1"],
        default="ckplug_nested_v1",
    )
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


def _official_rag_prompt(
    question: str,
    context: str,
    schema: str,
    prompt_contract: str,
) -> str:
    if prompt_contract == "context_dpo_official_v1":
        prompt = f"{context}\nQ: {question}\nA: "
        if schema == "instr":
            return "Instruction: read the given information and answer the corresponding question.\n\n" + prompt
        return prompt
    if prompt_contract != "ckplug_nested_v1":
        raise ValueError(f"Unsupported official prompt contract: {prompt_contract}")
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


def _source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _convert_confiqa(
    args: argparse.Namespace,
    alias_rows: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = _load_json(args.data_json)
    source_start = int(args.start)
    initial_stop = len(data)
    if args.max_rows is not None:
        initial_stop = min(len(data), source_start + int(args.max_rows))
    scan_stop = len(data) if args.target_rows is not None else initial_stop
    rows: list[dict[str, Any]] = []
    admitted_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    inspected_stop_exclusive = source_start
    for idx in range(source_start, scan_stop):
        item = data[idx]
        inspected_stop_exclusive = idx + 1
        sample_id = f"confiqa_{idx:06d}"
        question = _norm_text(item.get("question", ""))
        context = _norm_text(item.get("cf_context", ""))
        orig_answer = _norm_text(item.get("orig_answer", ""))
        cf_answer = _norm_text(item.get("cf_answer", ""))
        missing = [
            key
            for key, value in (
                ("question", question),
                ("cf_context", context),
                ("orig_answer", orig_answer),
                ("cf_answer", cf_answer),
            )
            if not value
        ]
        if missing:
            rejected_rows.append(
                {
                    "sample_id": sample_id,
                    "source_index": idx,
                    "reason": "missing_required:" + "+".join(missing),
                    "question": question,
                    "orig_answer": orig_answer,
                    "cf_answer": cf_answer,
                }
            )
            continue
        if _normalize_answer(orig_answer) == _normalize_answer(cf_answer):
            rejected_rows.append(
                {
                    "sample_id": sample_id,
                    "source_index": idx,
                    "reason": "identical_competition_endpoints",
                    "question": question,
                    "orig_answer": orig_answer,
                    "cf_answer": cf_answer,
                }
            )
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
            rejected_rows.append(
                {
                    "sample_id": sample_id,
                    "source_index": idx,
                    "reason": "empty_endpoint_after_alias_cleanup",
                    "question": question,
                    "orig_answer": orig_answer,
                    "cf_answer": cf_answer,
                }
            )
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
                    "official_rag": _official_rag_prompt(
                        question,
                        context,
                        args.schema,
                        args.official_prompt_contract,
                    ),
                    "strong_rag": _strong_prompt(question, context),
                    "verbose_rag": _verbose_prompt(question, context),
                },
            }
        )
        admitted_rows.append(
            {
                "sample_id": sample_id,
                "source_index": idx,
                "admission_role": "initial_window" if idx < initial_stop else "replacement",
                "question": question,
                "orig_answer": orig_answer,
                "cf_answer": cf_answer,
            }
        )
        if args.target_rows is not None and len(rows) == int(args.target_rows):
            break
    if args.target_rows is not None and len(rows) != int(args.target_rows):
        raise ValueError(
            f"requested {int(args.target_rows)} valid rows from source index {source_start}, "
            f"but only found {len(rows)} before source exhaustion"
        )
    replacements = [row for row in admitted_rows if row["admission_role"] == "replacement"]
    audit = {
        "dataset": "confiqa",
        "source": str(args.data_json.resolve()),
        "source_sha256": _source_sha256(args.data_json),
        "source_rows": len(data),
        "source_start": source_start,
        "initial_stop_exclusive": initial_stop,
        "target_valid_rows": args.target_rows,
        "scan_stop_exclusive": inspected_stop_exclusive,
        "inspected_rows": inspected_stop_exclusive - source_start,
        "admitted_rows": len(rows),
        "initial_window_admitted_rows": len(admitted_rows) - len(replacements),
        "replacement_rows": len(replacements),
        "rejected_rows": len(rejected_rows),
        "admitted_source_indices": [row["source_index"] for row in admitted_rows],
        "rejected_source_indices": [row["source_index"] for row in rejected_rows],
        "replacement_source_indices": [row["source_index"] for row in replacements],
        "admitted_records": admitted_rows,
        "rejected_records": rejected_rows,
        "replacement_records": replacements,
    }
    return rows, audit


def main() -> None:
    args = parse_args()
    alias_rows = _load_alias_rows(args)
    if args.dataset == "confiqa":
        rows, audit = _convert_confiqa(args, alias_rows)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    dump_jsonl(args.out_jsonl, rows)
    if args.audit_dir is not None:
        args.audit_dir.mkdir(parents=True, exist_ok=True)
        dump_csv(args.audit_dir / "admitted_source_rows.csv", audit.pop("admitted_records"))
        dump_jsonl(args.audit_dir / "rejected_source_rows.jsonl", audit.pop("rejected_records"))
        dump_jsonl(args.audit_dir / "replacement_source_rows.jsonl", audit.pop("replacement_records"))
        audit["output_jsonl"] = str(args.out_jsonl.resolve())
        dump_json(args.audit_dir / "sample_admission_manifest.json", audit)
    print(
        f"[prepare-ckplug-open] dataset={args.dataset} rows={len(rows)} "
        f"rejected={audit['rejected_rows']} replacements={audit['replacement_rows']} "
        f"scan_stop={audit['scan_stop_exclusive']} aliases={len(alias_rows)} out={args.out_jsonl}"
    )


if __name__ == "__main__":
    main()
