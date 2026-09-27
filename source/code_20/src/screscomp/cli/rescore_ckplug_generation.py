from __future__ import annotations

import argparse
import csv
import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from screscomp.cli.score_ckplug_generation import _classify_generation
from screscomp.data import dump_csv, dump_jsonl, load_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rescore CK-PLUG generations with a different alias policy.")
    p.add_argument("--generations_jsonl", type=Path, required=True)
    p.add_argument("--data_json", type=Path, required=True)
    p.add_argument("--out_generations_jsonl", type=Path, required=True)
    p.add_argument("--out_summary_csv", type=Path, required=True)
    p.add_argument("--alias_policy", choices=["answer_only", "raw", "safe_raw"], default="raw")
    p.add_argument("--max_rows", type=int, default=None)
    return p.parse_args()


def _normalize_answer(text: str) -> str:
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _norm_text(value: Any) -> str:
    return " ".join(str(value).strip().split())


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


def _safe_alias(alias: Any) -> bool:
    text = _norm_text(alias)
    key = _normalize_answer(text)
    if not key:
        return False
    if len(key) <= 2:
        return False
    if len(key) == 3 and key.isupper():
        return False
    if key in {"do", "in", "us", "uk", "dr", "rd", "dc", "fw", "mf"}:
        return False
    return True


def _answers(item: dict[str, Any], answer_key: str, alias_key: str, policy: str) -> list[str]:
    answer = _norm_text(item.get(answer_key, ""))
    aliases = item.get(alias_key, []) or []
    if policy == "answer_only":
        return [answer]
    if policy == "raw":
        return _dedup([answer, *aliases])
    if policy == "safe_raw":
        return _dedup([answer, *[alias for alias in aliases if _safe_alias(alias)]])
    raise ValueError(f"Unsupported alias policy: {policy}")


def _remove_overlaps(orig_answers: list[str], cf_answers: list[str]) -> tuple[list[str], list[str]]:
    orig_main = _normalize_answer(orig_answers[0]) if orig_answers else ""
    cf_main = _normalize_answer(cf_answers[0]) if cf_answers else ""
    orig_norms = {_normalize_answer(x) for x in orig_answers}
    cf_norms = {_normalize_answer(x) for x in cf_answers}
    overlaps = (orig_norms & cf_norms) - {orig_main, cf_main}
    if not overlaps:
        return orig_answers, cf_answers
    return (
        [x for x in orig_answers if _normalize_answer(x) not in overlaps],
        [x for x in cf_answers if _normalize_answer(x) not in overlaps],
    )


def _summarize(rows: list[dict]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["method"]].append(row)

    output: list[dict[str, Any]] = []
    for method, group in sorted(groups.items()):
        n = len(group)
        counts = Counter(row["outcome"] for row in group)
        fine_counts = Counter(row.get("fine_outcome", row["outcome"]) for row in group)
        source_fine_counts = Counter(row.get("source_fine_outcome", row["outcome"]) for row in group)
        form_fine_counts = Counter(row.get("form_fine_outcome", "") for row in group)
        ps = sum(1 for row in group if row["cf_hit"]) / n if n else 0.0
        po = sum(1 for row in group if row["orig_hit"]) / n if n else 0.0
        denom = ps + po
        output.append(
            {
                "method": method,
                "n": n,
                "ps": ps,
                "po": po,
                "mr": po / denom if denom > 0 else 0.0,
                "context_only_rate": counts["context_only"] / n if n else 0.0,
                "prior_only_rate": counts["prior_only"] / n if n else 0.0,
                "both_rate": counts["both"] / n if n else 0.0,
                "neither_rate": counts["neither"] / n if n else 0.0,
                "mixed_rate": fine_counts["mixed"] / n if n else 0.0,
                "neither_verbose_rate": fine_counts["neither_verbose"] / n if n else 0.0,
                "neither_short_rate": fine_counts["neither_short"] / n if n else 0.0,
                "short_exact_rate": fine_counts["short_exact"] / n if n else 0.0,
                "source_fine_mixed_rate": source_fine_counts["mixed"] / n if n else 0.0,
                "source_fine_neither_verbose_rate": source_fine_counts["neither_verbose"] / n if n else 0.0,
                "source_fine_neither_short_rate": source_fine_counts["neither_short"] / n if n else 0.0,
                "form_fine_short_exact_rate": form_fine_counts["short_exact"] / n if n else 0.0,
                "form_fine_context_verbose_rate": form_fine_counts["context_verbose"] / n if n else 0.0,
                "form_fine_context_inexact_rate": form_fine_counts["context_inexact"] / n if n else 0.0,
                "cf_em_rate": sum(1 for row in group if row["cf_em"]) / n if n else 0.0,
                "orig_em_rate": sum(1 for row in group if row["orig_em"]) / n if n else 0.0,
                "mean_output_chars": mean(len(row["prediction"]) for row in group) if group else 0.0,
            }
        )
    return output


def main() -> None:
    args = parse_args()
    data = json.loads(args.data_json.read_text(encoding="utf-8-sig"))
    source_by_index = {idx: item for idx, item in enumerate(data)}
    rows = load_jsonl(args.generations_jsonl)
    if args.max_rows is not None:
        allowed_ids = {f"confiqa_{idx:06d}" for idx in range(args.max_rows)}
        rows = [row for row in rows if row.get("sample_id") in allowed_ids]

    rescored: list[dict] = []
    for row in rows:
        item = source_by_index[int(row["source_index"])]
        orig_answers = _answers(item, "orig_answer", "orig_alias", args.alias_policy)
        cf_answers = _answers(item, "cf_answer", "cf_alias", args.alias_policy)
        if args.alias_policy == "safe_raw":
            orig_answers, cf_answers = _remove_overlaps(orig_answers, cf_answers)
        classified = _classify_generation(row["prediction"], orig_answers, cf_answers)
        rescored.append(
            {
                **row,
                "alias_policy_rescored": args.alias_policy,
                "orig_answers_rescored": orig_answers,
                "cf_answers_rescored": cf_answers,
                **classified,
            }
        )

    dump_jsonl(args.out_generations_jsonl, rescored)
    dump_csv(args.out_summary_csv, _summarize(rescored))
    print(f"[rescore-ckplug-generation] rows={len(rescored)} alias_policy={args.alias_policy}")


if __name__ == "__main__":
    main()
