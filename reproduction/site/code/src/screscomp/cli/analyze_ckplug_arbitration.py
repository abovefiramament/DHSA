from __future__ import annotations

import argparse
import random
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from screscomp.data import dump_csv, dump_jsonl, load_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Audit CK-style open-generation outputs as a multi-axis arbitration problem: "
            "source reliance, exact answer form, and verbosity."
        )
    )
    p.add_argument("--generations_jsonl", type=Path, action="append", required=True)
    p.add_argument("--out_pareto_csv", type=Path, required=True)
    p.add_argument("--out_audit_csv", type=Path, required=True)
    p.add_argument("--out_examples_jsonl", type=Path, default=None)
    p.add_argument("--examples_per_bucket", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _normalize_answer(text: str) -> str:
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _answers(row: dict[str, Any], key: str) -> list[str]:
    value = row.get(key, "")
    if isinstance(value, list):
        return [str(item) for item in value]
    if value:
        return [str(value)]
    singular = key[:-1] if key.endswith("s") else key
    value = row.get(singular, "")
    return [str(value)] if value else []


def _exact_hit(text: str, answers: list[str]) -> bool:
    norm = _normalize_answer(text)
    return any(norm == _normalize_answer(answer) for answer in answers if _normalize_answer(answer))


def _contains_hit(text: str, answers: list[str]) -> bool:
    norm = _normalize_answer(text)
    return any(_normalize_answer(answer) in norm for answer in answers if _normalize_answer(answer))


_PREFIX_PATTERNS = [
    r"^answer\s*[:\-]\s*",
    r"^the answer is\s+",
    r"^the correct answer is\s+",
    r"^it is\s+",
    r"^it's\s+",
]


def _first_answer_span(prediction: str) -> str:
    text = str(prediction).strip()
    if not text:
        return ""
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    first_line = first_line.strip(" \t\"'`")
    lowered = first_line.lower()
    for pattern in _PREFIX_PATTERNS:
        lowered_new = re.sub(pattern, "", lowered, flags=re.IGNORECASE).strip()
        if lowered_new != lowered:
            first_line = re.sub(pattern, "", first_line, flags=re.IGNORECASE).strip()
            lowered = lowered_new
    # Keep this conservative: split only at common explanation boundaries, not every period.
    first_line = re.split(r"\s+(?:because|which|who|that|and it|and was)\b", first_line, maxsplit=1)[0].strip()
    first_line = first_line.strip(" \t.;,")
    return first_line


def _bucket(row: dict[str, Any], cf_exact_first: bool, cf_exact_span: bool) -> str:
    fine = str(row.get("fine_outcome", ""))
    if fine:
        return fine
    cf_hit = bool(row.get("cf_hit"))
    orig_hit = bool(row.get("orig_hit"))
    cf_em = bool(row.get("cf_em"))
    output_chars = int(row.get("output_chars", len(str(row.get("prediction", "")))))
    if cf_em and output_chars <= 48:
        return "short_exact"
    if cf_exact_first or cf_exact_span:
        return "first_answer_cf_exact"
    if cf_hit and orig_hit:
        return "mixed"
    if cf_hit:
        return "contains_cf_not_exact"
    if orig_hit:
        return "prior_only"
    if output_chars > 48:
        return "neither_verbose"
    return "neither_short"


def _enrich_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        prediction = str(row.get("prediction", ""))
        cf_answers = _answers(row, "cf_answers") or _answers(row, "cf_answer")
        orig_answers = _answers(row, "orig_answers") or _answers(row, "orig_answer")
        first_span = _first_answer_span(prediction)
        cf_exact_first = _exact_hit(first_span, cf_answers)
        orig_exact_first = _exact_hit(first_span, orig_answers)
        cf_contains_first = _contains_hit(first_span, cf_answers)
        orig_contains_first = _contains_hit(first_span, orig_answers)
        output_chars = int(row.get("output_chars", len(prediction)))
        enriched.append(
            {
                **row,
                "first_answer_span": first_span,
                "first_answer_chars": len(first_span),
                "first_answer_cf_em": cf_exact_first,
                "first_answer_orig_em": orig_exact_first,
                "first_answer_cf_hit": cf_contains_first,
                "first_answer_orig_hit": orig_contains_first,
                "short_output": output_chars <= 48,
                "audit_bucket": _bucket(row, cf_exact_first=cf_exact_first, cf_exact_span=cf_contains_first),
            }
        )
    return enriched


def _method_metrics(group: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(group)
    counts = Counter(row["audit_bucket"] for row in group)
    fine_counts = Counter(row.get("fine_outcome", row.get("outcome", "")) for row in group)
    ps = sum(1 for row in group if row.get("cf_hit")) / n if n else 0.0
    po = sum(1 for row in group if row.get("orig_hit")) / n if n else 0.0
    denom = ps + po
    context_only = sum(1 for row in group if row.get("outcome") == "context_only") / n if n else 0.0
    prior_only = sum(1 for row in group if row.get("outcome") == "prior_only") / n if n else 0.0
    neither = sum(1 for row in group if row.get("outcome") == "neither") / n if n else 0.0
    cf_em = sum(1 for row in group if row.get("cf_em")) / n if n else 0.0
    first_cf_em = sum(1 for row in group if row.get("first_answer_cf_em")) / n if n else 0.0
    short_output = sum(1 for row in group if row.get("short_output")) / n if n else 0.0
    mean_chars = mean(int(row.get("output_chars", len(str(row.get("prediction", ""))))) for row in group) if n else 0.0
    return {
        "n": n,
        "ps": ps,
        "po": po,
        "mr": po / denom if denom else 0.0,
        "context_only_rate": context_only,
        "prior_only_rate": prior_only,
        "neither_rate": neither,
        "mixed_rate": fine_counts.get("mixed", 0) / n if n else 0.0,
        "neither_verbose_rate": fine_counts.get("neither_verbose", 0) / n if n else 0.0,
        "neither_short_rate": fine_counts.get("neither_short", 0) / n if n else 0.0,
        "short_exact_rate": fine_counts.get("short_exact", 0) / n if n else 0.0,
        "cf_em_rate": cf_em,
        "first_answer_cf_em_rate": first_cf_em,
        "short_output_rate": short_output,
        "mean_output_chars": mean_chars,
        "source_axis_score": context_only - prior_only - po,
        "form_axis_score": first_cf_em + short_output - (mean_chars / 200.0),
        "joint_axis_score": context_only + first_cf_em - po - neither - (mean_chars / 400.0),
        **{f"bucket_{key}_rate": counts[key] / n if n else 0.0 for key in sorted(counts)},
    }


def _pareto_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[str(row["method"])].append(row)
    output: list[dict[str, Any]] = []
    for method, group in sorted(by_method.items()):
        output.append({"method": method, **_method_metrics(group)})
    output.sort(key=lambda row: (-float(row["joint_axis_score"]), str(row["method"])))
    return output


def _audit_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), str(row["audit_bucket"]))].append(row)
    output: list[dict[str, Any]] = []
    for (method, bucket), group in sorted(grouped.items()):
        n = len(group)
        output.append(
            {
                "method": method,
                "audit_bucket": bucket,
                "n": n,
                "mean_output_chars": mean(int(row.get("output_chars", 0)) for row in group) if group else 0.0,
                "mean_first_answer_chars": mean(int(row.get("first_answer_chars", 0)) for row in group) if group else 0.0,
            }
        )
    return output


def _example_rows(rows: list[dict[str, Any]], *, examples_per_bucket: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), str(row["audit_bucket"]))].append(row)
    examples: list[dict[str, Any]] = []
    for (method, bucket), group in sorted(grouped.items()):
        group = list(group)
        rng.shuffle(group)
        for row in group[:examples_per_bucket]:
            examples.append(
                {
                    "method": method,
                    "audit_bucket": bucket,
                    "sample_id": row.get("sample_id", ""),
                    "orig_answer": row.get("orig_answer", ""),
                    "cf_answer": row.get("cf_answer", ""),
                    "first_answer_span": row.get("first_answer_span", ""),
                    "prediction": row.get("prediction", ""),
                }
            )
    return examples


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for path in args.generations_jsonl:
        rows.extend(load_jsonl(path))
    enriched = _enrich_rows(rows)
    dump_csv(args.out_pareto_csv, _pareto_rows(enriched))
    dump_csv(args.out_audit_csv, _audit_rows(enriched))
    if args.out_examples_jsonl is not None:
        dump_jsonl(
            args.out_examples_jsonl,
            _example_rows(enriched, examples_per_bucket=args.examples_per_bucket, seed=args.seed),
        )
    print(f"[analyze-ckplug-arbitration] rows={len(enriched)} files={len(args.generations_jsonl)}")


if __name__ == "__main__":
    main()
