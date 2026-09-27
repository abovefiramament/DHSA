from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from screscomp.data import dump_csv, dump_json, dump_jsonl


PROMPT_CANDIDATES = ("prompt", "question", "problem", "input")
INITIAL_CANDIDATES = (
    "initial_reason_steps",
    "initial_reasoning_steps",
    "initial_steps",
    "reasoning_prefix",
    "prefix",
)
CHOSEN_CANDIDATES = ("chosen", "chosen_step", "positive", "y_plus")
REJECTED_CANDIDATES = ("rejected", "rejected_step", "negative", "y_minus")
FULL_CHOSEN_CANDIDATES = ("full_chosen", "chosen_full", "full_positive")
FULL_REJECTED_CANDIDATES = ("full_rejected", "rejected_full", "full_negative")
ANSWER_CANDIDATES = ("answer", "final_answer", "gold_answer", "target")
ID_CANDIDATES = ("problem_id", "question_id", "id", "uid")


@dataclass(frozen=True, slots=True)
class PreparedPair:
    raw_index: int
    source_row_index: int
    admitted: bool
    admit_reason: str
    split: str
    sample_id: str
    pair_id: str
    problem_id: str
    prompt: str
    initial_reason_steps: str
    prefix: str
    chosen: str
    rejected: str
    full_chosen: str
    full_rejected: str
    answer: str
    pair_mode: str
    chosen_answer: str
    rejected_answer: str
    prompt_field: str
    initial_field: str
    chosen_field: str
    rejected_field: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Convert Step-DPO-style step preferences into CECM/CAST actuator pairs. "
            "The pair object is x=(prompt + initial_reason_steps), y_plus=chosen, y_minus=rejected."
        )
    )
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, default=None, help="Local .jsonl, .json, or .csv source file.")
    source.add_argument("--hf-dataset", type=str, default="", help="Optional Hugging Face dataset id.")
    p.add_argument("--hf-config", type=str, default=None)
    p.add_argument("--hf-split", type=str, default="train")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--event", type=str, default="step_correct_over_error")
    p.add_argument("--source-dataset", type=str, default="")
    p.add_argument("--source-filter-field", type=str, default="")
    p.add_argument(
        "--source-filter-contains",
        type=str,
        default="",
        help="Optional case-insensitive substring filter applied before row slicing, e.g. GSM.",
    )
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--max-source-rows", type=int, default=0)
    p.add_argument("--train-rows", type=int, default=240)
    p.add_argument("--val-rows", type=int, default=60)
    p.add_argument("--eval-rows", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument(
        "--split-unit",
        choices=["row", "problem"],
        default="row",
        help="row gives exact row targets; problem keeps all steps from the same problem in one split.",
    )
    p.add_argument("--prompt-field", type=str, default="")
    p.add_argument("--initial-field", type=str, default="")
    p.add_argument("--chosen-field", type=str, default="")
    p.add_argument("--rejected-field", type=str, default="")
    p.add_argument("--full-chosen-field", type=str, default="")
    p.add_argument("--full-rejected-field", type=str, default="")
    p.add_argument("--answer-field", type=str, default="")
    p.add_argument("--id-field", type=str, default="")
    p.add_argument(
        "--pair-mode",
        choices=["step", "full"],
        default="step",
        help="step uses prompt+initial -> chosen/rejected step; full uses prompt -> full_chosen/full_rejected.",
    )
    p.add_argument(
        "--admission-policy",
        choices=["basic", "final_answer_strict"],
        default="basic",
        help="final_answer_strict keeps only pairs where y_plus final answer matches answer and y_minus does not.",
    )
    p.add_argument(
        "--include-initial-in-full-prompt",
        action="store_true",
        help="For pair-mode=full, include initial_reason_steps in the prompt. Default is ordinary DPO prompt-only.",
    )
    p.add_argument("--prefix-separator", type=str, default="\\n\\n")
    p.add_argument("--continuation-separator", type=str, default="\\n")
    p.add_argument(
        "--keep-rejected-rows",
        action="store_true",
        help="Also write rejected source rows to pairs.csv with admitted=0 for audit.",
    )
    return p.parse_args(argv)


def _decode_escapes(value: str) -> str:
    return bytes(value, "utf-8").decode("unicode_escape")


def _load_local_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"JSONL rows must be objects: {path}")
                    rows.append(value)
        return rows
    if suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            if not all(isinstance(item, dict) for item in raw):
                raise ValueError(f"JSON list must contain objects: {path}")
            return list(raw)
        if isinstance(raw, dict):
            for key in ("data", "rows", "train", "examples"):
                value = raw.get(key)
                if isinstance(value, list):
                    if not all(isinstance(item, dict) for item in value):
                        raise ValueError(f"JSON field {key!r} must contain objects: {path}")
                    return list(value)
            if raw and all(isinstance(value, list) for value in raw.values()):
                keys = list(raw.keys())
                length = len(raw[keys[0]])
                if any(len(raw[key]) != length for key in keys):
                    raise ValueError(f"Column-oriented JSON has unequal column lengths: {path}")
                return [{key: raw[key][idx] for key in keys} for idx in range(length)]
        raise ValueError(f"Could not find record list in {path}")
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"Unsupported input format {path.suffix!r}; expected .jsonl, .json, or .csv")


def _load_hf_records(dataset: str, *, config: str | None, split: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise SystemExit(
            "Loading --hf-dataset requires the optional 'datasets' package. "
            "Install it or download the dataset to JSONL/CSV and pass --input."
        ) from exc
    loaded = load_dataset(dataset, config, split=split) if config else load_dataset(dataset, split=split)
    return [dict(item) for item in loaded]


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(_as_text(item) for item in value if _as_text(item).strip())
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _pick_field(record: dict[str, Any], explicit: str, candidates: Iterable[str]) -> tuple[str, str]:
    if explicit:
        return explicit, _as_text(record.get(explicit))
    for candidate in candidates:
        if candidate in record:
            return candidate, _as_text(record.get(candidate))
    return "", ""


def _filter_records(records: list[dict[str, Any]], *, field: str, contains: str) -> list[dict[str, Any]]:
    if not field or not contains:
        return records
    needles = [item.strip().lower() for item in contains.split(",") if item.strip()]
    if not needles:
        return records
    out: list[dict[str, Any]] = []
    for record in records:
        haystack = _as_text(record.get(field)).lower()
        if any(needle in haystack for needle in needles):
            out.append(record)
    return out


def _hash_text(*parts: str) -> str:
    digest = hashlib.sha1()
    for part in parts:
        digest.update(part.encode("utf-8", errors="replace"))
        digest.update(b"\x00")
    return digest.hexdigest()[:16]


def _build_prefix(
    prompt: str,
    initial_reason_steps: str,
    *,
    prefix_separator: str,
    continuation_separator: str,
) -> str:
    prompt = prompt.strip()
    initial_reason_steps = initial_reason_steps.strip()
    if initial_reason_steps:
        prefix = prefix_separator.join((prompt, initial_reason_steps))
    else:
        prefix = prompt
    if continuation_separator and not prefix.endswith(continuation_separator):
        prefix = prefix.rstrip() + continuation_separator
    return prefix


def _strip_boxed(value: str) -> str:
    match = re.search(r"\\boxed\{([^{}]+)\}", value)
    if match:
        return match.group(1)
    return value


def _last_number(value: str) -> str:
    matches = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+(?:\.\d+)?)?", value)
    return matches[-1] if matches else ""


def _extract_final_answer(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    candidates: list[str] = []
    for pattern in (
        r"####\s*([^\n]+)",
        r"(?:the\s+)?answer\s+is\s*:?\s*([^\n]+)",
        r"final\s+answer\s*:?\s*([^\n]+)",
    ):
        candidates.extend(match.group(1) for match in re.finditer(pattern, text, flags=re.IGNORECASE))
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    candidates.extend(boxed)
    if candidates:
        value = candidates[-1].strip()
        value = re.split(r"(?:\.|\n)", value, maxsplit=1)[0].strip()
        return value
    return _last_number(text)


def _normalize_answer(value: str) -> str:
    value = _strip_boxed(value)
    value = value.strip().lower()
    value = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", value)
    value = value.replace("$", "")
    value = value.replace(",", "")
    value = value.replace("\\", "")
    value = re.sub(r"^(?:the\s+answer\s+is\s*:?\s*)", "", value)
    value = re.sub(r"[^a-z0-9./+-]+", "", value)
    return value


def _answers_match(observed: str, gold: str) -> bool:
    observed_norm = _normalize_answer(observed)
    gold_norm = _normalize_answer(gold)
    if not observed_norm or not gold_norm:
        return False
    return observed_norm == gold_norm


def _prepare_one(
    record: dict[str, Any],
    *,
    raw_index: int,
    source_row_index: int,
    args: argparse.Namespace,
) -> PreparedPair:
    prompt_field, prompt = _pick_field(record, args.prompt_field, PROMPT_CANDIDATES)
    initial_field, initial = _pick_field(record, args.initial_field, INITIAL_CANDIDATES)
    step_chosen_field, step_chosen = _pick_field(record, args.chosen_field, CHOSEN_CANDIDATES)
    step_rejected_field, step_rejected = _pick_field(record, args.rejected_field, REJECTED_CANDIDATES)
    full_chosen_field, full_chosen = _pick_field(record, args.full_chosen_field, FULL_CHOSEN_CANDIDATES)
    full_rejected_field, full_rejected = _pick_field(
        record,
        args.full_rejected_field,
        FULL_REJECTED_CANDIDATES,
    )
    _answer_field, answer = _pick_field(record, args.answer_field, ANSWER_CANDIDATES)
    _id_field, explicit_problem_id = _pick_field(record, args.id_field, ID_CANDIDATES)

    prompt = prompt.strip()
    initial = initial.strip()
    step_chosen = step_chosen.strip()
    step_rejected = step_rejected.strip()
    full_chosen = full_chosen.strip()
    full_rejected = full_rejected.strip()
    answer = answer.strip()
    if args.pair_mode == "full":
        chosen_field = full_chosen_field or step_chosen_field
        rejected_field = full_rejected_field or step_rejected_field
        chosen = full_chosen or step_chosen
        rejected = full_rejected or step_rejected
        initial_for_prefix = initial if args.include_initial_in_full_prompt else ""
    else:
        chosen_field = step_chosen_field
        rejected_field = step_rejected_field
        chosen = step_chosen
        rejected = step_rejected
        initial_for_prefix = initial
    prefix = _build_prefix(
        prompt,
        initial_for_prefix,
        prefix_separator=_decode_escapes(args.prefix_separator),
        continuation_separator=_decode_escapes(args.continuation_separator),
    )
    problem_id = explicit_problem_id.strip() if explicit_problem_id.strip() else f"problem_{_hash_text(prompt)}"
    pair_id = f"pair_{_hash_text(prefix, chosen, rejected)}"

    reasons: list[str] = []
    if not prompt:
        reasons.append("missing_prompt")
    if not chosen:
        reasons.append("missing_chosen")
    if not rejected:
        reasons.append("missing_rejected")
    if chosen and rejected and chosen == rejected:
        reasons.append("identical_endpoints")
    if not chosen_field:
        reasons.append("missing_chosen_field")
    if not rejected_field:
        reasons.append("missing_rejected_field")
    chosen_answer = _extract_final_answer(chosen)
    rejected_answer = _extract_final_answer(rejected)
    if args.admission_policy == "final_answer_strict":
        if not answer:
            reasons.append("missing_answer")
        elif not _answers_match(chosen_answer, answer):
            reasons.append("chosen_answer_mismatch")
        if answer and _answers_match(rejected_answer, answer):
            reasons.append("rejected_answer_matches")
    admitted = not reasons

    return PreparedPair(
        raw_index=raw_index,
        source_row_index=source_row_index,
        admitted=admitted,
        admit_reason="ok" if admitted else ";".join(reasons),
        split="unassigned",
        sample_id="",
        pair_id=pair_id,
        problem_id=problem_id,
        prompt=prompt,
        initial_reason_steps=initial,
        prefix=prefix,
        chosen=chosen,
        rejected=rejected,
        full_chosen=full_chosen,
        full_rejected=full_rejected,
        answer=answer,
        pair_mode=args.pair_mode,
        chosen_answer=chosen_answer,
        rejected_answer=rejected_answer,
        prompt_field=prompt_field,
        initial_field=initial_field,
        chosen_field=chosen_field,
        rejected_field=rejected_field,
    )


def _with_split(pair: PreparedPair, *, split: str, sample_id: str) -> PreparedPair:
    return PreparedPair(
        raw_index=pair.raw_index,
        source_row_index=pair.source_row_index,
        admitted=pair.admitted,
        admit_reason=pair.admit_reason,
        split=split,
        sample_id=sample_id,
        pair_id=pair.pair_id,
        problem_id=pair.problem_id,
        prompt=pair.prompt,
        initial_reason_steps=pair.initial_reason_steps,
        prefix=pair.prefix,
        chosen=pair.chosen,
        rejected=pair.rejected,
        full_chosen=pair.full_chosen,
        full_rejected=pair.full_rejected,
        answer=pair.answer,
        pair_mode=pair.pair_mode,
        chosen_answer=pair.chosen_answer,
        rejected_answer=pair.rejected_answer,
        prompt_field=pair.prompt_field,
        initial_field=pair.initial_field,
        chosen_field=pair.chosen_field,
        rejected_field=pair.rejected_field,
    )


def _assign_row_splits(pairs: list[PreparedPair], *, args: argparse.Namespace) -> list[PreparedPair]:
    ordered = list(pairs)
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(ordered)
    limits = (("train", args.train_rows), ("val", args.val_rows), ("eval", args.eval_rows))
    out: list[PreparedPair] = []
    cursor = 0
    counters = {"train": 0, "val": 0, "eval": 0}
    for split, limit in limits:
        if limit <= 0:
            continue
        for pair in ordered[cursor : cursor + limit]:
            counters[split] += 1
            out.append(_with_split(pair, split=split, sample_id=f"{args.event}_{split}_{counters[split]:06d}"))
        cursor += max(limit, 0)
    return out


def _assign_problem_splits(pairs: list[PreparedPair], *, args: argparse.Namespace) -> list[PreparedPair]:
    groups: dict[str, list[PreparedPair]] = {}
    for pair in pairs:
        groups.setdefault(pair.problem_id, []).append(pair)
    group_items = list(groups.items())
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(group_items)
    limits = {"train": args.train_rows, "val": args.val_rows, "eval": args.eval_rows}
    counts = {"train": 0, "val": 0, "eval": 0}
    split_order = ["train", "val", "eval"]
    split_idx = 0
    out: list[PreparedPair] = []
    for _problem_id, group in group_items:
        while split_idx < len(split_order):
            split = split_order[split_idx]
            limit = limits[split]
            if limit <= 0 or counts[split] + len(group) > limit:
                split_idx += 1
                continue
            for pair in group:
                counts[split] += 1
                out.append(_with_split(pair, split=split, sample_id=f"{args.event}_{split}_{counts[split]:06d}"))
            break
        if split_idx >= len(split_order):
            break
    return out


def _pair_to_row(pair: PreparedPair, *, args: argparse.Namespace, source_dataset: str, source_split: str) -> dict[str, object]:
    return {
        "sample_id": pair.sample_id or f"{args.event}_rejected_{pair.raw_index:06d}",
        "split": pair.split if pair.admitted else "rejected",
        "event": args.event,
        "admitted": 1 if pair.admitted else 0,
        "admit_reason": pair.admit_reason,
        "prompt": pair.prefix,
        "y_plus": pair.chosen,
        "y_minus": pair.rejected,
        "y_plus_continuation": pair.chosen,
        "y_minus_continuation": pair.rejected,
        "y_plus_continuations_json": json.dumps([pair.chosen], ensure_ascii=False) if pair.chosen else "",
        "y_minus_continuations_json": json.dumps([pair.rejected], ensure_ascii=False) if pair.rejected else "",
        "source_dataset": source_dataset,
        "source_split": source_split,
        "raw_index": pair.raw_index,
        "source_row_index": pair.source_row_index,
        "problem_id": pair.problem_id,
        "pair_id": pair.pair_id,
        "base_problem": pair.prompt,
        "initial_reason_steps": pair.initial_reason_steps,
        "chosen": pair.chosen,
        "rejected": pair.rejected,
        "full_chosen": pair.full_chosen,
        "full_rejected": pair.full_rejected,
        "answer": pair.answer,
        "pair_mode": pair.pair_mode,
        "chosen_answer": pair.chosen_answer,
        "rejected_answer": pair.rejected_answer,
        "prompt_field": pair.prompt_field,
        "initial_field": pair.initial_field,
        "chosen_field": pair.chosen_field,
        "rejected_field": pair.rejected_field,
        "competitive_margin": "C(M;x)=S_M(y_plus|x)-S_M(y_minus|x)",
        "state_transition_driver": "logic_competition_advantage",
        "causal_object": (
            "component support for the preferred full solution over the rejected solution"
            if pair.pair_mode == "full"
            else "component support for the chosen reasoning step over the rejected step"
        ),
    }


def _summary_rows(pairs: list[PreparedPair]) -> list[dict[str, object]]:
    counts: dict[tuple[str, str], int] = {}
    for pair in pairs:
        split = pair.split if pair.admitted else "rejected"
        key = (split, pair.admit_reason)
        counts[key] = counts.get(key, 0) + 1
    return [
        {"split": split, "admit_reason": reason, "n": n}
        for (split, reason), n in sorted(counts.items(), key=lambda item: (item[0][0], item[0][1]))
    ]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.train_rows < 0 or args.val_rows < 0 or args.eval_rows < 0:
        raise ValueError("row counts must be non-negative")
    if args.input:
        all_records = _load_local_records(args.input)
        source_dataset = args.source_dataset or str(args.input)
        source_split = "local"
    else:
        all_records = _load_hf_records(args.hf_dataset, config=args.hf_config, split=args.hf_split)
        source_dataset = args.source_dataset or args.hf_dataset
        source_split = args.hf_split

    filtered_records = _filter_records(
        all_records,
        field=args.source_filter_field,
        contains=args.source_filter_contains,
    )
    start = max(args.start, 0)
    records = filtered_records[start:]
    if args.max_source_rows > 0:
        records = records[: args.max_source_rows]

    prepared_all = [
        _prepare_one(record, raw_index=idx, source_row_index=start + idx, args=args)
        for idx, record in enumerate(records)
    ]
    admitted = [pair for pair in prepared_all if pair.admitted]
    assigned = (
        _assign_problem_splits(admitted, args=args)
        if args.split_unit == "problem"
        else _assign_row_splits(admitted, args=args)
    )
    rejected = [pair for pair in prepared_all if not pair.admitted]
    output_pairs: list[PreparedPair] = list(assigned)
    if args.keep_rejected_rows:
        output_pairs.extend(rejected)

    rows = [
        _pair_to_row(pair, args=args, source_dataset=source_dataset, source_split=source_split)
        for pair in output_pairs
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dump_csv(args.out_dir / "pairs.csv", rows)
    dump_jsonl(args.out_dir / "pairs.jsonl", rows)
    dump_csv(args.out_dir / "pair_admission_summary.csv", _summary_rows(output_pairs + ([] if args.keep_rejected_rows else rejected)))

    split_counts: dict[str, int] = {"train": 0, "val": 0, "eval": 0, "rejected": len(rejected)}
    for pair in assigned:
        split_counts[pair.split] = split_counts.get(pair.split, 0) + 1
    dump_json(
        args.out_dir / "objective_spec.json",
        {
            "objective_id": args.event,
            "source_dataset": source_dataset,
            "source_split": source_split,
            "pair_mode": args.pair_mode,
            "admission_policy": args.admission_policy,
            "x": "prompt" if args.pair_mode == "full" and not args.include_initial_in_full_prompt else "prompt + initial_reason_steps",
            "y_plus": "preferred full solution" if args.pair_mode == "full" else "chosen reasoning step",
            "y_minus": "rejected full solution" if args.pair_mode == "full" else "rejected reasoning step",
            "score_mode": "avglogp",
            "competitive_margin": "C(M;x)=S_M(y_plus|x)-S_M(y_minus|x)",
            "logic_competition_advantage": (
                "The chosen step is treated as the locally better reasoning continuation; "
                "training drives component-local gain in this advantage rather than explicit state labels."
            ),
            "component_discovery": "Delta(c,x)=C(M_full;x)-C(M_zero_c;x)",
            "actuator_training": "gain(U,x)=C(M_U;x)-C(M_full;x)",
        },
    )
    dump_json(
        args.out_dir / "pair_build_manifest.json",
        {
            "source_dataset": source_dataset,
            "source_split": source_split,
            "source_rows_total": len(all_records),
            "source_rows_after_filter": len(filtered_records),
            "source_filter_field": args.source_filter_field,
            "source_filter_contains": args.source_filter_contains,
            "source_start": start,
            "source_rows_seen": len(records),
            "admitted_rows_seen": len(admitted),
            "rejected_rows_seen": len(rejected),
            "pair_mode": args.pair_mode,
            "admission_policy": args.admission_policy,
            "include_initial_in_full_prompt": bool(args.include_initial_in_full_prompt),
            "split_unit": args.split_unit,
            "shuffle": bool(args.shuffle),
            "seed": args.seed,
            "target_counts": {
                "train": args.train_rows,
                "val": args.val_rows,
                "eval": args.eval_rows,
            },
            "actual_counts": split_counts,
            "event": args.event,
            "prompt_field": args.prompt_field or "auto",
            "initial_field": args.initial_field or "auto",
            "chosen_field": args.chosen_field or "auto",
            "rejected_field": args.rejected_field or "auto",
            "prefix_separator": args.prefix_separator,
            "continuation_separator": args.continuation_separator,
            "semantics": (
                "This adapter only constructs admitted preference pairs. "
                "Component discovery and actuator training stay in the generic CECM pair interface."
            ),
        },
    )
    print(
        (
            f"[step-dpo-pairs] out={args.out_dir} admitted_seen={len(admitted)} "
            f"train={split_counts.get('train', 0)} val={split_counts.get('val', 0)} "
            f"eval={split_counts.get('eval', 0)} rejected={len(rejected)}"
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
