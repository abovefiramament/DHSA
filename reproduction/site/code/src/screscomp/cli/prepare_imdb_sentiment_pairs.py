from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from screscomp.data import dump_csv, dump_json, dump_jsonl


@dataclass(frozen=True, slots=True)
class GenerationCandidate:
    row_index: int
    sample_id: str
    prompt_id: str
    split: str
    prompt: str
    completion_id: str
    completion: str
    score: float
    source_row_index: str
    raw: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RejectedGroup:
    group_key: str
    split: str
    reason: str
    n_rows: int
    n_valid: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert scored IMDb sentiment-control generations into generic CECM actuator pairs."
    )
    p.add_argument("--input", type=Path, required=True, help="Scored generation .jsonl, .json, or .csv.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--event", type=str, default="imdb_positive_sentiment")
    p.add_argument("--group-field", type=str, default="sample_id")
    p.add_argument("--prompt-id-field", type=str, default="prompt_id")
    p.add_argument("--prompt-field", type=str, default="prompt")
    p.add_argument("--completion-field", type=str, default="completion")
    p.add_argument("--completion-id-field", type=str, default="completion_id")
    p.add_argument("--score-field", type=str, default="positive_sentiment_score")
    p.add_argument("--split-field", type=str, default="split")
    p.add_argument("--source-row-index-field", type=str, default="source_row_index")
    p.add_argument("--min-score-margin", type=float, default=0.0)
    p.add_argument(
        "--pair-selection-mode",
        type=str,
        default="all_pairs",
        choices=["all_pairs", "top_bottom"],
        help="all_pairs keeps every score-separated pair; top_bottom keeps one chosen/rejected pair per prompt.",
    )
    p.add_argument(
        "--max-pairs-per-prompt",
        type=int,
        default=0,
        help="0 keeps all score-separated pairs. Four completions yield six pairs.",
    )
    p.add_argument("--max-prompts-per-split", type=int, default=0)
    p.add_argument("--keep-rejected-rows", action="store_true")
    p.add_argument("--existing-pairs-csv", type=Path, default=None,
                    help="If set, read existing pairs from this CSV and only add new ones (no duplicate prompt_sample_id).")
    p.add_argument("--target-train-pairs", type=int, default=0,
                    help="If >0, stop adding train pairs once this count is reached (existing + new).")
    p.add_argument("--target-val-pairs", type=int, default=0,
                    help="If >0, stop adding val pairs once this count is reached (existing + new).")
    return p.parse_args(argv)


def _load_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line:
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"JSONL rows must be objects: {path}")
                    rows.append(value)
        return rows
    if suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(raw, list):
            if not all(isinstance(item, dict) for item in raw):
                raise ValueError(f"JSON list must contain objects: {path}")
            return list(raw)
        raise ValueError(f"Expected JSON list in {path}")
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"Unsupported input format {path.suffix!r}; expected .jsonl, .json, or .csv")


def _text(row: dict[str, Any], key: str) -> str:
    value = row.get(key, "")
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _float_or_none(value: Any) -> float | None:
    try:
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except Exception:
        return None


def _candidate(row: dict[str, Any], *, row_index: int, args: argparse.Namespace) -> GenerationCandidate | None:
    prompt = _text(row, args.prompt_field)
    completion = _text(row, args.completion_field)
    score = _float_or_none(row.get(args.score_field))
    group_key = _text(row, args.group_field)
    if not group_key:
        group_key = _text(row, args.prompt_id_field)
    if not group_key or not prompt or not completion or score is None:
        return None
    return GenerationCandidate(
        row_index=row_index,
        sample_id=group_key,
        prompt_id=_text(row, args.prompt_id_field),
        split=_text(row, args.split_field) or "train",
        prompt=prompt,
        completion_id=_text(row, args.completion_id_field) or str(row_index),
        completion=completion,
        score=score,
        source_row_index=_text(row, args.source_row_index_field),
        raw=row,
    )


def _limited_groups(
    groups: dict[str, list[GenerationCandidate]],
    *,
    max_prompts_per_split: int,
) -> dict[str, list[GenerationCandidate]]:
    if max_prompts_per_split <= 0:
        return groups
    out: dict[str, list[GenerationCandidate]] = {}
    counts: dict[str, int] = {}
    for key in sorted(groups.keys(), key=lambda item: min(candidate.row_index for candidate in groups[item])):
        split = groups[key][0].split
        if counts.get(split, 0) >= max_prompts_per_split:
            continue
        counts[split] = counts.get(split, 0) + 1
        out[key] = groups[key]
    return out


def _pair_rows_for_group(
    group_key: str,
    candidates: list[GenerationCandidate],
    *,
    args: argparse.Namespace,
    pair_offset: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    sorted_candidates = sorted(candidates, key=lambda item: (item.score, -item.row_index), reverse=True)
    rows: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    if args.pair_selection_mode == "top_bottom":
        plus = sorted_candidates[0]
        minus = sorted_candidates[-1]
        margin = plus.score - minus.score
        if margin <= args.min_score_margin:
            rejected.append(
                {
                    "group_key": group_key,
                    "split": plus.split,
                    "admit_reason": "score_tie_or_below_margin",
                    "y_plus_completion_id": plus.completion_id,
                    "y_minus_completion_id": minus.completion_id,
                    "score_margin": margin,
                }
            )
            return rows, rejected
        source_row_index = plus.source_row_index or minus.source_row_index
        rows.append(
            {
                "sample_id": f"{args.event}_{plus.split}_{pair_offset + 1:08d}",
                "split": plus.split,
                "event": args.event,
                "admitted": 1,
                "admit_reason": "ok",
                "prompt": plus.prompt,
                "y_plus": plus.completion,
                "y_minus": minus.completion,
                "y_plus_continuation": plus.completion,
                "y_minus_continuation": minus.completion,
                "y_plus_continuations_json": json.dumps([plus.completion], ensure_ascii=False),
                "y_minus_continuations_json": json.dumps([minus.completion], ensure_ascii=False),
                "pair_mode": "imdb_sentiment_scored_generation",
                "y_plus_reward": plus.score,
                "y_minus_reward": minus.score,
                "reward_margin": margin,
                "score_field": args.score_field,
                "prompt_sample_id": group_key,
                "prompt_id": plus.prompt_id or minus.prompt_id,
                "y_plus_completion_id": plus.completion_id,
                "y_minus_completion_id": minus.completion_id,
                "source_row_index": source_row_index,
                "row_index": source_row_index,
                "y_plus_generation_row_index": plus.row_index,
                "y_minus_generation_row_index": minus.row_index,
            }
        )
        return rows, rejected
    local_pair_idx = 0
    for high_idx in range(len(sorted_candidates)):
        for low_idx in range(high_idx + 1, len(sorted_candidates)):
            plus = sorted_candidates[high_idx]
            minus = sorted_candidates[low_idx]
            margin = plus.score - minus.score
            if margin <= args.min_score_margin:
                rejected.append(
                    {
                        "group_key": group_key,
                        "split": plus.split,
                        "admit_reason": "score_tie_or_below_margin",
                        "y_plus_completion_id": plus.completion_id,
                        "y_minus_completion_id": minus.completion_id,
                        "score_margin": margin,
                    }
                )
                continue
            local_pair_idx += 1
            if args.max_pairs_per_prompt > 0 and local_pair_idx > args.max_pairs_per_prompt:
                continue
            pair_idx = pair_offset + len(rows) + 1
            source_row_index = plus.source_row_index or minus.source_row_index
            rows.append(
                {
                    "sample_id": f"{args.event}_{plus.split}_{pair_idx:08d}",
                    "split": plus.split,
                    "event": args.event,
                    "admitted": 1,
                    "admit_reason": "ok",
                    "prompt": plus.prompt,
                    "y_plus": plus.completion,
                    "y_minus": minus.completion,
                    "y_plus_continuation": plus.completion,
                    "y_minus_continuation": minus.completion,
                    "y_plus_continuations_json": json.dumps([plus.completion], ensure_ascii=False),
                    "y_minus_continuations_json": json.dumps([minus.completion], ensure_ascii=False),
                    "pair_mode": "imdb_sentiment_scored_generation",
                    "y_plus_reward": plus.score,
                    "y_minus_reward": minus.score,
                    "reward_margin": margin,
                    "score_field": args.score_field,
                    "prompt_sample_id": group_key,
                    "prompt_id": plus.prompt_id or minus.prompt_id,
                    "y_plus_completion_id": plus.completion_id,
                    "y_minus_completion_id": minus.completion_id,
                    "source_row_index": source_row_index,
                    "row_index": source_row_index,
                    "y_plus_generation_row_index": plus.row_index,
                    "y_minus_generation_row_index": minus.row_index,
                }
            )
    return rows, rejected


def _summary_rows(pairs: list[dict[str, object]], rejected: list[RejectedGroup], rejected_pairs: list[dict[str, object]]) -> list[dict[str, object]]:
    counts: dict[tuple[str, str], int] = {}
    for row in pairs:
        key = (str(row.get("split", "")), "ok")
        counts[key] = counts.get(key, 0) + 1
    for group in rejected:
        key = (group.split or "unknown", group.reason)
        counts[key] = counts.get(key, 0) + 1
    for row in rejected_pairs:
        key = (str(row.get("split", "")), str(row.get("admit_reason", "")))
        counts[key] = counts.get(key, 0) + 1
    return [
        {"split": split, "admit_reason": reason, "n": n}
        for (split, reason), n in sorted(counts.items(), key=lambda item: (item[0][0], item[0][1]))
    ]


def build_imdb_sentiment_pairs_from_args(args: argparse.Namespace) -> dict[str, object]:
    # Load existing pairs for incremental mode
    existing_prompt_sample_ids: set[str] = set()
    existing_pairs: list[dict[str, object]] = []
    existing_train_count = 0
    existing_val_count = 0
    if args.existing_pairs_csv is not None and args.existing_pairs_csv.exists():
        with args.existing_pairs_csv.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                existing_pairs.append(row)
                psid = str(row.get("prompt_sample_id", ""))
                if psid:
                    existing_prompt_sample_ids.add(psid)
                split = str(row.get("split", "train"))
                if split == "train":
                    existing_train_count += 1
                else:
                    existing_val_count += 1
        print(f"[imdb-sentiment-pairs] Loaded {len(existing_pairs)} existing pairs "
              f"(train={existing_train_count}, val={existing_val_count})", flush=True)

    records = _load_records(args.input)
    groups_raw: dict[str, list[dict[str, Any]]] = {}
    group_order: list[str] = []
    groups: dict[str, list[GenerationCandidate]] = {}
    for row_index, row in enumerate(records):
        group_key = _text(row, args.group_field) or _text(row, args.prompt_id_field)
        if group_key:
            if group_key not in groups_raw:
                group_order.append(group_key)
            groups_raw.setdefault(group_key, []).append(row)
        candidate = _candidate(row, row_index=row_index, args=args)
        if candidate is not None:
            groups.setdefault(candidate.sample_id, []).append(candidate)
    groups = _limited_groups(groups, max_prompts_per_split=args.max_prompts_per_split)

    pair_rows: list[dict[str, object]] = list(existing_pairs)
    rejected_groups: list[RejectedGroup] = []
    rejected_pair_rows: list[dict[str, object]] = []
    new_train_count = 0
    new_val_count = 0
    for group_key in group_order:
        if group_key not in groups:
            raw_rows = groups_raw[group_key]
            split = _text(raw_rows[0], args.split_field) or "train"
            rejected_groups.append(
                RejectedGroup(
                    group_key=group_key,
                    split=split,
                    reason="no_valid_scored_completions",
                    n_rows=len(raw_rows),
                    n_valid=0,
                )
            )
            continue
        candidates = groups[group_key]
        if len(candidates) < 2:
            rejected_groups.append(
                RejectedGroup(
                    group_key=group_key,
                    split=candidates[0].split if candidates else "train",
                    reason="fewer_than_two_valid_completions",
                    n_rows=len(groups_raw.get(group_key, [])),
                    n_valid=len(candidates),
                )
            )
            continue
        rows, rejected_pairs = _pair_rows_for_group(group_key, candidates, args=args, pair_offset=len(pair_rows))
        if not rows:
            rejected_groups.append(
                RejectedGroup(
                    group_key=group_key,
                    split=candidates[0].split,
                    reason="no_score_separated_pairs",
                    n_rows=len(groups_raw.get(group_key, [])),
                    n_valid=len(candidates),
                )
            )
        # Skip prompt ids that are already present in an existing pairs file when
        # running incrementally. New rows from the current build should still be
        # allowed to contribute multiple score-separated pairs for the same prompt.
        for row in rows:
            psid = str(row.get("prompt_sample_id", ""))
            if psid and psid in existing_prompt_sample_ids:
                continue
            split = str(row.get("split", "train"))
            # Check target limits
            if split == "train" and args.target_train_pairs > 0:
                if existing_train_count + new_train_count >= args.target_train_pairs:
                    continue
            if split != "train" and args.target_val_pairs > 0:
                if existing_val_count + new_val_count >= args.target_val_pairs:
                    continue
            pair_rows.append(row)
            if split == "train":
                new_train_count += 1
            else:
                new_val_count += 1
        rejected_pair_rows.extend(rejected_pairs)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dump_csv(args.out_dir / "pairs.csv", pair_rows)
    dump_jsonl(args.out_dir / "pairs.jsonl", pair_rows)
    dump_csv(
        args.out_dir / "rejected_groups.csv",
        [
            {
                "group_key": item.group_key,
                "split": item.split,
                "admit_reason": item.reason,
                "n_rows": item.n_rows,
                "n_valid": item.n_valid,
            }
            for item in rejected_groups
        ],
    )
    if args.keep_rejected_rows:
        dump_csv(args.out_dir / "rejected_pairs.csv", rejected_pair_rows)
        dump_jsonl(args.out_dir / "rejected_pairs.jsonl", rejected_pair_rows)
    dump_csv(args.out_dir / "pair_admission_summary.csv", _summary_rows(pair_rows, rejected_groups, rejected_pair_rows))
    split_counts: dict[str, int] = {}
    for row in pair_rows:
        split = str(row.get("split", "train"))
        split_counts[split] = split_counts.get(split, 0) + 1
    dump_json(
        args.out_dir / "objective_spec.json",
        {
            "objective_id": args.event,
            "task_family": "controlled_sentiment",
            "environment_stage": "scored_generation_pairs",
            "x": "IMDb review prefix",
            "y_plus": "higher positive-sentiment completion",
            "y_minus": "lower positive-sentiment completion",
            "reward_score_field": args.score_field,
            "preference_rule": "higher score is preferred",
            "actuator_pair_interface": "generic y_plus/y_minus CECM pairs",
            "recommended_training_loss": (
                "Use cecm_train_fixed_actuator with preference_loss_mode=dpo, dpo_beta=1, and "
                "score_mode=avglogp. The main IMDb objective should be length-normalized causal "
                "continuation conditional log-probability over the fixed y_plus/y_minus pair. "
                "score_mode=answer_rest_margin is a discrete-token diagnostic, not the primary controlled "
                "sentiment setting, because the task is rewarded by an automatic sentiment scorer rather "
                "than a fixed answer token."
            ),
        },
    )
    dump_json(
        args.out_dir / "pair_build_manifest.json",
        {
            "input": str(args.input),
            "event": args.event,
            "source_rows": len(records),
            "groups_seen": len(groups_raw),
            "groups_valid": len(groups),
            "admitted_pairs": len(pair_rows),
            "rejected_groups": len(rejected_groups),
            "rejected_pair_candidates": len(rejected_pair_rows),
            "group_field": args.group_field,
            "prompt_id_field": args.prompt_id_field,
            "prompt_field": args.prompt_field,
            "completion_field": args.completion_field,
            "score_field": args.score_field,
            "split_field": args.split_field,
            "min_score_margin": args.min_score_margin,
            "pair_selection_mode": args.pair_selection_mode,
            "max_pairs_per_prompt": args.max_pairs_per_prompt,
            "max_prompts_per_split": args.max_prompts_per_split,
            "actual_counts": split_counts,
            "baseline_status": "not_run",
            "semantics": (
                "For each prompt, generated completions are sorted by the sentiment reward. "
                "Every score-separated high/low completion pair becomes one generic actuator pair."
            ),
        },
    )
    result = {
        "pairs_csv": str(args.out_dir / "pairs.csv"),
        "pairs_jsonl": str(args.out_dir / "pairs.jsonl"),
        "admitted_pairs": len(pair_rows),
        "rejected_pairs": len(rejected_pair_rows),
        "rejected_groups": len(rejected_groups),
        "groups_seen": len(groups_raw),
        "groups_valid": len(groups),
    }
    print(
        (
            f"[imdb-sentiment-pairs] out={args.out_dir} groups={len(groups_raw)} "
            f"valid_groups={len(groups)} pairs={len(pair_rows)} rejected_groups={len(rejected_groups)}"
        ),
        flush=True,
    )
    return result


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    build_imdb_sentiment_pairs_from_args(args)


if __name__ == "__main__":
    main()
