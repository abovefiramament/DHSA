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


TEXT_CANDIDATES = ("text", "review", "content", "prompt")
LABEL_CANDIDATES = ("label", "sentiment")


@dataclass(frozen=True, slots=True)
class PreparedPrompt:
    raw_index: int
    source_row_index: int
    admitted: bool
    admit_reason: str
    split: str
    sample_id: str
    prompt_id: str
    prompt: str
    prefix: str
    prefix_token_count: int
    source_text: str
    source_label: str
    text_field: str
    label_field: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build the prompt-level IMDb positive-sentiment control environment. "
            "This prepares prefixes and scoring/generation specs only; it does not run baselines."
        )
    )
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, default=None, help="Local .jsonl, .json, or .csv source file.")
    source.add_argument("--hf-dataset", type=str, default="", help="Optional Hugging Face dataset id, e.g. imdb.")
    p.add_argument("--hf-config", type=str, default=None)
    p.add_argument("--hf-split", type=str, default="train")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--event", type=str, default="imdb_positive_sentiment")
    p.add_argument("--source-dataset", type=str, default="")
    p.add_argument("--text-field", type=str, default="")
    p.add_argument("--label-field", type=str, default="")
    p.add_argument("--start", type=int, default=0)
    p.add_argument(
        "--max-source-rows",
        type=int,
        default=25000,
        help="Rows considered after --start. Use 0 for all available rows.",
    )
    p.add_argument("--train-rows", type=int, default=25000)
    p.add_argument("--val-rows", type=int, default=0)
    p.add_argument("--eval-rows", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--prefix-token-min", type=int, default=2)
    p.add_argument("--prefix-token-max", type=int, default=8)
    p.add_argument(
        "--prefix-mode",
        choices=["whitespace", "tokenizer"],
        default="whitespace",
        help="whitespace is dependency-light; tokenizer uses --tokenizer for closer paper matching.",
    )
    p.add_argument("--tokenizer", type=str, default="", help="Tokenizer id/path when --prefix-mode=tokenizer.")
    p.add_argument(
        "--normalize-imdb-breaks",
        action="store_true",
        help="Replace common IMDb <br /> tags before prefix extraction. Off by default for raw-dataset fidelity.",
    )
    p.add_argument("--prompt-template", type=str, default="{prefix}")
    p.add_argument("--target-sentiment", type=str, default="positive")
    p.add_argument("--reward-score-field", type=str, default="positive_sentiment_score")
    p.add_argument("--completions-per-prefix", type=int, default=4)
    p.add_argument("--scorer-model", type=str, default="siebert/sentiment-roberta-large-english")
    p.add_argument("--scorer-positive-label", type=str, default="POSITIVE")
    p.add_argument("--reference-model", type=str, default="gpt2-large")
    p.add_argument(
        "--sft-policy-note",
        type=str,
        default="Exact DPO-style comparison uses a GPT-2-large policy SFT-trained on IMDb for one epoch.",
    )
    p.add_argument(
        "--keep-rejected-rows",
        action="store_true",
        help="Also write prompt rows rejected during environment construction for audit.",
    )
    return p.parse_args(argv)


def _load_local_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
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
        with path.open("r", encoding="utf-8-sig", newline="") as f:
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


def _hash_text(*parts: str) -> str:
    digest = hashlib.sha1()
    for part in parts:
        digest.update(part.encode("utf-8", errors="ignore"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def _normalize_text(text: str, *, imdb_breaks: bool) -> str:
    if imdb_breaks:
        text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    return text.strip()


def _load_tokenizer(tokenizer_id: str) -> Any:
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover - dependency import failure
        raise SystemExit("--prefix-mode=tokenizer requires transformers to be installed.") from exc
    return AutoTokenizer.from_pretrained(tokenizer_id)


def _build_prefix(
    text: str,
    *,
    args: argparse.Namespace,
    rng: random.Random,
    tokenizer: Any | None,
) -> tuple[bool, str, int, str]:
    if args.prefix_token_min <= 0:
        raise ValueError("--prefix-token-min must be positive")
    if args.prefix_token_max < args.prefix_token_min:
        raise ValueError("--prefix-token-max must be >= --prefix-token-min")

    target_len = rng.randint(args.prefix_token_min, args.prefix_token_max)
    if args.prefix_mode == "tokenizer":
        if tokenizer is None:
            raise ValueError("--prefix-mode=tokenizer requires --tokenizer")
        tokenized = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=target_len,
            verbose=False,
        )
        token_ids = tokenized["input_ids"]
        if len(token_ids) < args.prefix_token_min:
            return False, "", 0, "too_short"
        prefix = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        if not prefix:
            return False, "", 0, "empty_prefix"
        return True, prefix, len(token_ids), "ok"

    tokens = text.split()
    if len(tokens) < args.prefix_token_min:
        return False, "", 0, "too_short"
    take = min(target_len, len(tokens))
    prefix = " ".join(tokens[:take]).strip()
    if not prefix:
        return False, "", 0, "empty_prefix"
    return True, prefix, take, "ok"


def _prepare_prompt(
    record: dict[str, Any],
    *,
    raw_index: int,
    source_row_index: int,
    args: argparse.Namespace,
    rng: random.Random,
    tokenizer: Any | None,
) -> PreparedPrompt:
    text_field, source_text = _pick_field(record, args.text_field, TEXT_CANDIDATES)
    label_field, source_label = _pick_field(record, args.label_field, LABEL_CANDIDATES)
    source_text = _normalize_text(source_text, imdb_breaks=bool(args.normalize_imdb_breaks))

    if not text_field:
        return PreparedPrompt(
            raw_index=raw_index,
            source_row_index=source_row_index,
            admitted=False,
            admit_reason="missing_text_field",
            split="unassigned",
            sample_id="",
            prompt_id=f"{args.event}_rejected_{raw_index:06d}",
            prompt="",
            prefix="",
            prefix_token_count=0,
            source_text="",
            source_label=source_label,
            text_field=text_field,
            label_field=label_field,
        )

    admitted, prefix, token_count, reason = _build_prefix(source_text, args=args, rng=rng, tokenizer=tokenizer)
    prompt = args.prompt_template.format(prefix=prefix) if admitted else ""
    prompt_id = _hash_text(args.event, str(source_row_index), source_text, prefix)
    return PreparedPrompt(
        raw_index=raw_index,
        source_row_index=source_row_index,
        admitted=admitted,
        admit_reason=reason,
        split="unassigned",
        sample_id="",
        prompt_id=prompt_id,
        prompt=prompt,
        prefix=prefix,
        prefix_token_count=token_count,
        source_text=source_text,
        source_label=source_label,
        text_field=text_field,
        label_field=label_field,
    )


def _with_split(prompt: PreparedPrompt, *, split: str, sample_id: str) -> PreparedPrompt:
    return PreparedPrompt(
        raw_index=prompt.raw_index,
        source_row_index=prompt.source_row_index,
        admitted=prompt.admitted,
        admit_reason=prompt.admit_reason,
        split=split,
        sample_id=sample_id,
        prompt_id=prompt.prompt_id,
        prompt=prompt.prompt,
        prefix=prompt.prefix,
        prefix_token_count=prompt.prefix_token_count,
        source_text=prompt.source_text,
        source_label=prompt.source_label,
        text_field=prompt.text_field,
        label_field=prompt.label_field,
    )


def _assign_splits(prompts: list[PreparedPrompt], *, args: argparse.Namespace) -> list[PreparedPrompt]:
    ordered = list(prompts)
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(ordered)
    limits = [("train", args.train_rows), ("val", args.val_rows), ("eval", args.eval_rows)]
    out: list[PreparedPrompt] = []
    cursor = 0
    counters = {"train": 0, "val": 0, "eval": 0}
    for split, limit in limits:
        if limit <= 0:
            continue
        for prompt in ordered[cursor : cursor + limit]:
            counters[split] += 1
            out.append(_with_split(prompt, split=split, sample_id=f"{args.event}_{split}_{counters[split]:06d}"))
        cursor += max(limit, 0)
    return out


def _prompt_to_row(
    prompt: PreparedPrompt,
    *,
    args: argparse.Namespace,
    source_dataset: str,
    source_split: str,
) -> dict[str, object]:
    split = prompt.split if prompt.admitted else "rejected"
    sample_id = prompt.sample_id or f"{args.event}_rejected_{prompt.raw_index:06d}"
    return {
        "sample_id": sample_id,
        "split": split,
        "event": args.event,
        "admitted": 1 if prompt.admitted else 0,
        "admit_reason": prompt.admit_reason,
        "prompt": prompt.prompt,
        "prefix": prompt.prefix,
        "prefix_token_count": prompt.prefix_token_count,
        "prefix_mode": args.prefix_mode,
        "tokenizer": args.tokenizer,
        "completion_slots": args.completions_per_prefix,
        "target_sentiment": args.target_sentiment,
        "scorer_model": args.scorer_model,
        "scorer_positive_label": args.scorer_positive_label,
        "source_dataset": source_dataset,
        "source_split": source_split,
        "raw_index": prompt.raw_index,
        "source_row_index": prompt.source_row_index,
        "prompt_id": prompt.prompt_id,
        "source_label": prompt.source_label,
        "text_field": prompt.text_field,
        "label_field": prompt.label_field,
        "source_text": prompt.source_text,
    }


def _summary_rows(prompts: list[PreparedPrompt]) -> list[dict[str, object]]:
    counts: dict[tuple[str, str], int] = {}
    for prompt in prompts:
        split = prompt.split if prompt.admitted else "rejected"
        key = (split, prompt.admit_reason)
        counts[key] = counts.get(key, 0) + 1
    return [
        {"split": split, "admit_reason": reason, "n": n}
        for (split, reason), n in sorted(counts.items(), key=lambda item: (item[0][0], item[0][1]))
    ]


def _pair_count(completions_per_prefix: int) -> int:
    if completions_per_prefix < 2:
        return 0
    return completions_per_prefix * (completions_per_prefix - 1) // 2


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.prefix_mode == "tokenizer" and not args.tokenizer:
        raise SystemExit("--prefix-mode=tokenizer requires --tokenizer")
    if args.completions_per_prefix < 1:
        raise SystemExit("--completions-per-prefix must be positive")

    if args.input is not None:
        all_records = _load_local_records(args.input)
        source_dataset = args.source_dataset or str(args.input)
        source_split = "local"
    else:
        all_records = _load_hf_records(args.hf_dataset, config=args.hf_config, split=args.hf_split)
        source_dataset = args.source_dataset or args.hf_dataset
        source_split = args.hf_split

    source_slice = all_records[args.start :]
    if args.max_source_rows > 0:
        source_slice = source_slice[: args.max_source_rows]

    rng = random.Random(args.seed)
    tokenizer = _load_tokenizer(args.tokenizer) if args.prefix_mode == "tokenizer" else None
    prepared_all = [
        _prepare_prompt(
            record,
            raw_index=idx,
            source_row_index=args.start + idx,
            args=args,
            rng=rng,
            tokenizer=tokenizer,
        )
        for idx, record in enumerate(source_slice)
    ]

    admitted = [prompt for prompt in prepared_all if prompt.admitted]
    assigned = _assign_splits(admitted, args=args)
    rejected = [prompt for prompt in prepared_all if not prompt.admitted]
    output_prompts: list[PreparedPrompt] = list(assigned)
    if args.keep_rejected_rows:
        output_prompts.extend(rejected)

    rows = [
        _prompt_to_row(prompt, args=args, source_dataset=source_dataset, source_split=source_split)
        for prompt in output_prompts
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dump_csv(args.out_dir / "prompts.csv", rows)
    dump_jsonl(args.out_dir / "prompts.jsonl", rows)
    dump_csv(args.out_dir / "prompt_admission_summary.csv", _summary_rows(output_prompts + ([] if args.keep_rejected_rows else rejected)))

    split_counts: dict[str, int] = {"train": 0, "val": 0, "eval": 0, "rejected": len(rejected)}
    for prompt in assigned:
        split_counts[prompt.split] = split_counts.get(prompt.split, 0) + 1

    pairs_per_prefix = _pair_count(args.completions_per_prefix)
    reward_score_field = args.reward_score_field
    target_sentiment = args.target_sentiment.strip() or "positive"
    dump_json(
        args.out_dir / "objective_spec.json",
        {
            "objective_id": args.event,
            "source_dataset": source_dataset,
            "source_split": source_split,
            "task_family": "controlled_sentiment",
            "behavior": f"increase {target_sentiment} sentiment in model continuations from IMDb review prefixes",
            "environment_stage": "prompt_set_only",
            "x": "IMDb review prefix",
            "y": "model continuation",
            "reward": {
                "scorer_model": args.scorer_model,
                "positive_label": args.scorer_positive_label,
                "score_name": reward_score_field,
                "preference_rule": f"higher {reward_score_field} is preferred",
            },
            "dpo_style_setup": {
                "prefix_length": [args.prefix_token_min, args.prefix_token_max],
                "prefix_mode": args.prefix_mode,
                "tokenizer": args.tokenizer,
                "completions_per_prefix": args.completions_per_prefix,
                "pairs_per_prefix": pairs_per_prefix,
                "pair_rule": "score generated completions with the sentiment classifier and build all unordered preferred/rejected pairs by score",
                "reference_model": args.reference_model,
                "sft_policy_note": args.sft_policy_note,
            },
            "outputs": {
                "prompts_csv": "prompts.csv",
                "prompts_jsonl": "prompts.jsonl",
                "generation_plan": "generation_plan.json",
                "manifest": "environment_manifest.json",
            },
        },
    )
    dump_json(
        args.out_dir / "generation_plan.json",
        {
            "input_prompts": "prompts.jsonl",
            "completion_slots_per_prompt": args.completions_per_prefix,
            "expected_completion_fields": [
                "sample_id",
                "split",
                "completion_id",
                "prompt",
                "completion",
                "full_text",
                "positive_sentiment_score",
                "negative_sentiment_score",
                "scorer_model",
            ],
            "scoring": {
                "model": args.scorer_model,
                "positive_label": args.scorer_positive_label,
                "rank_by": reward_score_field,
            },
            "preference_pair_construction": {
                "pairs_per_prefix": pairs_per_prefix,
                "y_plus": "higher-scoring completion",
                "y_minus": "lower-scoring completion",
                "tie_policy": "drop exact ties or apply a small margin threshold in the scoring stage",
            },
            "baseline_status": "not_run",
        },
    )
    dump_json(
        args.out_dir / "environment_manifest.json",
        {
            "source_dataset": source_dataset,
            "source_split": source_split,
            "source_rows_total": len(all_records),
            "source_rows_after_slice": len(source_slice),
            "start": args.start,
            "max_source_rows": args.max_source_rows,
            "text_field": args.text_field or "auto",
            "label_field": args.label_field or "auto",
            "prefix_token_min": args.prefix_token_min,
            "prefix_token_max": args.prefix_token_max,
            "prefix_mode": args.prefix_mode,
            "tokenizer": args.tokenizer,
            "normalize_imdb_breaks": bool(args.normalize_imdb_breaks),
            "prompt_template": args.prompt_template,
            "shuffle": bool(args.shuffle),
            "seed": args.seed,
            "target_counts": {
                "train": args.train_rows,
                "val": args.val_rows,
                "eval": args.eval_rows,
            },
            "actual_counts": split_counts,
            "admitted_seen": len(admitted),
            "assigned_total": len(assigned),
            "unused_admitted": max(0, len(admitted) - len(assigned)),
            "event": args.event,
            "target_sentiment": target_sentiment,
            "scorer_model": args.scorer_model,
            "reward_score_field": reward_score_field,
            "reference_model": args.reference_model,
            "completions_per_prefix": args.completions_per_prefix,
            "pairs_per_prefix": pairs_per_prefix,
            "baseline_status": "not_run",
        },
    )
    print(
        (
            f"[imdb-sentiment-env] out={args.out_dir} admitted_seen={len(admitted)} "
            f"train={split_counts.get('train', 0)} val={split_counts.get('val', 0)} "
            f"eval={split_counts.get('eval', 0)} rejected={len(rejected)} "
            f"completion_slots={args.completions_per_prefix} pairs_per_prefix={pairs_per_prefix}"
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
