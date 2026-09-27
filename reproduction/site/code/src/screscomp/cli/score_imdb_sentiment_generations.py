from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, desc=None):
        return iterable

from screscomp.data import dump_csv, dump_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score IMDb sentiment-control generations with a sentiment classifier.")
    p.add_argument("--input", type=Path, required=True, help="Generation .jsonl, .json, or .csv.")
    p.add_argument("--out-jsonl", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, default=None)
    p.add_argument("--summary-csv", type=Path, default=None)
    p.add_argument("--scorer-model", type=str, default="siebert/sentiment-roberta-large-english")
    p.add_argument("--positive-label", type=str, default="POSITIVE")
    p.add_argument("--negative-label", type=str, default="NEGATIVE")
    p.add_argument(
        "--score-text",
        choices=["completion", "full_text", "prompt_plus_completion"],
        default="completion",
        help="DPO-style sentiment control should usually score the generated continuation.",
    )
    p.add_argument("--completion-field", type=str, default="completion")
    p.add_argument("--prompt-field", type=str, default="prompt")
    p.add_argument("--full-text-field", type=str, default="full_text")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--truncation", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--scorer-max-length", type=int, default=512)
    p.add_argument(
        "--device",
        type=int,
        default=-1,
        help="Transformers pipeline device id. Use -1 for CPU, 0 for first CUDA device.",
    )
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument(
        "--stream-summary-every-batches",
        type=int,
        default=0,
        help="If >0, overwrite summary_csv with the current partial score summary every N scorer batches.",
    )
    p.add_argument(
        "--progress-json",
        type=Path,
        default=None,
        help="Optional progress JSON that is updated alongside streaming summaries.",
    )
    p.add_argument("--overwrite", action="store_true", help="Delete existing outputs before scoring.")
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


def _score_text(row: dict[str, Any], *, args: argparse.Namespace) -> str:
    if args.score_text == "completion":
        return str(row.get(args.completion_field, ""))
    if args.score_text == "full_text":
        return str(row.get(args.full_text_field, ""))
    return f"{row.get(args.prompt_field, '')}{row.get(args.completion_field, '')}"


def _positive_score(result: Any, *, positive_label: str) -> tuple[str, float, float]:
    if isinstance(result, list):
        candidates = result
    else:
        candidates = [result]
    normalized = [
        {
            "label": str(item.get("label", "")),
            "score": float(item.get("score", 0.0)),
        }
        for item in candidates
        if isinstance(item, dict)
    ]
    if not normalized:
        return "", 0.0, 0.0
    positive_label_upper = positive_label.upper()
    for item in normalized:
        if item["label"].upper() == positive_label_upper:
            return item["label"], item["score"], item["score"]
    best = max(normalized, key=lambda item: item["score"])
    if len(normalized) == 1 and best["label"].upper() != positive_label_upper:
        return best["label"], best["score"], 1.0 - best["score"]
    return best["label"], best["score"], 0.0


def _label_score(result: Any, *, label: str) -> float | None:
    if isinstance(result, list):
        candidates = result
    else:
        candidates = [result]
    label_upper = label.upper()
    for item in candidates:
        if isinstance(item, dict) and str(item.get("label", "")).upper() == label_upper:
            return float(item.get("score", 0.0))
    return None


def _top_label_score(result: Any) -> tuple[str, float]:
    if isinstance(result, list):
        candidates = result
    else:
        candidates = [result]
    normalized = [
        {
            "label": str(item.get("label", "")),
            "score": float(item.get("score", 0.0)),
        }
        for item in candidates
        if isinstance(item, dict)
    ]
    if not normalized:
        return "", 0.0
    best = max(normalized, key=lambda item: item["score"])
    return best["label"], best["score"]


def _float_or_none(value: Any) -> float | None:
    try:
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except Exception:
        return None


def _summary_key(row: dict[str, object]) -> tuple[str, str, str]:
    control = str(row.get("control_name") or "generated")
    alpha_value = _float_or_none(row.get("alpha"))
    alpha = "" if alpha_value is None else f"{alpha_value:.8g}"
    split = str(row.get("split") or "unknown")
    return split, control, alpha


def _new_group_stats() -> dict[str, float]:
    return {
        "n": 0.0,
        "positive_sum": 0.0,
        "paper_logodds_reward_sum": 0.0,
        "positive_ge_0p5_sum": 0.0,
        "label_score_sum": 0.0,
        "completion_chars_sum": 0.0,
        "positive_min": math.inf,
        "positive_max": -math.inf,
    }


def _paper_logodds_reward(positive: float) -> float:
    eps = 1e-6
    p = min(max(float(positive), eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def _update_group_stats(stats: dict[str, float], row: dict[str, object]) -> None:
    positive = float(row.get("positive_sentiment_score", 0.0))
    paper_reward = float(row.get("paper_logodds_reward", _paper_logodds_reward(positive)))
    label_score = float(row.get("sentiment_label_score", 0.0))
    completion_chars = float(len(str(row.get("completion", ""))))
    stats["n"] += 1.0
    stats["positive_sum"] += positive
    stats["paper_logodds_reward_sum"] += paper_reward
    stats["positive_ge_0p5_sum"] += 1.0 if positive >= 0.5 else 0.0
    stats["label_score_sum"] += label_score
    stats["completion_chars_sum"] += completion_chars
    stats["positive_min"] = min(stats["positive_min"], positive)
    stats["positive_max"] = max(stats["positive_max"], positive)


def _summary_rows_from_stats(grouped: dict[tuple[str, str, str], dict[str, float]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for (split, control, alpha), stats in sorted(grouped.items(), key=lambda item: item[0]):
        n = int(stats["n"])
        if n <= 0:
            continue
        n_float = float(n)
        summary.append(
            {
                "split": split,
                "control_name": control,
                "alpha": alpha,
                "n": n,
                "mean_positive_sentiment_score": stats["positive_sum"] / n_float,
                "paper_mean_logodds_reward": stats["paper_logodds_reward_sum"] / n_float,
                "positive_rate_ge_0p5": stats["positive_ge_0p5_sum"] / n_float,
                "mean_sentiment_label_score": stats["label_score_sum"] / n_float,
                "mean_completion_chars": stats["completion_chars_sum"] / n_float,
                "min_positive_sentiment_score": stats["positive_min"],
                "max_positive_sentiment_score": stats["positive_max"],
            }
        )
    return summary


def _append_jsonl_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_csv_rows(path: Path, rows: list[dict[str, object]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _ordered_field_union(rows: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for row in rows:
        for key in row.keys():
            key_text = str(key)
            if key_text in seen:
                continue
            seen.add(key_text)
            ordered.append(key_text)
    return ordered


def _write_progress(
    *,
    summary_csv: Path,
    progress_json: Path | None,
    grouped: dict[tuple[str, str, str], dict[str, float]],
    rows_scored: int,
    rows_total: int,
    batches_done: int,
    batches_total: int,
) -> None:
    dump_csv(summary_csv, _summary_rows_from_stats(grouped))
    if progress_json is None:
        return
    dump_json(
        progress_json,
        {
            "rows_scored": rows_scored,
            "rows_total": rows_total,
            "rows_remaining": max(0, rows_total - rows_scored),
            "progress_fraction": (rows_scored / rows_total) if rows_total > 0 else 0.0,
            "batches_done": batches_done,
            "batches_total": batches_total,
            "summary_csv": str(summary_csv),
        },
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rows = _load_records(args.input)
    if args.max_rows is not None:
        rows = rows[: args.max_rows]
    if not rows:
        raise SystemExit(f"No rows found: {args.input}")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")

    try:
        from transformers import pipeline
    except Exception as exc:  # pragma: no cover - optional runtime path
        raise SystemExit("Scoring requires transformers. Install it or pass already scored generations to pair builder.") from exc

    summary_csv = args.summary_csv or (args.out_jsonl.parent / "score_summary.csv")
    progress_json = args.progress_json
    if args.overwrite:
        for path in [args.out_jsonl, args.out_csv, summary_csv, progress_json]:
            if path is not None and path.exists():
                path.unlink()

    print(f"[imdb-sentiment-score] loading scorer={args.scorer_model} rows={len(rows)}", flush=True)
    scorer = pipeline("sentiment-analysis", model=args.scorer_model, device=args.device, top_k=None)
    texts = [_score_text(row, args=args) for row in rows]

    total_batches = (len(rows) + args.batch_size - 1) // args.batch_size
    grouped_stats: dict[tuple[str, str, str], dict[str, float]] = {}
    csv_fields: list[str] | None = None
    input_fields = _ordered_field_union(rows)
    rows_scored = 0
    for batch_idx, start in enumerate(tqdm(range(0, len(rows), args.batch_size), desc="score sentiment"), start=1):
        batch_rows = rows[start : start + args.batch_size]
        batch_texts = texts[start : start + args.batch_size]
        score_kwargs: dict[str, object] = {"truncation": bool(args.truncation)}
        if args.scorer_max_length > 0:
            score_kwargs["max_length"] = args.scorer_max_length
        results = scorer(batch_texts, **score_kwargs)
        scored_batch: list[dict[str, object]] = []
        for row, text, result in zip(batch_rows, batch_texts, results):
            label, label_score, positive = _positive_score(result, positive_label=args.positive_label)
            negative = _label_score(result, label=args.negative_label)
            if negative is None:
                negative = 1.0 - positive
            top_label, top_label_score = _top_label_score(result)
            paper_reward = _paper_logodds_reward(positive)
            scored_row = {
                **row,
                "scored_text": text,
                "scored_text_mode": args.score_text,
                "sentiment_label": label,
                "sentiment_label_score": label_score,
                "top_sentiment_label": top_label,
                "top_sentiment_label_score": top_label_score,
                "positive_sentiment_score": positive,
                "negative_sentiment_score": negative,
                "negative_minus_positive_score": negative - positive,
                "paper_logodds_reward": paper_reward,
                "scorer_model": args.scorer_model,
                "scorer_positive_label": args.positive_label,
                "scorer_negative_label": args.negative_label,
            }
            scored_batch.append(scored_row)
            key = _summary_key(scored_row)
            stats = grouped_stats.setdefault(key, _new_group_stats())
            _update_group_stats(stats, scored_row)

        if not scored_batch:
            continue
        _append_jsonl_rows(args.out_jsonl, scored_batch)
        if args.out_csv is not None:
            if csv_fields is None:
                csv_fields = input_fields + [key for key in scored_batch[0].keys() if key not in input_fields]
            _append_csv_rows(args.out_csv, scored_batch, fieldnames=csv_fields)
        rows_scored += len(scored_batch)

        if args.stream_summary_every_batches > 0 and batch_idx % args.stream_summary_every_batches == 0:
            _write_progress(
                summary_csv=summary_csv,
                progress_json=progress_json,
                grouped=grouped_stats,
                rows_scored=rows_scored,
                rows_total=len(rows),
                batches_done=batch_idx,
                batches_total=total_batches,
            )

    _write_progress(
        summary_csv=summary_csv,
        progress_json=progress_json,
        grouped=grouped_stats,
        rows_scored=rows_scored,
        rows_total=len(rows),
        batches_done=total_batches,
        batches_total=total_batches,
    )

    dump_json(
        args.out_jsonl.parent / "score_manifest.json",
        {
            "input": str(args.input),
            "out_jsonl": str(args.out_jsonl),
            "out_csv": str(args.out_csv) if args.out_csv else "",
            "rows": rows_scored,
            "summary_csv": str(summary_csv),
            "progress_json": str(progress_json) if progress_json else "",
            "scorer_model": args.scorer_model,
            "positive_label": args.positive_label,
            "score_text": args.score_text,
            "truncation": bool(args.truncation),
            "scorer_max_length": args.scorer_max_length,
            "batch_size": args.batch_size,
            "score_field": "positive_sentiment_score",
            "stream_summary_every_batches": args.stream_summary_every_batches,
        },
    )
    print(f"[imdb-sentiment-score] wrote rows={rows_scored} path={args.out_jsonl}", flush=True)


if __name__ == "__main__":
    main()
