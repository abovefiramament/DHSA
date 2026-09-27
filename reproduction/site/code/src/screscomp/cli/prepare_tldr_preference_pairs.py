from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from screscomp.data import dump_csv, dump_json, dump_jsonl


DEFAULT_EVENT = "tldr_summary_preference"
DEFAULT_PAIR_MODE = "tldr_summary_preference"
DEFAULT_PROMPT_FORMAT = "openai_tldr_structured_v1"


@dataclass(frozen=True, slots=True)
class ComparisonRow:
    row_index: int
    raw_split: str
    split: str
    sample_id: str
    prompt_id: str
    prompt: str
    chosen: str
    rejected: str
    batch: str
    worker: str
    choice: int
    chosen_policy: str
    rejected_policy: str
    chosen_note: str
    rejected_note: str
    info_id: str
    subreddit: str
    title: str
    post: str
    raw: dict[str, Any]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Convert OpenAI TL;DR summary comparison files into generic CECM/CAST "
            "teacher-forced preference pairs and prompt-only eval inputs."
        )
    )
    p.add_argument("--comparisons-dir", type=Path, required=True, help="Directory containing batch*.json comparison files.")
    p.add_argument(
        "--filtered-dir",
        type=Path,
        default=None,
        help="Optional directory with tldr_3_filtered {train,valid,test}.jsonl for prompt-only eval exports.",
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--event", type=str, default=DEFAULT_EVENT)
    p.add_argument("--comparisons-glob", type=str, default="batch*.json")
    p.add_argument("--prompt-format", choices=[DEFAULT_PROMPT_FORMAT], default=DEFAULT_PROMPT_FORMAT)
    p.add_argument("--keep-rejected-rows", action="store_true")
    return p.parse_args(argv)


def _normalize_split(raw: str) -> str:
    value = str(raw or "").strip().lower()
    if not value:
        return "train"
    if value.startswith("train"):
        return "train"
    if value.startswith("valid") or value.startswith("val"):
        return "val"
    if value.startswith("test"):
        return "test"
    return value


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _normalize_subreddit(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.startswith("r/"):
        return text
    return f"r/{text.lstrip('/')}"


def _normalize_summary_continuation(value: Any) -> str:
    text = _as_text(value).strip()
    if not text:
        return ""
    return f" {text}"


def _format_prompt(info: dict[str, Any], *, prompt_format: str) -> str:
    if prompt_format != DEFAULT_PROMPT_FORMAT:
        raise ValueError(f"Unsupported prompt_format={prompt_format!r}")
    subreddit = _normalize_subreddit(_as_text(info.get("subreddit")))
    title = _as_text(info.get("title")).strip()
    post = _as_text(info.get("post")).strip()
    parts: list[str] = []
    if subreddit:
        parts.append(f"SUBREDDIT: {subreddit}")
    if title:
        parts.append(f"TITLE: {title}")
    if post:
        parts.append(f"POST: {post}")
    parts.append("TL;DR:")
    return "\n".join(parts)


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object rows in {path}")
            rows.append(value)
    return rows


def _comparison_candidate(
    row: dict[str, Any],
    *,
    row_index: int,
    event: str,
    prompt_format: str,
) -> tuple[ComparisonRow | None, dict[str, object] | None]:
    info = row.get("info")
    summaries = row.get("summaries")
    choice = row.get("choice")
    if not isinstance(info, dict):
        return None, {"row_index": row_index, "admit_reason": "missing_info"}
    if not isinstance(summaries, list) or len(summaries) != 2 or not all(isinstance(item, dict) for item in summaries):
        return None, {"row_index": row_index, "admit_reason": "expected_two_summary_dicts"}
    if choice not in (0, 1):
        return None, {"row_index": row_index, "admit_reason": "invalid_choice"}
    chosen_raw = summaries[int(choice)]
    rejected_raw = summaries[1 - int(choice)]
    chosen = _normalize_summary_continuation(chosen_raw.get("text"))
    rejected = _normalize_summary_continuation(rejected_raw.get("text"))
    prompt = _format_prompt(info, prompt_format=prompt_format)
    info_id = _as_text(info.get("id")).strip()
    if not info_id or not prompt or not chosen or not rejected:
        return None, {"row_index": row_index, "admit_reason": "missing_required_text"}
    split = _normalize_split(_as_text(row.get("split")))
    return (
        ComparisonRow(
            row_index=row_index,
            raw_split=_as_text(row.get("split")).strip(),
            split=split,
            sample_id=f"{event}_{split}_{row_index + 1:08d}",
            prompt_id=info_id,
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            batch=_as_text(row.get("batch")).strip(),
            worker=_as_text(row.get("worker")).strip(),
            choice=int(choice),
            chosen_policy=_as_text(chosen_raw.get("policy")).strip(),
            rejected_policy=_as_text(rejected_raw.get("policy")).strip(),
            chosen_note=_as_text(chosen_raw.get("note")).strip(),
            rejected_note=_as_text(rejected_raw.get("note")).strip(),
            info_id=info_id,
            subreddit=_normalize_subreddit(_as_text(info.get("subreddit"))),
            title=_as_text(info.get("title")).strip(),
            post=_as_text(info.get("post")).strip(),
            raw=row,
        ),
        None,
    )


def _pair_row(candidate: ComparisonRow, *, event: str) -> dict[str, object]:
    return {
        "sample_id": candidate.sample_id,
        "split": candidate.split,
        "event": event,
        "admitted": 1,
        "admit_reason": "ok",
        "prompt": candidate.prompt,
        "y_plus": candidate.chosen,
        "y_minus": candidate.rejected,
        "y_plus_continuation": candidate.chosen,
        "y_minus_continuation": candidate.rejected,
        "y_plus_continuations_json": json.dumps([candidate.chosen], ensure_ascii=False),
        "y_minus_continuations_json": json.dumps([candidate.rejected], ensure_ascii=False),
        "pair_mode": DEFAULT_PAIR_MODE,
        "prompt_sample_id": candidate.info_id,
        "prompt_id": candidate.prompt_id,
        "source_row_index": candidate.row_index,
        "row_index": candidate.row_index,
        "comparison_batch": candidate.batch,
        "comparison_worker": candidate.worker,
        "comparison_choice": candidate.choice,
        "comparison_raw_split": candidate.raw_split,
        "y_plus_policy": candidate.chosen_policy,
        "y_minus_policy": candidate.rejected_policy,
        "y_plus_note": candidate.chosen_note,
        "y_minus_note": candidate.rejected_note,
        "raw_post_id": candidate.info_id,
        "raw_subreddit": candidate.subreddit,
        "raw_title": candidate.title,
        "raw_post": candidate.post,
        "prompt_format": DEFAULT_PROMPT_FORMAT,
    }


def _pair_jsonl_rows(candidates: Iterable[ComparisonRow], *, event: str) -> Iterable[dict[str, object]]:
    for candidate in candidates:
        yield {
            "pair": _pair_row(candidate, event=event),
            "raw_row": candidate.raw,
        }


def _load_comparison_candidates(args: argparse.Namespace) -> tuple[list[ComparisonRow], list[dict[str, object]], dict[str, int]]:
    candidates: list[ComparisonRow] = []
    rejected: list[dict[str, object]] = []
    counts_by_source_split: dict[str, int] = {}
    row_index = 0
    for path in sorted(args.comparisons_dir.glob(args.comparisons_glob)):
        for raw in _read_jsonl_objects(path):
            candidate, rejected_row = _comparison_candidate(
                raw,
                row_index=row_index,
                event=args.event,
                prompt_format=args.prompt_format,
            )
            if candidate is None:
                rejected.append(
                    {
                        "source_path": str(path),
                        **(rejected_row or {"row_index": row_index, "admit_reason": "unknown"}),
                    }
                )
            else:
                candidates.append(candidate)
                counts_by_source_split[candidate.raw_split or candidate.split] = (
                    counts_by_source_split.get(candidate.raw_split or candidate.split, 0) + 1
                )
            row_index += 1
    return candidates, rejected, counts_by_source_split


def _prompt_record_from_filtered_row(
    row: dict[str, Any],
    *,
    row_index: int,
    raw_split_name: str,
    prompt_format: str,
) -> dict[str, object] | None:
    info = {
        "id": _as_text(row.get("id")),
        "subreddit": _as_text(row.get("subreddit")),
        "title": _as_text(row.get("title")),
        "post": _as_text(row.get("post")),
    }
    prompt = _format_prompt(info, prompt_format=prompt_format)
    if not _as_text(row.get("id")).strip() or not prompt:
        return None
    summary = _normalize_summary_continuation(row.get("summary"))
    split = _normalize_split(raw_split_name)
    return {
        "prompt_id": _as_text(row.get("id")).strip(),
        "sample_id": f"tldr_prompt_{split}_{row_index + 1:08d}",
        "split": split,
        "raw_split": raw_split_name,
        "prompt": prompt,
        "reference_summary": summary,
        "raw_subreddit": _normalize_subreddit(_as_text(row.get("subreddit"))),
        "raw_title": _as_text(row.get("title")).strip(),
        "raw_post": _as_text(row.get("post")).strip(),
        "source_row_index": row_index,
        "prompt_format": prompt_format,
    }


def _export_filtered_prompts(filtered_dir: Path, *, out_dir: Path, prompt_format: str) -> dict[str, object]:
    prompts_dir = out_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    split_files = {
        "train": filtered_dir / "train.jsonl",
        "valid": filtered_dir / "valid.jsonl",
        "test": filtered_dir / "test.jsonl",
    }
    manifest: dict[str, object] = {"source_dir": str(filtered_dir), "splits": {}}
    for raw_name, path in split_files.items():
        if not path.exists():
            continue
        rows = _read_jsonl_objects(path)
        prompt_rows = [
            record
            for idx, row in enumerate(rows)
            if (record := _prompt_record_from_filtered_row(row, row_index=idx, raw_split_name=raw_name, prompt_format=prompt_format))
            is not None
        ]
        normalized_split = _normalize_split(raw_name)
        dump_jsonl(prompts_dir / f"{normalized_split}_prompts.jsonl", prompt_rows)
        manifest["splits"][normalized_split] = {
            "raw_split": raw_name,
            "rows": len(prompt_rows),
            "path": str(prompts_dir / f"{normalized_split}_prompts.jsonl"),
        }
    dump_json(prompts_dir / "prompt_manifest.json", manifest)
    return manifest


def build_tldr_preference_pairs(args: argparse.Namespace) -> dict[str, object]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pair_dir = args.out_dir
    candidates, rejected, counts_by_source_split = _load_comparison_candidates(args)
    pair_rows = [_pair_row(candidate, event=args.event) for candidate in candidates]
    split_counts: dict[str, int] = {}
    unique_prompts_by_split: dict[str, set[str]] = {}
    for candidate in candidates:
        split_counts[candidate.split] = split_counts.get(candidate.split, 0) + 1
        unique_prompts_by_split.setdefault(candidate.split, set()).add(candidate.prompt_id)

    dump_csv(pair_dir / "pairs.csv", pair_rows)
    dump_jsonl(pair_dir / "pairs.jsonl", _pair_jsonl_rows(candidates, event=args.event))
    if args.keep_rejected_rows:
        dump_csv(pair_dir / "rejected_pairs.csv", rejected)

    filtered_prompt_manifest = None
    if args.filtered_dir is not None and args.filtered_dir.exists():
        filtered_prompt_manifest = _export_filtered_prompts(
            args.filtered_dir,
            out_dir=args.out_dir,
            prompt_format=args.prompt_format,
        )

    manifest = {
        "adapter": "tldr_openai_summary_comparisons",
        "event": args.event,
        "prompt_format": args.prompt_format,
        "pair_mode": DEFAULT_PAIR_MODE,
        "comparisons_dir": str(args.comparisons_dir),
        "comparisons_glob": args.comparisons_glob,
        "filtered_dir": str(args.filtered_dir) if args.filtered_dir is not None else "",
        "admitted_pairs": len(pair_rows),
        "rejected_rows": len(rejected),
        "actual_counts": split_counts,
        "unique_prompts_by_split": {
            split: len(prompt_ids) for split, prompt_ids in unique_prompts_by_split.items()
        },
        "source_split_counts": counts_by_source_split,
        "semantics": (
            "Teacher-forced TL;DR preference pairs: x is the structured Reddit post prompt ending in "
            "'TL;DR:', y_plus is the human-preferred summary, y_minus is the non-chosen summary."
        ),
        "filtered_prompt_manifest": filtered_prompt_manifest,
    }
    dump_json(pair_dir / "pair_build_manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = build_tldr_preference_pairs(args)
    print(
        json.dumps(
            {
                "admitted_pairs": manifest["admitted_pairs"],
                "actual_counts": manifest["actual_counts"],
                "unique_prompts_by_split": manifest["unique_prompts_by_split"],
                "rejected_rows": manifest["rejected_rows"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
