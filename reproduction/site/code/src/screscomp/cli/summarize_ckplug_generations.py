from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from screscomp.cli.score_ckplug_generation import (
    _comparison_rows,
    _parse_compare_pairs,
    _summarize,
    _transition_rows,
)
from screscomp.data import dump_csv, load_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize one or more CK-style generation JSONL files.")
    p.add_argument("--generations_jsonl", type=Path, action="append", required=True)
    p.add_argument("--out_summary_csv", type=Path, required=True)
    p.add_argument("--out_comparison_csv", type=Path, default=None)
    p.add_argument("--out_transitions_csv", type=Path, default=None)
    p.add_argument(
        "--compare_pairs",
        type=str,
        default=(
            "strong_rag>ours_delta_context,"
            "strong_rag>ours_delta_prefill_context,"
            "random_delta_context>ours_delta_context,"
            "random_delta_prefill_context>ours_delta_prefill_context,"
            "ckplug_official_ck_shared_strong_alpha0.5>ours_delta_prefill_context,"
            "ckplug_official_ck_shared_strong_alpha0.5>ours_delta_context,"
            "ckplug_official_ck_shared_strong_chat_alpha0.5>ours_delta_prefill_context,"
            "ckplug_official_ck_shared_strong_chat_alpha0.5>ours_delta_context"
        ),
    )
    p.add_argument("--bootstrap_samples", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for path in args.generations_jsonl:
        rows.extend(load_jsonl(path))

    metadata: dict[str, Any] = {}
    compare_pairs = _parse_compare_pairs(args.compare_pairs)
    dump_csv(args.out_summary_csv, _summarize(rows, metadata=metadata))
    if args.out_comparison_csv is not None:
        dump_csv(
            args.out_comparison_csv,
            _comparison_rows(
                rows,
                compare_pairs=compare_pairs,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed,
                metadata=metadata,
            ),
        )
    if args.out_transitions_csv is not None:
        dump_csv(
            args.out_transitions_csv,
            _transition_rows(rows, compare_pairs=compare_pairs, metadata=metadata),
        )
    print(f"[summarize-ckplug-generations] rows={len(rows)} files={len(args.generations_jsonl)}")


if __name__ == "__main__":
    main()
