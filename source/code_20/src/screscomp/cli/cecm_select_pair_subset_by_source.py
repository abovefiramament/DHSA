from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from screscomp.data import dump_csv, dump_json, load_csv


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Select a source-balanced subset of admitted pair rows. This is used when a full "
            "trajectory pool has multiple rows per question but component discovery should use "
            "one trajectory per source question."
        )
    )
    p.add_argument("--pairs-csv", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--event", type=str, default="")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--source-key", type=str, default="source_sample_id")
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--per-source", type=int, default=1)
    p.add_argument("--start", type=int, default=0)
    return p.parse_args(argv)


def _admitted(row: dict[str, Any]) -> bool:
    return str(row.get("admitted", "1")).strip() not in {"0", "false", "False"}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.per_source <= 0:
        raise SystemExit("--per-source must be positive")
    rows = load_csv(args.pairs_csv)
    filtered: list[dict[str, str]] = []
    for row in rows:
        if not _admitted(row):
            continue
        if args.event and str(row.get("event", "")) != args.event:
            continue
        if args.split and str(row.get("split", "")) != args.split:
            continue
        source_id = str(row.get(args.source_key, "")).strip()
        if not source_id:
            continue
        filtered.append(row)

    filtered = filtered[max(0, args.start) :]
    selected: list[dict[str, str]] = []
    source_counts: Counter[str] = Counter()
    for row in filtered:
        source_id = str(row.get(args.source_key, "")).strip()
        if source_counts[source_id] >= args.per_source:
            continue
        source_counts[source_id] += 1
        selected.append(row)
        if args.max_rows > 0 and len(selected) >= args.max_rows:
            break

    if not selected:
        raise SystemExit(
            f"No rows selected from {args.pairs_csv} split={args.split!r} event={args.event!r} "
            f"source_key={args.source_key!r}"
        )

    dump_csv(args.out_csv, selected)
    dump_json(
        args.out_csv.with_suffix(".manifest.json"),
        {
            "pairs_csv": str(args.pairs_csv),
            "out_csv": str(args.out_csv),
            "event": args.event,
            "split": args.split,
            "source_key": args.source_key,
            "max_rows": args.max_rows,
            "per_source": args.per_source,
            "input_rows": len(rows),
            "eligible_rows": len(filtered),
            "selected_rows": len(selected),
            "selected_sources": len(source_counts),
            "selection_semantics": (
                "Rows are selected in existing order with at most per_source rows per source question. "
                "For K-rollout trajectory pools, per_source=1 gives one trajectory-answer-slot state "
                "per question for component/head discovery while leaving training free to use all "
                "trajectory rows."
            ),
        },
    )
    print(
        (
            f"[cecm-select-pair-subset] selected={len(selected)} sources={len(source_counts)} "
            f"out={args.out_csv}"
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
