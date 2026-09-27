from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import random
from typing import Any

from screscomp.data import dump_csv, dump_json, load_csv


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Select an admitted pair subset for discovery. Source-balanced mode keeps at most "
            "per_source rows per source id; random_rows mode samples rows uniformly after filtering."
        )
    )
    p.add_argument("--pairs-csv", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--event", type=str, default="")
    p.add_argument("--split", type=str, default="train")
    p.add_argument(
        "--selection-mode",
        choices=["source_balanced", "random_rows"],
        default="source_balanced",
        help="source_balanced reproduces the existing per-source selector; random_rows draws a seeded row sample.",
    )
    p.add_argument("--source-key", type=str, default="source_sample_id")
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--per-source", type=int, default=1)
    p.add_argument("--start", type=int, default=0)
    p.add_argument(
        "--shuffle",
        action="store_true",
        help="Randomize source order and within-source row order before subset selection.",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def _admitted(row: dict[str, Any]) -> bool:
    return str(row.get("admitted", "1")).strip() not in {"0", "false", "False"}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.selection_mode == "source_balanced" and args.per_source <= 0:
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
        if args.selection_mode == "source_balanced":
            source_id = str(row.get(args.source_key, "")).strip()
            if not source_id:
                continue
        filtered.append(row)

    if args.selection_mode == "random_rows":
        rng = random.Random(int(args.seed))
        selected = list(filtered)
        if args.shuffle:
            rng.shuffle(selected)
        elif args.max_rows > 0 and len(selected) > args.max_rows:
            selected = rng.sample(selected, k=int(args.max_rows))
        start = max(0, int(args.start))
        if args.shuffle:
            selected = selected[start:]
        elif start > 0:
            selected = selected[start:]
        if args.max_rows > 0:
            selected = selected[: int(args.max_rows)]
        if not selected:
            raise SystemExit(
                f"No rows selected from {args.pairs_csv} split={args.split!r} event={args.event!r} "
                f"selection_mode={args.selection_mode!r}"
            )
        dump_csv(args.out_csv, selected)
        dump_json(
            args.out_csv.with_suffix(".manifest.json"),
            {
                "pairs_csv": str(args.pairs_csv),
                "out_csv": str(args.out_csv),
                "event": args.event,
                "split": args.split,
                "selection_mode": args.selection_mode,
                "max_rows": args.max_rows,
                "start": args.start,
                "shuffle": bool(args.shuffle),
                "seed": int(args.seed),
                "input_rows": len(rows),
                "eligible_rows": len(filtered),
                "selected_rows": len(selected),
                "selection_semantics": (
                    "Rows are filtered by admitted/event/split and then selected as a seeded random row subset. "
                    "When shuffle=1 the filtered pool is shuffled first and start/max_rows are applied on that order; "
                    "otherwise max_rows>0 uses a uniform seeded sample without replacement."
                ),
            },
        )
        print(
            f"[cecm-select-pair-subset] selected={len(selected)} mode=random_rows out={args.out_csv}",
            flush=True,
        )
        return

    grouped: dict[str, list[dict[str, str]]] = {}
    source_order: list[str] = []
    for row in filtered:
        source_id = str(row.get(args.source_key, "")).strip()
        if source_id not in grouped:
            grouped[source_id] = []
            source_order.append(source_id)
        grouped[source_id].append(row)

    if args.shuffle:
        rng = random.Random(int(args.seed))
        rng.shuffle(source_order)
        for source_id in source_order:
            rng.shuffle(grouped[source_id])

    source_order = source_order[max(0, args.start) :]
    selected: list[dict[str, str]] = []
    source_counts: Counter[str] = Counter()
    for source_id in source_order:
        rows_for_source = grouped[source_id]
        take_n = min(len(rows_for_source), int(args.per_source))
        for row in rows_for_source[:take_n]:
            source_counts[source_id] += 1
            selected.append(row)
            if args.max_rows > 0 and len(selected) >= args.max_rows:
                break
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
            "selection_mode": args.selection_mode,
            "source_key": args.source_key,
            "max_rows": args.max_rows,
            "per_source": args.per_source,
            "shuffle": bool(args.shuffle),
            "seed": int(args.seed),
            "input_rows": len(rows),
            "eligible_rows": len(filtered),
            "eligible_sources": len(grouped),
            "selected_rows": len(selected),
            "selected_sources": len(source_counts),
            "selection_semantics": (
                "Rows are selected at the source-question level with at most per_source rows per source question. "
                "When shuffle=1, source order and within-source row order are randomized with the provided seed. "
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
