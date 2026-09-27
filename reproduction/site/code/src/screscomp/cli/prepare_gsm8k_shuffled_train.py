from __future__ import annotations

import argparse
import random
from pathlib import Path

from screscomp.data import dump_json, dump_jsonl, load_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare deterministic shuffled GSM8K train variants.")
    p.add_argument("--input-jsonl", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seeds", type=str, default="11,23,37,47,59")
    p.add_argument("--train-rows", type=int, default=5000)
    p.add_argument("--select-rows", type=int, default=1000)
    p.add_argument("--stop-rows", type=int, default=1473)
    return p.parse_args()


def _parse_seeds(raw: str) -> list[int]:
    out: list[int] = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    if not out:
        raise ValueError("No seeds provided")
    return out


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input_jsonl)
    total = len(rows)
    if total == 0:
        raise ValueError(f"Empty input: {args.input_jsonl}")

    seeds = _parse_seeds(args.seeds)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.train_rows < 0 or args.select_rows < 0 or args.stop_rows < 0:
        raise ValueError("Split sizes must be non-negative")
    if args.train_rows + args.select_rows + args.stop_rows > total:
        raise ValueError(
            f"Requested split rows exceed total train size: "
            f"{args.train_rows}+{args.select_rows}+{args.stop_rows}>{total}"
        )

    variants: list[dict[str, object]] = []
    for seed in seeds:
        rng = random.Random(seed)
        order = list(range(total))
        rng.shuffle(order)
        shuffled_rows = [rows[idx] for idx in order]

        jsonl_path = out_dir / f"train_seed{seed}.jsonl"
        dump_jsonl(jsonl_path, shuffled_rows)

        manifest = {
            "dataset": "gsm8k_train",
            "seed": seed,
            "source_jsonl": str(args.input_jsonl),
            "shuffled_jsonl": str(jsonl_path),
            "total_rows": total,
            "order": order,
            "suggested_splits": {
                "train_pool": {
                    "start": 0,
                    "rows": args.train_rows,
                },
                "round_select": {
                    "start": args.train_rows,
                    "rows": args.select_rows,
                },
                "stop_eval": {
                    "start": args.train_rows + args.select_rows,
                    "rows": args.stop_rows,
                },
            },
            "semantics": (
                "Rows are a deterministic permutation of official GSM8K train. "
                "Existing start/max_rows slicing can be reused directly on shuffled_jsonl."
            ),
        }
        manifest_path = out_dir / f"train_seed{seed}.manifest.json"
        dump_json(manifest_path, manifest)
        variants.append(
            {
                "seed": seed,
                "jsonl": str(jsonl_path),
                "manifest": str(manifest_path),
            }
        )

    summary = {
        "dataset": "gsm8k_train",
        "source_jsonl": str(args.input_jsonl),
        "total_rows": total,
        "seeds": seeds,
        "variants": variants,
    }
    summary_path = out_dir / "shuffle_summary.json"
    dump_json(summary_path, summary)
    print(f"[gsm8k-shuffle] wrote {len(variants)} variants under {out_dir}")
    print(f"[gsm8k-shuffle] summary={summary_path}")


if __name__ == "__main__":
    main()
