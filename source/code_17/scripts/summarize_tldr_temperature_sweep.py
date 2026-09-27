from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize TL;DR temperature sweep full-swap judge outputs.")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--temps", nargs="+", required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--bootstrap-iters", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260614)
    return parser.parse_args()


def temp_tag(temp_text: str) -> str:
    temp = float(temp_text)
    if abs(temp) < 1e-9:
        return "temp0"
    if abs(temp - round(temp)) < 1e-9:
        return f"temp{int(round(temp))}"
    return ("temp%.1f" % temp).replace(".", "p")


def judge_dir_for_temp(out_root: Path, sweep_root: Path, temp_text: str) -> Path:
    tag = temp_tag(temp_text)
    if tag == "temp0":
        return out_root / "open_test" / "ds4_old_four_temp0_vs_ppo_temp0_fullswap"
    return sweep_root / f"ds4_old_four_vs_ppo_{tag}_fullswap"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    if not values:
        return float("nan")
    k = (len(values) - 1) * p
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - k) + values[hi] * (k - lo)


def bootstrap_ci(values: list[float], *, iters: int, seed: int) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(iters):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    return percentile(means, 0.025), percentile(means, 0.975)


def read_balanced_values(judge_dir: Path, comparison: str, left: str) -> tuple[list[float], float | None]:
    first = {
        (str(row["comparison"]), str(row["sample_id"])): str(row["winner"])
        for row in load_jsonl(judge_dir / "judge_first_pass.jsonl")
        if row.get("status") == "ok"
    }
    review = {
        (str(row["comparison"]), str(row["sample_id"])): str(row["winner"])
        for row in load_jsonl(judge_dir / "judge_order_swap_review.jsonl")
        if row.get("status") == "ok"
    }
    sample_ids = sorted(
        sid for (comp, sid) in set(first).intersection(review) if comp == comparison
    )
    values = []
    agreements = 0
    for sample_id in sample_ids:
        first_winner = first[(comparison, sample_id)]
        review_winner = review[(comparison, sample_id)]
        values.append(((1.0 if first_winner == left else 0.0) + (1.0 if review_winner == left else 0.0)) / 2.0)
        agreements += int(first_winner == review_winner)
    agreement_rate = agreements / len(sample_ids) if sample_ids else None
    return values, agreement_rate


def main() -> None:
    args = parse_args()
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for temp_text in args.temps:
        tag = temp_tag(temp_text)
        judge_dir = judge_dir_for_temp(args.out_root, args.sweep_root, temp_text)
        if not (judge_dir / "pairwise_summary.csv").exists():
            rows.append(
                {
                    "temperature": temp_text,
                    "tag": tag,
                    "comparison": "",
                    "left": "",
                    "right": "",
                    "n": 0,
                    "balanced_left_win_rate": "",
                    "bootstrap95_low": "",
                    "bootstrap95_high": "",
                    "order_swap_agreement_rate": "",
                    "judge_dir": str(judge_dir),
                    "status": "missing",
                }
            )
            continue
        with (judge_dir / "pairwise_summary.csv").open(newline="", encoding="utf-8") as stream:
            summary_rows = list(csv.DictReader(stream))
        for summary_row in summary_rows:
            comparison = summary_row["pair"]
            left = summary_row["left"]
            values, agreement_rate = read_balanced_values(judge_dir, comparison, left)
            low, high = bootstrap_ci(
                values,
                iters=args.bootstrap_iters,
                seed=args.seed + int(round(float(temp_text) * 1000)) + len(rows),
            )
            rows.append(
                {
                    "temperature": temp_text,
                    "tag": tag,
                    "comparison": comparison,
                    "left": left,
                    "right": summary_row["right"],
                    "n": len(values),
                    "balanced_left_win_rate": sum(values) / len(values) if values else "",
                    "bootstrap95_low": low,
                    "bootstrap95_high": high,
                    "order_swap_agreement_rate": agreement_rate if agreement_rate is not None else "",
                    "judge_dir": str(judge_dir),
                    "status": "ok" if values else "missing_judgments",
                }
            )
    fieldnames = [
        "temperature",
        "tag",
        "comparison",
        "left",
        "right",
        "n",
        "balanced_left_win_rate",
        "bootstrap95_low",
        "bootstrap95_high",
        "order_swap_agreement_rate",
        "judge_dir",
        "status",
    ]
    with args.out_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[sweep-summary] wrote {args.out_csv}")


if __name__ == "__main__":
    main()
