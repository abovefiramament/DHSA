from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from screscomp.data import dump_csv


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select the best alpha per IMDb control from a matrix score summary.")
    p.add_argument("--matrix-csv", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--metric", type=str, default="mean_positive_sentiment_score")
    p.add_argument("--reward-field", type=str, default="", help="Optional explicit reward field; defaults to --metric.")
    p.add_argument("--kl-field", type=str, default="mean_sequence_kl")
    p.add_argument(
        "--selection-mode",
        choices=["reward_only", "reward_under_kl_cap", "min_kl_within_reward_ratio"],
        default="reward_only",
    )
    p.add_argument("--reward-ratio", type=float, default=0.99)
    p.add_argument("--kl-cap", type=float, default=0.0, help="Only used by reward_under_kl_cap; <=0 disables the cap.")
    p.add_argument(
        "--fallback-reward-only-if-missing-kl",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If KL-based selection is requested but the KL field is absent/empty, fall back to reward_only.",
    )
    p.add_argument("--groups", type=str, default="", help="Optional comma-separated control names to keep.")
    p.add_argument("--split", type=str, default="eval")
    return p.parse_args(argv)


def _parse_groups(raw: str) -> set[str]:
    return {item.strip() for item in str(raw or "").split(",") if item.strip()}


def _f(value: Any, default: float = 0.0) -> float:
    try:
        text = str(value).strip()
        return float(text) if text else default
    except Exception:
        return default


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    groups = _parse_groups(args.groups)
    reward_field = args.reward_field or args.metric
    with args.matrix_csv.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    selected = [
        row
        for row in rows
        if (not args.split or str(row.get("split", "")) == args.split)
        and (not groups or str(row.get("control_name", "")) in groups)
    ]
    if not selected:
        raise SystemExit(f"No matching rows in {args.matrix_csv}")

    by_group: dict[str, list[dict[str, str]]] = {}
    for row in selected:
        by_group.setdefault(str(row.get("control_name", "")), []).append(row)

    effective_mode = args.selection_mode
    if args.selection_mode != "reward_only":
        has_kl = any(str(row.get(args.kl_field, "")).strip() for row in selected)
        if not has_kl:
            if not args.fallback_reward_only_if_missing_kl:
                raise SystemExit(
                    f"selection_mode={args.selection_mode} requires KL field {args.kl_field!r}, but it is missing in {args.matrix_csv}"
                )
            effective_mode = "reward_only"
            print(
                f"[select-imdb-best-alpha] KL field {args.kl_field!r} missing; falling back to reward_only",
                flush=True,
            )

    out_rows: list[dict[str, object]] = []
    for control_name, group_rows in sorted(by_group.items()):
        baseline = None
        best_reward = None
        for row in group_rows:
            alpha = _f(row.get("alpha"), 0.0)
            reward_value = _f(row.get(reward_field), 0.0)
            if abs(alpha) < 1e-12:
                baseline = row
            if best_reward is None or reward_value > best_reward:
                best_reward = reward_value

        assert best_reward is not None

        candidates = list(group_rows)
        if effective_mode == "reward_under_kl_cap" and args.kl_cap > 0:
            constrained = [row for row in candidates if _f(row.get(args.kl_field), float("inf")) <= args.kl_cap]
            if constrained:
                candidates = constrained
        elif effective_mode == "min_kl_within_reward_ratio":
            threshold = float(args.reward_ratio) * float(best_reward)
            constrained = [row for row in candidates if _f(row.get(reward_field), 0.0) >= threshold]
            if constrained:
                candidates = constrained

        best = None
        best_key = None
        for row in candidates:
            alpha = _f(row.get("alpha"), 0.0)
            reward_value = _f(row.get(reward_field), 0.0)
            kl_value = _f(row.get(args.kl_field), float("inf"))
            if effective_mode == "min_kl_within_reward_ratio":
                key = (-kl_value, reward_value, -abs(alpha), -alpha)
            else:
                key = (reward_value, -abs(alpha), -alpha)
            if best is None or key > best_key:
                best = row
                best_key = key

        assert best is not None
        baseline_metric = _f(baseline.get(reward_field), 0.0) if baseline is not None else 0.0
        best_metric = _f(best.get(reward_field), 0.0)
        out_rows.append(
            {
                "control_name": control_name,
                "best_alpha": _f(best.get("alpha"), 0.0),
                "best_metric": best_metric,
                "baseline_metric": baseline_metric,
                "delta_vs_baseline": best_metric - baseline_metric,
                "metric": reward_field,
                "selection_mode": effective_mode,
                "reward_ratio": args.reward_ratio,
                "kl_field": args.kl_field,
                "best_kl": _f(best.get(args.kl_field), 0.0),
                "split": args.split,
                "baseline_n": int(_f(baseline.get("n"), 0.0)) if baseline is not None else 0,
                "best_n": int(_f(best.get("n"), 0.0)),
            }
        )

    dump_csv(args.out_csv, out_rows)
    print(f"[select-imdb-best-alpha] groups={len(out_rows)} path={args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
