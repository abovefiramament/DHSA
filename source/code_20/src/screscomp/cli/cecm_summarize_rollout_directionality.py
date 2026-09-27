from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

from screscomp.data import dump_csv, dump_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Summarize rollout zero-ablation samples into component-level directionality metrics. "
            "This keeps positive and negative support masses separate instead of relying only on "
            "the signed mean delta."
        )
    )
    p.add_argument("--delta-csv", type=Path, required=True, help="component_delta_samples.csv from rollout scan")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--target-threshold",
        type=float,
        default=0.5,
        help="Classifier threshold used to define target-active vs target-inactive states.",
    )
    p.add_argument(
        "--strong-delta-threshold",
        type=float,
        default=0.2,
        help="Absolute delta threshold used for strong positive/negative support rates.",
    )
    p.add_argument(
        "--min-ablated-format-ok",
        type=float,
        default=0.9,
        help="Minimum ablated format-ok rate for a component to count as usable.",
    )
    p.add_argument(
        "--max-format-collapse-rate",
        type=float,
        default=0.1,
        help="Maximum allowed format collapse rate for a component to count as usable.",
    )
    p.add_argument("--top-k", type=int, default=8)
    return p.parse_args(argv)


def _iter_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        yield from csv.DictReader(f)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        text = str(value).strip()
        return float(text) if text else default
    except Exception:
        return default


def _rate(count: int, total: int) -> float:
    return count / total if total else math.nan


def _mean(total: float, count: int) -> float:
    return total / count if count else math.nan


class DirectionalityAccumulator:
    def __init__(self, *, component_id: str, layer_idx: int, component_type: str, event: str, apply_mode: str) -> None:
        self.component_id = component_id
        self.layer_idx = layer_idx
        self.component_type = component_type
        self.event = event
        self.apply_mode = apply_mode

        self.n = 0
        self.delta_sum = 0.0
        self.pos_mass_sum = 0.0
        self.neg_mass_sum = 0.0
        self.pos_count = 0
        self.neg_count = 0
        self.strong_pos_count = 0
        self.strong_neg_count = 0
        self.pos_delta_sum = 0.0
        self.neg_delta_sum = 0.0

        self.full_format_ok_sum = 0.0
        self.ablated_format_ok_sum = 0.0
        self.format_collapse_sum = 0.0
        self.healthy_n = 0

        self.healthy_delta_sum = 0.0
        self.healthy_pos_mass_sum = 0.0
        self.healthy_neg_mass_sum = 0.0
        self.healthy_pos_count = 0
        self.healthy_neg_count = 0
        self.healthy_pos_delta_sum = 0.0
        self.healthy_neg_delta_sum = 0.0

        self.healthy_full_pos_count = 0
        self.healthy_ablated_pos_count = 0
        self.healthy_pos_to_nonpos_count = 0
        self.healthy_nonpos_to_pos_count = 0
        self.healthy_stable_pos_count = 0
        self.healthy_stable_nonpos_count = 0

    def update(self, row: dict[str, str], *, target_threshold: float, strong_delta_threshold: float) -> None:
        delta = _as_float(row.get("delta"))
        full_format_ok = int(_as_float(row.get("full_format_ok")))
        ablated_format_ok = int(_as_float(row.get("ablated_format_ok")))
        format_collapse = int(_as_float(row.get("format_collapse")))

        self.n += 1
        self.delta_sum += delta
        self.pos_mass_sum += max(delta, 0.0)
        self.neg_mass_sum += max(-delta, 0.0)
        self.full_format_ok_sum += full_format_ok
        self.ablated_format_ok_sum += ablated_format_ok
        self.format_collapse_sum += format_collapse

        if delta > 0:
            self.pos_count += 1
            self.pos_delta_sum += delta
        elif delta < 0:
            self.neg_count += 1
            self.neg_delta_sum += -delta
        if delta >= strong_delta_threshold:
            self.strong_pos_count += 1
        if delta <= -strong_delta_threshold:
            self.strong_neg_count += 1

        healthy = full_format_ok == 1 and ablated_format_ok == 1
        if not healthy:
            return

        full_target_score = _as_float(row.get("full_target_score"))
        ablated_target_score = _as_float(row.get("ablated_target_score"))
        full_pos = full_target_score >= target_threshold
        ablated_pos = ablated_target_score >= target_threshold

        self.healthy_n += 1
        self.healthy_delta_sum += delta
        self.healthy_pos_mass_sum += max(delta, 0.0)
        self.healthy_neg_mass_sum += max(-delta, 0.0)
        if delta > 0:
            self.healthy_pos_count += 1
            self.healthy_pos_delta_sum += delta
        elif delta < 0:
            self.healthy_neg_count += 1
            self.healthy_neg_delta_sum += -delta

        if full_pos:
            self.healthy_full_pos_count += 1
        if ablated_pos:
            self.healthy_ablated_pos_count += 1
        if full_pos and not ablated_pos:
            self.healthy_pos_to_nonpos_count += 1
        elif (not full_pos) and ablated_pos:
            self.healthy_nonpos_to_pos_count += 1
        elif full_pos and ablated_pos:
            self.healthy_stable_pos_count += 1
        else:
            self.healthy_stable_nonpos_count += 1

    def row(
        self,
        *,
        min_ablated_format_ok: float,
        max_format_collapse_rate: float,
    ) -> dict[str, object]:
        full_nonpos_count = self.healthy_n - self.healthy_full_pos_count
        mean_ablated_format_ok = _mean(self.ablated_format_ok_sum, self.n)
        format_collapse_rate = _mean(self.format_collapse_sum, self.n)
        health_ok = int(
            mean_ablated_format_ok >= min_ablated_format_ok and format_collapse_rate <= max_format_collapse_rate
        )
        healthy_mass_gap = _mean(self.healthy_pos_mass_sum, self.healthy_n) - _mean(
            self.healthy_neg_mass_sum, self.healthy_n
        )
        support_direction = "neutral"
        if healthy_mass_gap > 0:
            support_direction = "positive_support"
        elif healthy_mass_gap < 0:
            support_direction = "negative_support"

        return {
            "component_id": self.component_id,
            "layer_idx": self.layer_idx,
            "component_type": self.component_type,
            "event": self.event,
            "apply_mode": self.apply_mode,
            "n": self.n,
            "mean_delta": _mean(self.delta_sum, self.n),
            "positive_rate": _rate(self.pos_count, self.n),
            "negative_rate": _rate(self.neg_count, self.n),
            "pos_mass": _mean(self.pos_mass_sum, self.n),
            "neg_mass": _mean(self.neg_mass_sum, self.n),
            "mass_gap": _mean(self.pos_mass_sum, self.n) - _mean(self.neg_mass_sum, self.n),
            "pos_active_mean": _mean(self.pos_delta_sum, self.pos_count),
            "neg_active_mean": _mean(self.neg_delta_sum, self.neg_count),
            "strong_positive_rate": _rate(self.strong_pos_count, self.n),
            "strong_negative_rate": _rate(self.strong_neg_count, self.n),
            "mean_full_format_ok": _mean(self.full_format_ok_sum, self.n),
            "mean_ablated_format_ok": mean_ablated_format_ok,
            "healthy_pair_rate": _rate(self.healthy_n, self.n),
            "format_collapse_rate": format_collapse_rate,
            "health_ok": health_ok,
            "healthy_n": self.healthy_n,
            "healthy_mean_delta": _mean(self.healthy_delta_sum, self.healthy_n),
            "healthy_positive_rate": _rate(self.healthy_pos_count, self.healthy_n),
            "healthy_negative_rate": _rate(self.healthy_neg_count, self.healthy_n),
            "healthy_pos_mass": _mean(self.healthy_pos_mass_sum, self.healthy_n),
            "healthy_neg_mass": _mean(self.healthy_neg_mass_sum, self.healthy_n),
            "healthy_mass_gap": healthy_mass_gap,
            "healthy_pos_active_mean": _mean(self.healthy_pos_delta_sum, self.healthy_pos_count),
            "healthy_neg_active_mean": _mean(self.healthy_neg_delta_sum, self.healthy_neg_count),
            "healthy_full_pos_rate": _rate(self.healthy_full_pos_count, self.healthy_n),
            "healthy_ablated_pos_rate": _rate(self.healthy_ablated_pos_count, self.healthy_n),
            "healthy_pos_to_nonpos_rate": _rate(self.healthy_pos_to_nonpos_count, self.healthy_n),
            "healthy_nonpos_to_pos_rate": _rate(self.healthy_nonpos_to_pos_count, self.healthy_n),
            "healthy_pos_to_nonpos_given_full_pos": _rate(
                self.healthy_pos_to_nonpos_count, self.healthy_full_pos_count
            ),
            "healthy_nonpos_to_pos_given_full_nonpos": _rate(
                self.healthy_nonpos_to_pos_count, full_nonpos_count
            ),
            "healthy_stable_pos_rate": _rate(self.healthy_stable_pos_count, self.healthy_n),
            "healthy_stable_nonpos_rate": _rate(self.healthy_stable_nonpos_count, self.healthy_n),
            "healthy_transition_gap": _rate(self.healthy_pos_to_nonpos_count, self.healthy_n)
            - _rate(self.healthy_nonpos_to_pos_count, self.healthy_n),
            "support_direction": support_direction,
            "usable_support_direction": support_direction if health_ok else "excluded_unhealthy",
        }


def _component_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    by_component: dict[str, DirectionalityAccumulator] = {}
    for row in _iter_csv(args.delta_csv):
        component_id = str(row.get("component_id", "")).strip()
        if not component_id:
            continue
        entry = by_component.setdefault(
            component_id,
            DirectionalityAccumulator(
                component_id=component_id,
                layer_idx=int(_as_float(row.get("layer_idx"))),
                component_type=str(row.get("component_type", "")).strip(),
                event=str(row.get("event", "")).strip(),
                apply_mode=str(row.get("apply_mode", "")).strip(),
            ),
        )
        entry.update(
            row,
            target_threshold=args.target_threshold,
            strong_delta_threshold=args.strong_delta_threshold,
        )
    rows = [
        entry.row(
            min_ablated_format_ok=args.min_ablated_format_ok,
            max_format_collapse_rate=args.max_format_collapse_rate,
        )
        for entry in by_component.values()
    ]
    rows.sort(
        key=lambda row: (
            -int(row["health_ok"]),
            -abs(float(row["healthy_mass_gap"])),
            -abs(float(row["mass_gap"])),
            str(row["component_id"]),
        )
    )
    return rows


def _positive_rank(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    selected = [row for row in rows if int(row["health_ok"]) == 1]
    return sorted(
        selected,
        key=lambda row: (
            -float(row["healthy_pos_mass"]),
            -float(row["healthy_pos_to_nonpos_rate"]),
            -float(row["healthy_pos_active_mean"]),
            str(row["component_id"]),
        ),
    )


def _negative_rank(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    selected = [row for row in rows if int(row["health_ok"]) == 1]
    return sorted(
        selected,
        key=lambda row: (
            -float(row["healthy_neg_mass"]),
            -float(row["healthy_nonpos_to_pos_rate"]),
            -float(row["healthy_neg_active_mean"]),
            str(row["component_id"]),
        ),
    )


def _summary_rows(rows: list[dict[str, object]], *, top_k: int) -> list[dict[str, object]]:
    usable = [row for row in rows if int(row["health_ok"]) == 1]
    usable_positive = [row for row in usable if str(row["support_direction"]) == "positive_support"]
    usable_negative = [row for row in usable if str(row["support_direction"]) == "negative_support"]
    top_positive = _positive_rank(rows)[:top_k]
    top_negative = _negative_rank(rows)[:top_k]
    return [
        {"metric": "components_total", "value": len(rows)},
        {"metric": "usable_components", "value": len(usable)},
        {"metric": "usable_positive_support_components", "value": len(usable_positive)},
        {"metric": "usable_negative_support_components", "value": len(usable_negative)},
        {
            "metric": "top_positive_components",
            "value": ",".join(str(row["component_id"]) for row in top_positive),
        },
        {
            "metric": "top_negative_components",
            "value": ",".join(str(row["component_id"]) for row in top_negative),
        },
    ]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rows = _component_rows(args)
    positive_rank = _positive_rank(rows)
    negative_rank = _negative_rank(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dump_csv(args.out_dir / "component_directionality.csv", rows)
    dump_csv(args.out_dir / "positive_direction_rank.csv", positive_rank)
    dump_csv(args.out_dir / "negative_direction_rank.csv", negative_rank)
    dump_csv(args.out_dir / "directionality_summary.csv", _summary_rows(rows, top_k=args.top_k))
    dump_json(
        args.out_dir / "directionality_manifest.json",
        {
            "delta_csv": str(args.delta_csv),
            "target_threshold": args.target_threshold,
            "strong_delta_threshold": args.strong_delta_threshold,
            "min_ablated_format_ok": args.min_ablated_format_ok,
            "max_format_collapse_rate": args.max_format_collapse_rate,
            "top_k": args.top_k,
            "semantics": {
                "delta": "target_score(full generation) - target_score(zero-ablated generation)",
                "pos_mass": "mean(max(delta, 0)); gross support for the target direction without cancellation",
                "neg_mass": "mean(max(-delta, 0)); gross reverse support without cancellation",
                "healthy_pair": "both full and zero-ablated generations pass the local format audit",
                "healthy_pos_to_nonpos_rate": (
                    "Among healthy rows, the fraction where zero-ablation flips target-active "
                    "full generations below the target threshold."
                ),
                "healthy_nonpos_to_pos_rate": (
                    "Among healthy rows, the fraction where zero-ablation flips target-inactive "
                    "full generations above the target threshold."
                ),
                "health_ok": (
                    "Component is usable only if ablated format-ok rate stays above the requested floor "
                    "and collapse rate stays below the requested ceiling."
                ),
            },
        },
    )
    print(
        (
            f"[cecm-directionality] components={len(rows)} usable={sum(int(row['health_ok']) for row in rows)} "
            f"out={args.out_dir}"
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
