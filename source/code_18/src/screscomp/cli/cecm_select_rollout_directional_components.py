from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from screscomp.data import dump_csv, dump_json, load_csv


GROUPS = (
    ("mlp_positive", "mlp", "positive"),
    ("mlp_negative", "mlp", "negative"),
    ("attn_positive", "attn", "positive"),
    ("attn_negative", "attn", "negative"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Select rollout components by single-direction gross support mass with health filtering. "
            "Positive groups rank by healthy_pos_mass; negative groups rank by healthy_neg_mass."
        )
    )
    p.add_argument("--component-directionality-csv", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--event", type=str, default="imdb_positive_sentiment")
    p.add_argument("--topk-mlp", type=int, default=4)
    p.add_argument("--topk-attn", type=int, default=4)
    p.add_argument("--min-ablated-format-ok", type=float, default=0.9)
    p.add_argument("--max-format-collapse-rate", type=float, default=0.1)
    p.add_argument("--min-healthy-pair-rate", type=float, default=0.9)
    p.add_argument("--min-healthy-n", type=int, default=1)
    p.add_argument(
        "--require-direction-match",
        action="store_true",
        help="Require positive groups to have support_direction=positive_support and negative groups to match negative_support.",
    )
    p.add_argument(
        "--disallow-cross-group-overlap",
        action="store_true",
        help="Do not reuse the same component within the same component type across positive/negative groups.",
    )
    return p.parse_args(argv)


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        raw = str(row.get(key, "")).strip()
        return float(raw) if raw else default
    except Exception:
        return default


def _int(row: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        raw = str(row.get(key, "")).strip()
        return int(float(raw)) if raw else default
    except Exception:
        return default


def _health_ok(
    row: dict[str, str],
    *,
    min_ablated_format_ok: float,
    max_format_collapse_rate: float,
    min_healthy_pair_rate: float,
    min_healthy_n: int,
) -> bool:
    if _float(row, "mean_ablated_format_ok") < min_ablated_format_ok:
        return False
    if _float(row, "format_collapse_rate") > max_format_collapse_rate:
        return False
    if _float(row, "healthy_pair_rate") < min_healthy_pair_rate:
        return False
    if _int(row, "healthy_n") < min_healthy_n:
        return False
    return True


def _direction_metric(direction: str) -> tuple[str, str, str]:
    if direction == "positive":
        return "healthy_pos_mass", "healthy_pos_to_nonpos_rate", "healthy_pos_active_mean"
    return "healthy_neg_mass", "healthy_nonpos_to_pos_rate", "healthy_neg_active_mean"


def _direction_matches(row: dict[str, str], direction: str) -> bool:
    support = str(row.get("support_direction", "")).strip()
    if direction == "positive":
        return support == "positive_support"
    return support == "negative_support"


def _component_row(
    row: dict[str, str],
    *,
    rank: int,
    group_name: str,
    direction: str,
    metric_name: str,
) -> dict[str, object]:
    return {
        "component_id": row["component_id"],
        "layer_idx": row["layer_idx"],
        "component_type": row["component_type"],
        "rank": rank,
        "group_name": group_name,
        "selection_direction": direction,
        "selection_metric": metric_name,
        "selection_score": row.get(metric_name, ""),
        "support_direction": row.get("support_direction", ""),
        "usable_support_direction": row.get("usable_support_direction", ""),
        "healthy_pos_mass": row.get("healthy_pos_mass", ""),
        "healthy_neg_mass": row.get("healthy_neg_mass", ""),
        "healthy_mass_gap": row.get("healthy_mass_gap", ""),
        "healthy_pos_to_nonpos_rate": row.get("healthy_pos_to_nonpos_rate", ""),
        "healthy_nonpos_to_pos_rate": row.get("healthy_nonpos_to_pos_rate", ""),
        "healthy_pos_active_mean": row.get("healthy_pos_active_mean", ""),
        "healthy_neg_active_mean": row.get("healthy_neg_active_mean", ""),
        "positive_rate": row.get("positive_rate", ""),
        "negative_rate": row.get("negative_rate", ""),
        "healthy_positive_rate": row.get("healthy_positive_rate", ""),
        "healthy_negative_rate": row.get("healthy_negative_rate", ""),
        "mean_ablated_format_ok": row.get("mean_ablated_format_ok", ""),
        "healthy_pair_rate": row.get("healthy_pair_rate", ""),
        "format_collapse_rate": row.get("format_collapse_rate", ""),
    }


def _rank_rows(rows: list[dict[str, str]], *, direction: str) -> list[dict[str, str]]:
    metric_name, transition_name, active_name = _direction_metric(direction)
    return sorted(
        rows,
        key=lambda row: (
            _float(row, metric_name),
            _float(row, transition_name),
            _float(row, active_name),
            abs(_float(row, "healthy_mass_gap")),
            str(row.get("component_id", "")),
        ),
        reverse=True,
    )


def _select_group(
    rows: list[dict[str, str]],
    *,
    component_type: str,
    direction: str,
    topk: int,
    min_ablated_format_ok: float,
    max_format_collapse_rate: float,
    min_healthy_pair_rate: float,
    min_healthy_n: int,
    require_direction_match: bool,
    exclude_components: set[str],
) -> list[dict[str, str]]:
    candidates = [
        row
        for row in rows
        if str(row.get("component_type", "")).strip() == component_type
        and row.get("component_id")
        and row["component_id"] not in exclude_components
        and _health_ok(
            row,
            min_ablated_format_ok=min_ablated_format_ok,
            max_format_collapse_rate=max_format_collapse_rate,
            min_healthy_pair_rate=min_healthy_pair_rate,
            min_healthy_n=min_healthy_n,
        )
    ]
    if require_direction_match:
        candidates = [row for row in candidates if _direction_matches(row, direction)]
    ranked = _rank_rows(candidates, direction=direction)
    return ranked[:topk] if topk > 0 else []


def _summary_rows(selected: dict[str, list[dict[str, str]]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen_by_type: dict[str, set[str]] = {"mlp": set(), "attn": set()}
    overlaps: dict[str, set[str]] = {"mlp": set(), "attn": set()}
    for group_name, rows_for_group in selected.items():
        rows.append({"metric": f"{group_name}_count", "value": len(rows_for_group)})
        for row in rows_for_group:
            component_type = str(row.get("component_type", "")).strip()
            component_id = str(row.get("component_id", "")).strip()
            if component_id in seen_by_type.setdefault(component_type, set()):
                overlaps.setdefault(component_type, set()).add(component_id)
            seen_by_type[component_type].add(component_id)
    rows.append({"metric": "mlp_overlap_components", "value": ",".join(sorted(overlaps.get("mlp", set())))})
    rows.append({"metric": "attn_overlap_components", "value": ",".join(sorted(overlaps.get("attn", set())))})
    return rows


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rows = load_csv(args.component_directionality_csv)
    if not rows:
        raise SystemExit(f"No rows found: {args.component_directionality_csv}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected: dict[str, list[dict[str, str]]] = {}
    used_by_type: dict[str, set[str]] = {"mlp": set(), "attn": set()}

    for group_name, component_type, direction in GROUPS:
        topk = args.topk_mlp if component_type == "mlp" else args.topk_attn
        excluded = used_by_type[component_type] if args.disallow_cross_group_overlap else set()
        group_rows = _select_group(
            rows,
            component_type=component_type,
            direction=direction,
            topk=topk,
            min_ablated_format_ok=args.min_ablated_format_ok,
            max_format_collapse_rate=args.max_format_collapse_rate,
            min_healthy_pair_rate=args.min_healthy_pair_rate,
            min_healthy_n=args.min_healthy_n,
            require_direction_match=args.require_direction_match,
            exclude_components=excluded,
        )
        selected[group_name] = group_rows
        if args.disallow_cross_group_overlap:
            used_by_type[component_type].update(str(row["component_id"]) for row in group_rows)

    all_rows: list[dict[str, object]] = []
    for group_name, component_type, direction in GROUPS:
        metric_name, _transition_name, _active_name = _direction_metric(direction)
        group_rows = [
            _component_row(row, rank=rank, group_name=group_name, direction=direction, metric_name=metric_name)
            for rank, row in enumerate(selected[group_name], start=1)
        ]
        dump_csv(args.out_dir / f"{group_name}_components.csv", group_rows)
        all_rows.extend(group_rows)

    positive_rows = [
        row for row in all_rows if str(row.get("selection_direction", "")) == "positive"
    ]
    negative_rows = [
        row for row in all_rows if str(row.get("selection_direction", "")) == "negative"
    ]
    dump_csv(args.out_dir / "positive_components.csv", positive_rows)
    dump_csv(args.out_dir / "negative_components.csv", negative_rows)
    dump_csv(args.out_dir / "all_selected_components.csv", all_rows)
    dump_csv(args.out_dir / "selection_summary.csv", _summary_rows(selected))
    dump_json(
        args.out_dir / "selection_manifest.json",
        {
            "event": args.event,
            "component_directionality_csv": str(args.component_directionality_csv),
            "topk_mlp": args.topk_mlp,
            "topk_attn": args.topk_attn,
            "min_ablated_format_ok": args.min_ablated_format_ok,
            "max_format_collapse_rate": args.max_format_collapse_rate,
            "min_healthy_pair_rate": args.min_healthy_pair_rate,
            "min_healthy_n": args.min_healthy_n,
            "require_direction_match": bool(args.require_direction_match),
            "disallow_cross_group_overlap": bool(args.disallow_cross_group_overlap),
            "selected_groups": {
                group_name: [str(row["component_id"]) for row in group_rows]
                for group_name, group_rows in selected.items()
            },
            "selection_semantics": {
                "positive": "rank by healthy_pos_mass with health audit; larger means stronger gross support for target-active behavior",
                "negative": "rank by healthy_neg_mass with health audit; larger means stronger gross support for reverse-direction behavior",
                "health": "require mean_ablated_format_ok, healthy_pair_rate, and format_collapse_rate to stay within thresholds",
            },
        },
    )
    print(
        (
            f"[cecm-directional-select] mlp_positive={len(selected['mlp_positive'])} "
            f"mlp_negative={len(selected['mlp_negative'])} "
            f"attn_positive={len(selected['attn_positive'])} "
            f"attn_negative={len(selected['attn_negative'])} "
            f"out={args.out_dir}"
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
