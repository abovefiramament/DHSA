from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from screscomp.data import dump_csv, dump_json, load_csv


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select positive and negative IMDb rollout heads from one shared pool.")
    parser.add_argument("--directionality-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--min-ablated-format-ok", type=float, default=0.9)
    parser.add_argument("--max-format-collapse-rate", type=float, default=0.1)
    parser.add_argument("--min-healthy-pair-rate", type=float, default=0.9)
    return parser.parse_args(argv)


def _float(row: dict[str, Any], key: str) -> float:
    try:
        return float(str(row.get(key, "")).strip() or 0.0)
    except Exception:
        return 0.0


def _healthy(row: dict[str, str], args: argparse.Namespace) -> bool:
    return (
        str(row.get("component_type", "")) == "head"
        and _float(row, "mean_ablated_format_ok") >= args.min_ablated_format_ok
        and _float(row, "format_collapse_rate") <= args.max_format_collapse_rate
        and _float(row, "healthy_pair_rate") >= args.min_healthy_pair_rate
        and int(_float(row, "n")) == 300
    )


def _selected_row(row: dict[str, str], *, rank: int, direction: str) -> dict[str, object]:
    metric = "healthy_pos_mass" if direction == "positive" else "healthy_neg_mass"
    head_id = str(row["component_id"])
    layer_text, head_text = head_id.removeprefix("L").split(".attn.h", 1)
    return {
        "rank": rank,
        "head_id": head_id,
        "component_id": head_id,
        "layer_idx": int(layer_text),
        "head_idx": int(head_text),
        "selection_direction": direction,
        "selection_metric": metric,
        "selection_score": row.get(metric, ""),
        "healthy_pos_mass": row.get("healthy_pos_mass", ""),
        "healthy_neg_mass": row.get("healthy_neg_mass", ""),
        "healthy_pos_to_nonpos_rate": row.get("healthy_pos_to_nonpos_rate", ""),
        "healthy_nonpos_to_pos_rate": row.get("healthy_nonpos_to_pos_rate", ""),
        "mean_ablated_format_ok": row.get("mean_ablated_format_ok", ""),
        "healthy_pair_rate": row.get("healthy_pair_rate", ""),
        "format_collapse_rate": row.get("format_collapse_rate", ""),
        "n": row.get("n", ""),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rows = [row for row in load_csv(args.directionality_csv) if _healthy(row, args)]
    if not rows:
        raise SystemExit("[imdb-rollout-head-select] ERROR: No healthy rollout heads are available for selection.")
    positive = sorted(
        rows,
        key=lambda row: (
            _float(row, "healthy_pos_mass"),
            _float(row, "healthy_pos_to_nonpos_rate"),
            _float(row, "healthy_pos_active_mean"),
        ),
        reverse=True,
    )[: args.topk]
    negative = sorted(
        rows,
        key=lambda row: (
            _float(row, "healthy_neg_mass"),
            _float(row, "healthy_nonpos_to_pos_rate"),
            _float(row, "healthy_neg_active_mean"),
        ),
        reverse=True,
    )[: args.topk]
    if len(positive) != args.topk or len(negative) != args.topk:
        raise SystemExit(
            f"[imdb-rollout-head-select] ERROR: Only found {len(positive)} positive and {len(negative)} negative heads "
            f"(wanted {args.topk} each)."
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    positive_rows = [_selected_row(row, rank=index, direction="positive") for index, row in enumerate(positive, start=1)]
    negative_rows = [_selected_row(row, rank=index, direction="negative") for index, row in enumerate(negative, start=1)]
    dump_csv(args.out_dir / "selected_boost_heads.csv", positive_rows)
    dump_csv(args.out_dir / "selected_suppress_heads.csv", negative_rows)
    (args.out_dir / "selected_boost_heads.txt").write_text(
        ",".join(str(row["head_id"]) for row in positive_rows), encoding="utf-8"
    )
    (args.out_dir / "selected_suppress_heads.txt").write_text(
        ",".join(str(row["head_id"]) for row in negative_rows), encoding="utf-8"
    )
    dump_json(
        args.out_dir / "head_selection_manifest.json",
        {
            "directionality_csv": str(args.directionality_csv),
            "shared_pool": True,
            "topk": args.topk,
            "selection": {
                "positive": "healthy_pos_mass",
                "negative": "healthy_neg_mass",
                "ci_filter": False,
            },
            "selected_boost_heads": [row["head_id"] for row in positive_rows],
            "selected_suppress_heads": [row["head_id"] for row in negative_rows],
        },
    )
    print(
        f"[imdb-rollout-head-select] positive={len(positive_rows)} negative={len(negative_rows)} out={args.out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
