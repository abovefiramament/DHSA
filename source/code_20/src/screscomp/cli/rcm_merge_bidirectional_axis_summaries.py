from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from screscomp.data import dump_csv, load_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Merge independently discovered forward/reverse component summaries into one signed RCM axis summary. "
            "Forward rows keep positive selection scores; reverse rows are written as negative/down edges."
        )
    )
    p.add_argument("--axis", required=True)
    p.add_argument("--forward_csv", type=Path, required=True)
    p.add_argument("--reverse_csv", type=Path, required=True)
    p.add_argument("--out_csv", type=Path, required=True)
    p.add_argument("--top_forward_k", type=int, default=16)
    p.add_argument("--top_reverse_k", type=int, default=16)
    return p.parse_args()


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value in ("", None):
        return default
    return float(value)


def _int(row: dict[str, Any], key: str, default: int = 10**9) -> int:
    value = row.get(key, default)
    if value in ("", None):
        return default
    return int(float(value))


def _rank(rows: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    usable = [
        row
        for row in rows
        if (row.get("component_id") or (row.get("layer_idx") and row.get("component_type")))
        and _float(row, "selection_score") > 0.0
    ]
    usable.sort(
        key=lambda row: (
            -_float(row, "selection_score"),
            _int(row, "selection_rank"),
            _int(row, "layer_idx"),
            str(row.get("component_type", "")),
        )
    )
    return usable[:limit]


def _signed_row(row: dict[str, str], *, axis: str, sign: str, rank: int) -> dict[str, Any]:
    score = abs(_float(row, "selection_score"))
    if sign == "down":
        score = -score
    return {
        **row,
        "axis": axis,
        "selection_score": score,
        "selection_rank": rank,
        "selection_sign_rank": row.get("selection_rank", rank),
        "signed_edge_sign": sign,
        "signed_edge_source_direction": row.get("direction", ""),
        "selection_rule": f"{row.get('selection_rule', '')}|bidirectional_{sign}_edge",
    }


def main() -> None:
    args = parse_args()
    forward = _rank(load_csv(args.forward_csv), args.top_forward_k)
    reverse = _rank(load_csv(args.reverse_csv), args.top_reverse_k)
    rows: list[dict[str, Any]] = []
    rows.extend(_signed_row(row, axis=args.axis, sign="up", rank=rank) for rank, row in enumerate(forward, start=1))
    reverse_start = len(rows) + 1
    rows.extend(
        _signed_row(row, axis=args.axis, sign="down", rank=reverse_start + offset - 1)
        for offset, row in enumerate(reverse, start=1)
    )
    dump_csv(args.out_csv, rows)
    print(
        f"[rcm-merge-bidirectional-axis] axis={args.axis} up={len(forward)} down={len(reverse)} out={args.out_csv}"
    )


if __name__ == "__main__":
    main()
