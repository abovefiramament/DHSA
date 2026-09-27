from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from screscomp.data import load_csv
from screscomp.rcm.graph import ComponentNode, RCMGraph, load_graph, save_component_rows, save_graph


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Expand an RCM graph from the candidate pool for readout experiments. "
            "This is for prediction/readout ablations, not the default sparse intervention graph."
        )
    )
    p.add_argument("--reference_graph_json", type=Path, required=True)
    p.add_argument("--candidate_edges_csv", type=Path, required=True)
    p.add_argument("--out_graph_json", type=Path, required=True)
    p.add_argument("--out_components_csv", type=Path, default=None)
    p.add_argument("--name", default="rcm_full_candidate_graph")
    p.add_argument("--mode", choices=["all", "top_k_per_axis_sign"], default="top_k_per_axis_sign")
    p.add_argument("--top_k_per_axis_sign", type=int, default=24)
    p.add_argument("--axes", default="", help="Comma-separated axes. Empty uses reference graph axes.")
    p.add_argument("--timings", default="", help="Comma-separated timings. Empty keeps candidate timings.")
    p.add_argument("--weight_mode", choices=["absolute_selection", "rank_score", "unit"], default="absolute_selection")
    return p.parse_args()


def _float(value: Any, default: float = 0.0) -> float:
    if value in ("", None):
        return default
    return float(value)


def _int(value: Any, default: int = 0) -> int:
    if value in ("", None):
        return default
    return int(float(value))


def _weight(row: dict[str, Any], mode: str) -> float:
    if mode == "unit":
        return 1.0
    if mode == "rank_score":
        return max(_float(row.get("rank_score"), 0.0), 0.0)
    return abs(_float(row.get("selection_score"), 0.0))


def _row_to_node(row: dict[str, Any], *, rank: int, weight_mode: str) -> ComponentNode:
    edge_id = str(row["edge_id"])
    metadata = {"edge_id": edge_id, "expanded_graph": True}
    for key in ("rank_score", "utility"):
        if key in row:
            metadata[key] = row[key]
    return ComponentNode(
        axis=str(row["axis"]),
        component_id=str(row["component_id"]),
        layer_idx=_int(row["layer_idx"]),
        component_type=str(row["component_type"]),
        sign=str(row["sign"]),
        weight=_weight(row, weight_mode),
        selection_score=_float(row.get("selection_score"), 0.0),
        selection_metric=str(row.get("selection_metric", "expanded_candidate")),
        timing=str(row["timing"]),
        alpha=_float(row.get("alpha"), 1.0),
        source_path=str(row.get("source_path", "")),
        rank=rank,
        metadata=metadata,
    )


def _parse_set(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}


def main() -> None:
    args = parse_args()
    reference = load_graph(args.reference_graph_json)
    axes = _parse_set(args.axes) or set(reference.axes)
    timings = _parse_set(args.timings)
    rows = [
        row
        for row in load_csv(args.candidate_edges_csv)
        if str(row.get("axis", "")) in axes and (not timings or str(row.get("timing", "")) in timings)
    ]
    if args.mode == "top_k_per_axis_sign":
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[(str(row["axis"]), str(row["sign"]))].append(row)
        selected: list[dict[str, Any]] = []
        for _key, local in sorted(grouped.items()):
            local.sort(key=lambda row: (abs(_float(row.get("selection_score"))), _float(row.get("rank_score"))), reverse=True)
            selected.extend(local[: args.top_k_per_axis_sign])
        rows = selected
    rows.sort(key=lambda row: (str(row["axis"]), str(row["sign"]), -abs(_float(row.get("selection_score"))), str(row["edge_id"])))
    components = [_row_to_node(row, rank=rank, weight_mode=args.weight_mode) for rank, row in enumerate(rows, start=1)]
    graph = RCMGraph(
        name=args.name,
        axes={axis: reference.axes[axis] for axis in sorted({node.axis for node in components})},
        components=components,
        metadata={
            **reference.metadata,
            "expanded_from": reference.name,
            "candidate_edges_csv": str(args.candidate_edges_csv),
            "mode": args.mode,
            "top_k_per_axis_sign": args.top_k_per_axis_sign,
            "weight_mode": args.weight_mode,
            "intended_use": "prediction_readout_ablation",
        },
        version=reference.version,
    )
    save_graph(graph, args.out_graph_json)
    if args.out_components_csv is not None:
        save_component_rows(graph, args.out_components_csv)
    print(
        f"[rcm-expand-graph] components={len(graph.components)} axes={','.join(sorted(graph.axes))} "
        f"out={args.out_graph_json}"
    )


if __name__ == "__main__":
    main()
