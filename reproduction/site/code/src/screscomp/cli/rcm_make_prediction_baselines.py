from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

from screscomp.data import dump_csv, load_csv
from screscomp.rcm.graph import ComponentNode, RCMGraph, load_graph, save_component_rows, save_graph


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build graph baselines for RCM prediction tables. The baselines preserve the reference "
            "graph's axis/sign/timing budget and replace only the component selection rule."
        )
    )
    p.add_argument("--reference_graph_json", type=Path, required=True)
    p.add_argument("--candidate_edges_csv", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--baselines", default="random,norm_ranked,attribution_ranked")
    p.add_argument("--random_replicates", type=int, default=5)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument(
        "--match_fields",
        default="axis,sign,timing,layer_bin,component_type",
        help=(
            "Comma-separated fields used to match baseline graph budget. Default preserves "
            "axis/sign/timing/layer-bin/component-type counts from the reference graph."
        ),
    )
    p.add_argument("--layer_bin_size", type=int, default=4)
    p.add_argument(
        "--exclude_reference_edges",
        action="store_true",
        help="Do not allow random/effect baselines to reuse exact reference edge ids.",
    )
    return p.parse_args()


def _float(value: Any, default: float = 0.0) -> float:
    if value in ("", None):
        return default
    return float(value)


def _int(value: Any, default: int = 0) -> int:
    if value in ("", None):
        return default
    return int(float(value))


def _row_to_node(row: dict[str, Any], *, weight: float, baseline: str, rank: int) -> ComponentNode:
    metadata = {"edge_id": row["edge_id"], "baseline": baseline}
    for key in ("rank_score", "utility"):
        if key in row:
            metadata[key] = row[key]
    return ComponentNode(
        axis=str(row["axis"]),
        component_id=str(row["component_id"]),
        layer_idx=_int(row["layer_idx"]),
        component_type=str(row["component_type"]),
        sign=str(row["sign"]),
        weight=weight,
        selection_score=_float(row.get("selection_score")),
        selection_metric=str(row.get("selection_metric", baseline)),
        timing=str(row["timing"]),
        alpha=_float(row.get("alpha"), 1.0),
        source_path=str(row.get("source_path", "")),
        rank=rank,
        metadata=metadata,
    )


def _layer_bin(layer_idx: int, bin_size: int) -> str:
    if bin_size <= 1:
        return str(layer_idx)
    start = (layer_idx // bin_size) * bin_size
    return f"{start}-{start + bin_size - 1}"


def _template_from_node(node: ComponentNode, fields: list[str], *, layer_bin_size: int) -> tuple[str, ...]:
    values = {
        "axis": node.axis,
        "sign": node.sign,
        "timing": node.timing,
        "layer_idx": str(node.layer_idx),
        "layer_bin": _layer_bin(node.layer_idx, layer_bin_size),
        "component_type": node.component_type,
    }
    return tuple(values[field] for field in fields)


def _template_from_row(row: dict[str, Any], fields: list[str], *, layer_bin_size: int) -> tuple[str, ...]:
    values = {field: str(row[field]) for field in row}
    values["layer_bin"] = _layer_bin(_int(row.get("layer_idx")), layer_bin_size)
    return tuple(values[field] for field in fields)


def _template_counts(graph: RCMGraph, fields: list[str], *, layer_bin_size: int) -> Counter[tuple[str, ...]]:
    return Counter(_template_from_node(node, fields, layer_bin_size=layer_bin_size) for node in graph.components)


def _candidate_pool(
    rows: list[dict[str, Any]],
    *,
    reference_edge_ids: set[str],
    exclude_reference_edges: bool,
    match_fields: list[str],
    layer_bin_size: int,
) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    by_template: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        edge_id = str(row.get("edge_id", ""))
        if not edge_id:
            continue
        if exclude_reference_edges and edge_id in reference_edge_ids:
            continue
        by_template[_template_from_row(row, match_fields, layer_bin_size=layer_bin_size)].append(row)
    return by_template


def _reference_weight_by_template(graph: RCMGraph, fields: list[str], *, layer_bin_size: int) -> dict[tuple[str, ...], float]:
    values: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for node in graph.components:
        values[_template_from_node(node, fields, layer_bin_size=layer_bin_size)].append(abs(float(node.weight)))
    return {key: sum(local) / len(local) for key, local in values.items()}


def _take_ranked(
    pool: dict[tuple[str, ...], list[dict[str, Any]]],
    template_counts: Counter[tuple[str, ...]],
    *,
    key_name: str,
    reverse: bool = True,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for template, count in sorted(template_counts.items()):
        local = sorted(
            pool.get(template, []),
            key=lambda row: (_float(row.get(key_name)), str(row.get("edge_id", ""))),
            reverse=reverse,
        )
        selected.extend(local[:count])
    return selected


def _take_effect_ranked(
    pool: dict[tuple[str, ...], list[dict[str, Any]]],
    template_counts: Counter[tuple[str, ...]],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for template, count in sorted(template_counts.items()):
        local = sorted(
            pool.get(template, []),
            key=lambda row: (abs(_float(row.get("selection_score"))), str(row.get("edge_id", ""))),
            reverse=True,
        )
        selected.extend(local[:count])
    return selected


def _norm_rank_value(row: dict[str, Any]) -> float:
    for key in ("direction_norm", "delta_norm", "activation_norm", "mean_direction_norm"):
        if key in row and row.get(key) not in ("", None):
            return abs(_float(row.get(key)))
    return abs(_float(row.get("selection_score")))


def _take_norm_ranked(
    pool: dict[tuple[str, ...], list[dict[str, Any]]],
    template_counts: Counter[tuple[str, ...]],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for template, count in sorted(template_counts.items()):
        local = sorted(
            pool.get(template, []),
            key=lambda row: (_norm_rank_value(row), str(row.get("edge_id", ""))),
            reverse=True,
        )
        selected.extend(local[:count])
    return selected


def _take_random(
    pool: dict[tuple[str, ...], list[dict[str, Any]]],
    template_counts: Counter[tuple[str, ...]],
    *,
    rng: random.Random,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for template, count in sorted(template_counts.items()):
        local = list(pool.get(template, []))
        rng.shuffle(local)
        selected.extend(local[:count])
    return selected


def _graph_from_rows(
    reference: RCMGraph,
    rows: list[dict[str, Any]],
    *,
    name: str,
    baseline: str,
    weight_by_template: dict[tuple[str, ...], float],
    match_fields: list[str],
    layer_bin_size: int,
) -> RCMGraph:
    components = [
        _row_to_node(
            row,
            weight=weight_by_template.get(_template_from_row(row, match_fields, layer_bin_size=layer_bin_size), 1.0),
            baseline=baseline,
            rank=rank,
        )
        for rank, row in enumerate(rows, start=1)
    ]
    return RCMGraph(
        name=name,
        axes={axis: reference.axes[axis] for axis in sorted({node.axis for node in components})},
        components=components,
        metadata={
            **reference.metadata,
            "baseline": baseline,
            "reference_graph": reference.name,
            "selection_rule": baseline,
            "preserves_axis_sign_timing_budget": True,
            "match_fields": ",".join(match_fields),
            "layer_bin_size": layer_bin_size,
        },
        version=reference.version,
    )


def _write_graph(graph: RCMGraph, out_dir: Path) -> dict[str, Any]:
    graph_path = out_dir / f"{graph.name}.json"
    components_path = out_dir / f"{graph.name}_components.csv"
    save_graph(graph, graph_path)
    save_component_rows(graph, components_path)
    return {
        "baseline": graph.metadata.get("baseline", graph.name),
        "graph_name": graph.name,
        "graph_json": str(graph_path),
        "components_csv": str(components_path),
        "edges": len(graph.components),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    reference = load_graph(args.reference_graph_json)
    match_fields = [field.strip() for field in args.match_fields.split(",") if field.strip()]
    reference_edge_ids = {str(node.metadata.get("edge_id", "")) for node in reference.components}
    template_counts = _template_counts(reference, match_fields, layer_bin_size=args.layer_bin_size)
    pool = _candidate_pool(
        load_csv(args.candidate_edges_csv),
        reference_edge_ids=reference_edge_ids,
        exclude_reference_edges=args.exclude_reference_edges,
        match_fields=match_fields,
        layer_bin_size=args.layer_bin_size,
    )
    weight_by_template = _reference_weight_by_template(reference, match_fields, layer_bin_size=args.layer_bin_size)
    manifest: list[dict[str, Any]] = []
    baselines = [item.strip() for item in args.baselines.split(",") if item.strip()]

    for baseline in baselines:
        if baseline == "random":
            for replicate in range(args.random_replicates):
                rows = _take_random(pool, template_counts, rng=random.Random(args.seed + replicate))
                graph = _graph_from_rows(
                    reference,
                    rows,
                    name=f"baseline_random_r{replicate + 1}",
                    baseline="random",
                    weight_by_template=weight_by_template,
                    match_fields=match_fields,
                    layer_bin_size=args.layer_bin_size,
                )
                manifest.append({**_write_graph(graph, args.out_dir), "replicate": replicate + 1})
        elif baseline in {"effect_ranked", "norm_ranked"}:
            rows = _take_norm_ranked(pool, template_counts) if baseline == "norm_ranked" else _take_effect_ranked(pool, template_counts)
            graph = _graph_from_rows(
                reference,
                rows,
                name=f"baseline_{baseline}",
                baseline=baseline,
                weight_by_template=weight_by_template,
                match_fields=match_fields,
                layer_bin_size=args.layer_bin_size,
            )
            manifest.append({**_write_graph(graph, args.out_dir), "replicate": ""})
        elif baseline == "attribution_ranked":
            rows = _take_ranked(pool, template_counts, key_name="utility", reverse=True)
            graph = _graph_from_rows(
                reference,
                rows,
                name="baseline_attribution_ranked",
                baseline="attribution_ranked",
                weight_by_template=weight_by_template,
                match_fields=match_fields,
                layer_bin_size=args.layer_bin_size,
            )
            manifest.append({**_write_graph(graph, args.out_dir), "replicate": ""})
        elif baseline == "reference_copy":
            graph = replace(reference, name="baseline_reference_copy", metadata={**reference.metadata, "baseline": "reference_copy"})
            manifest.append({**_write_graph(graph, args.out_dir), "replicate": ""})
        else:
            raise ValueError(f"Unknown baseline: {baseline}")

    dump_csv(args.out_dir / "baseline_graph_manifest.csv", manifest)
    print(f"[rcm-make-prediction-baselines] graphs={len(manifest)} out={args.out_dir}")


if __name__ == "__main__":
    main()
