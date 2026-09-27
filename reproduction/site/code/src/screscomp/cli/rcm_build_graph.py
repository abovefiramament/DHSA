from __future__ import annotations

import argparse
from pathlib import Path

from screscomp.rcm.graph import (
    build_graph_from_summaries,
    save_axis_relation_rows,
    save_component_relation_rows,
    save_component_rows,
    save_graph,
)


def _parse_axis_map(raw: str, *, value_type=str) -> dict[str, object]:
    output: dict[str, object] = {}
    if not raw:
        return output
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Expected axis=value item: {item}")
        axis, value = item.split("=", 1)
        output[axis.strip()] = value_type(value.strip())
    return output


def _collect_axis_summaries(args: argparse.Namespace) -> dict[str, Path]:
    summaries: dict[str, Path] = {}
    fixed = {
        "source_identity": args.source_summary_csv,
        "form": args.form_summary_csv,
        "commitment": args.commitment_summary_csv,
        "prior_suppression": args.prior_suppression_summary_csv,
    }
    for axis, path in fixed.items():
        if path is not None:
            summaries[axis] = path
    for item in args.axis_summary_csv or []:
        if "=" not in item:
            raise ValueError(f"--axis_summary_csv must look like axis=path: {item}")
        axis, path = item.split("=", 1)
        summaries[axis.strip()] = Path(path.strip())
    if not summaries:
        raise ValueError("Provide at least one axis summary CSV.")
    return summaries


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build an RCM signed component graph from axis scan summaries.")
    p.add_argument("--name", default="rcm_graph")
    p.add_argument("--source_summary_csv", type=Path, default=None)
    p.add_argument("--form_summary_csv", type=Path, default=None)
    p.add_argument("--commitment_summary_csv", type=Path, default=None)
    p.add_argument("--prior_suppression_summary_csv", type=Path, default=None)
    p.add_argument(
        "--axis_summary_csv",
        action="append",
        default=[],
        help="Additional or overriding axis summary in axis=path form. Can be repeated.",
    )
    p.add_argument("--top_up_k", type=int, default=4)
    p.add_argument("--top_down_k", type=int, default=4)
    p.add_argument("--axis_timings", default="", help="Comma-separated axis=timing overrides.")
    p.add_argument("--axis_alphas", default="", help="Comma-separated axis=float-alpha overrides.")
    p.add_argument("--out_graph_json", type=Path, required=True)
    p.add_argument("--out_components_csv", type=Path, default=None)
    p.add_argument("--out_component_relations_csv", type=Path, default=None)
    p.add_argument("--out_axis_relations_csv", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    summaries = _collect_axis_summaries(args)
    graph = build_graph_from_summaries(
        summaries,
        name=args.name,
        top_up_k=args.top_up_k,
        top_down_k=args.top_down_k,
        axis_timings={key: str(value) for key, value in _parse_axis_map(args.axis_timings).items()},
        axis_alphas={key: float(value) for key, value in _parse_axis_map(args.axis_alphas, value_type=float).items()},
    )
    save_graph(graph, args.out_graph_json)
    if args.out_components_csv is not None:
        save_component_rows(graph, args.out_components_csv)
    if args.out_component_relations_csv is not None:
        save_component_relation_rows(graph, args.out_component_relations_csv)
    if args.out_axis_relations_csv is not None:
        save_axis_relation_rows(graph, args.out_axis_relations_csv)
    print(
        f"[rcm-build-graph] axes={','.join(sorted(graph.axes))} components={len(graph.components)} "
        f"out={args.out_graph_json}"
    )


if __name__ == "__main__":
    main()
