from __future__ import annotations

import argparse
from pathlib import Path

from screscomp.data import dump_csv, dump_json, dump_jsonl
from screscomp.rcm.control import build_control_plan
from screscomp.rcm.graph import load_graph


def _parse_axis_operations(raw: str) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    if not raw:
        return output
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Expected axis=operation item: {item}")
        axis, operations = item.split("=", 1)
        output[axis.strip()] = [op.strip() for op in operations.split("+") if op.strip()]
    return output


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a single-axis or joint RCM control plan from a component graph.")
    p.add_argument("--graph_json", type=Path, required=True)
    p.add_argument("--name", default="rcm_control_plan")
    p.add_argument(
        "--axis_operations",
        default="",
        help=(
            "Comma-separated axis=op+op list. Operations: target, anti_target, amplify_up, suppress_up, "
            "amplify_down, suppress_down. Empty means target for every graph axis."
        ),
    )
    p.add_argument("--global_alpha", type=float, default=1.0)
    p.add_argument("--out_plan_json", type=Path, required=True)
    p.add_argument("--out_terms_csv", type=Path, default=None)
    p.add_argument("--out_component_decisions_csv", type=Path, default=None)
    p.add_argument("--out_additions_jsonl", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    graph = load_graph(args.graph_json)
    operations = _parse_axis_operations(args.axis_operations) or None
    plan = build_control_plan(
        graph,
        axis_operations=operations,
        name=args.name,
        global_alpha=args.global_alpha,
    )
    dump_json(args.out_plan_json, plan.to_dict())
    if args.out_terms_csv is not None:
        dump_csv(args.out_terms_csv, [term.to_dict() for term in plan.terms])
    if args.out_component_decisions_csv is not None:
        dump_csv(args.out_component_decisions_csv, plan.component_decision_rows())
    if args.out_additions_jsonl is not None:
        dump_jsonl(args.out_additions_jsonl, plan.addition_specs())
    print(
        f"[rcm-control-plan] graph={graph.name} axes={','.join(sorted(plan.operations))} "
        f"terms={len(plan.terms)} out={args.out_plan_json}"
    )


if __name__ == "__main__":
    main()
