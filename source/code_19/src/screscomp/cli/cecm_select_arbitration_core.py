from __future__ import annotations

import argparse
from pathlib import Path

from screscomp.cecm.components import (
    SOURCE_CONTEXT_OVER_PRIOR,
    select_components_by_clauses,
    summarize_generic_component_selection,
)
from screscomp.cecm.specs import EdgeClause, ObjectiveSpec, load_objective_spec
from screscomp.data import dump_csv, dump_json, dump_jsonl, load_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Select CECM components by intersecting objective-spec edge clauses."
    )
    p.add_argument("--core-manifest-csv", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--event",
        type=str,
        default="",
        help="Objective id. Defaults to source_context_over_prior when --spec-json is not provided.",
    )
    p.add_argument(
        "--spec-json",
        type=Path,
        default=None,
        help="Optional objective spec JSON. Component selection always uses spec.component_clauses.",
    )
    p.add_argument("--max-core-index", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_csv(args.core_manifest_csv)
    objective_id = args.event or (SOURCE_CONTEXT_OVER_PRIOR if args.spec_json is None else "")
    spec = load_objective_spec(objective_id=objective_id, spec_json=args.spec_json)
    clauses = tuple(
        EdgeClause(
            name=clause.name,
            axis=clause.axis,
            event=clause.event,
            edge_sign=clause.edge_sign,
            edge_direction=clause.edge_direction,
            max_core_index=(
                args.max_core_index
                if clause.max_core_index is None
                else min(clause.max_core_index, args.max_core_index)
            ),
        )
        for clause in spec.component_clauses
    )
    spec = ObjectiveSpec(
        objective_id=spec.objective_id,
        prompt_key=spec.prompt_key,
        y_plus_keys=spec.y_plus_keys,
        y_minus_keys=spec.y_minus_keys,
        y_minus_fallback_keys=spec.y_minus_fallback_keys,
        component_clauses=clauses,
        continuation_prefix=spec.continuation_prefix,
        val_mod=spec.val_mod,
        description=spec.description,
    )
    selected, audit_edges = select_components_by_clauses(rows, spec=spec)
    if not selected:
        raise SystemExit(
            "No components selected. Check spec.component_clauses and max-core-index."
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dump_csv(args.out_dir / "components.csv", [item.to_row() for item in selected])
    dump_csv(args.out_dir / "selected_edges.csv", audit_edges)
    dump_jsonl(
        args.out_dir / "components.jsonl",
        (
            {
                "component": item.to_row(),
                "clause_edges": [
                    {"clause": clause.to_dict(), "edge": edge_row}
                    for clause, edge_row in item.clause_rows
                ],
            }
            for item in selected
        ),
    )
    dump_jsonl(args.out_dir / "selected_edges.jsonl", audit_edges)
    dump_csv(
        args.out_dir / "component_selection_summary.csv",
        summarize_generic_component_selection(
            total_edges=len(rows),
            selected=selected,
            audit_edges=audit_edges,
            spec=spec,
        ),
    )
    dump_json(
        args.out_dir / "component_selection_manifest.json",
        {
            "core_manifest_csv": str(args.core_manifest_csv),
            "event": spec.objective_id,
            "spec_json": str(args.spec_json) if args.spec_json else "",
            "max_core_index": args.max_core_index,
            "objective_spec": spec.to_dict(),
            "selection_rule": "A component is selected only if it satisfies every component_clause.",
            "selected_components": [item.component_id for item in selected],
        },
    )
    print(
        f"[cecm] selected={len(selected)} components={','.join(item.component_id for item in selected)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
