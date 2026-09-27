from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from screscomp.cli.score_ckplug_generation import _method_metrics
from screscomp.data import dump_csv, dump_json, load_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Suggest recursive RCM axis refinements from validation generations. "
            "This does not choose a final graph; it reports which non-target readout modes "
            "dominate validation failures and should be considered as new attractor nodes."
        )
    )
    p.add_argument("--generations_jsonl", type=Path, required=True)
    p.add_argument("--method", type=str, default="")
    p.add_argument("--readout_field", type=str, default="source_fine_outcome")
    p.add_argument("--target_value", type=str, default="context_only")
    p.add_argument("--min_rate", type=float, default=0.03)
    p.add_argument("--max_candidates", type=int, default=5)
    p.add_argument("--out_modes_csv", type=Path, required=True)
    p.add_argument("--out_suggestions_json", type=Path, required=True)
    return p.parse_args()


def _select_method(rows: list[dict[str, Any]], method: str) -> tuple[str, list[dict[str, Any]]]:
    methods = sorted({str(row.get("method", "")) for row in rows})
    if method:
        selected = [row for row in rows if str(row.get("method", "")) == method]
        if not selected:
            raise ValueError(f"Method {method!r} not found. Available methods: {methods}")
        return method, selected
    if len(methods) == 1:
        return methods[0], rows
    raise ValueError(f"Multiple methods found; pass --method. Available methods: {methods}")


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = []
    for row in rows:
        value = row.get(key, None)
        if value is None or value == "":
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return sum(values) / len(values) if values else 0.0


def _bool_rate(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if bool(row.get(key))) / len(rows)


def _mode_rows(
    *,
    rows: list[dict[str, Any]],
    readout_field: str,
    target_value: str,
    min_rate: float,
) -> list[dict[str, Any]]:
    total = len(rows)
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_mode[str(row.get(readout_field, ""))].append(row)

    output: list[dict[str, Any]] = []
    for mode, group in sorted(by_mode.items(), key=lambda item: (-len(item[1]), item[0])):
        rate = len(group) / total if total else 0.0
        is_target = mode == target_value
        metrics = _method_metrics(group)
        output.append(
            {
                "mode": mode,
                "n": len(group),
                "rate": rate,
                "is_target": is_target,
                "candidate_split": (not is_target) and rate >= min_rate,
                "context_only_rate": metrics.get("context_only_rate", 0.0),
                "prior_only_rate": metrics.get("prior_only_rate", 0.0),
                "both_rate": metrics.get("both_rate", 0.0),
                "neither_rate": metrics.get("neither_rate", 0.0),
                "cf_em_rate": metrics.get("cf_em_rate", 0.0),
                "mean_output_chars": metrics.get("mean_output_chars", 0.0),
                "mean_route_source_chars": _mean(group, "route_source_output_chars"),
                "mean_route_fallback_chars": _mean(group, "route_fallback_output_chars"),
                "short_output_rate": _bool_rate(group, "short_output"),
                "first_answer_cf_em_rate": _bool_rate(group, "first_answer_cf_em"),
            }
        )
    return output


def _stage_hints(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = [int(row.get("output_chars", len(str(row.get("prediction", ""))))) for row in rows]
    if not lengths:
        return {}
    lengths_sorted = sorted(lengths)
    p50 = lengths_sorted[len(lengths_sorted) // 2]
    p90 = lengths_sorted[int(0.9 * (len(lengths_sorted) - 1))]
    counter = Counter(str(row.get("route_chosen", "")) for row in rows if row.get("route_chosen", ""))
    return {
        "mean_output_chars": sum(lengths) / len(lengths),
        "median_output_chars": p50,
        "p90_output_chars": p90,
        "route_chosen_counts": dict(counter),
    }


def _suggestions(
    *,
    mode_rows: list[dict[str, Any]],
    target_value: str,
    max_candidates: int,
) -> list[dict[str, Any]]:
    candidates = [row for row in mode_rows if row["candidate_split"]]
    candidates.sort(key=lambda row: (-float(row["rate"]), str(row["mode"])))
    output = []
    for row in candidates[:max_candidates]:
        mode = str(row["mode"])
        timing_hint = _timing_hint_for_mode(mode)
        output.append(
            {
                "new_node": mode,
                "proposed_edge": f"{target_value}<->{mode}",
                "reason": f"{mode} is a frequent non-target validation mode (rate={float(row['rate']):.3f}).",
                "suggested_timing": timing_hint,
                "next_action": (
                    "Construct a contrastive edge for this node, scan components on discovery, "
                    "then validate whether adding the edge improves the target metric under constraints."
                ),
            }
        )
    return output


def _timing_hint_for_mode(mode: str) -> str:
    if mode == "prior_only":
        return "first_decode"
    if mode == "mixed":
        return "first_2_decode"
    if mode == "neither_verbose":
        return "prefill"
    if mode == "neither_short":
        return "first_decode"
    if mode in {"both", "neither"}:
        return "first_decode for source edge; decode/prefill for any form-related sub-edge"
    return "first_decode"


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.generations_jsonl)
    method, selected = _select_method(rows, args.method)
    mode_rows = _mode_rows(
        rows=selected,
        readout_field=args.readout_field,
        target_value=args.target_value,
        min_rate=args.min_rate,
    )
    suggestions = _suggestions(
        mode_rows=mode_rows,
        target_value=args.target_value,
        max_candidates=args.max_candidates,
    )
    payload = {
        "generations_jsonl": str(args.generations_jsonl),
        "method": method,
        "readout_field": args.readout_field,
        "target_value": args.target_value,
        "n": len(selected),
        "stage_hints": _stage_hints(selected),
        "suggestions": suggestions,
    }
    dump_csv(args.out_modes_csv, mode_rows)
    dump_json(args.out_suggestions_json, payload)
    print(
        f"[suggest-rcm-refinement] method={method} rows={len(selected)} "
        f"candidate_splits={len(suggestions)}"
    )


if __name__ == "__main__":
    main()
