from __future__ import annotations

import argparse
from pathlib import Path

from screscomp.data import dump_csv, load_csv
from screscomp.rcm.defaults import (
    CLEAN_AXIS_ALPHAS,
    CLEAN_AXIS_TIMINGS,
    CLEAN_AXIS_WEIGHTS,
    CLEAN_SELECTOR_DEFAULTS,
    objective_preset,
)
from screscomp.rcm.graph import save_axis_relation_rows, save_component_relation_rows, save_component_rows, save_graph
from screscomp.rcm.objective import apply_candidate_validation_scores, load_pair_adjustments
from screscomp.rcm.select import SelectionConfig, load_candidate_edges_from_summaries, select_edges_greedy


def _parse_axis_path_items(items: list[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected axis=path item: {item}")
        axis, path = item.split("=", 1)
        output[axis.strip()] = Path(path.strip())
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
    summaries.update(_parse_axis_path_items(args.axis_summary_csv or []))
    if not summaries:
        raise ValueError("Provide at least one axis summary CSV.")
    return summaries


def _parse_axis_float_map(raw: str) -> dict[str, float]:
    output: dict[str, float] = {}
    if not raw:
        return output
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Expected key=value item: {item}")
        key, value = item.split("=", 1)
        output[key.strip()] = float(value.strip())
    return output


def _parse_float_map(raw: str) -> dict[str, float]:
    return _parse_axis_float_map(raw)


def _parse_axis_timing_map(raw: str) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    if not raw:
        return output
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Expected axis=t1+t2 item: {item}")
        axis, timings = item.split("=", 1)
        output[axis.strip()] = [timing.strip() for timing in timings.split("+") if timing.strip()]
    return output


def _merge_float_defaults(defaults: dict[str, float], raw: str) -> dict[str, float]:
    output = dict(defaults)
    output.update(_parse_axis_float_map(raw))
    return output


def _merge_timing_defaults(defaults: dict[str, list[str]], raw: str) -> dict[str, list[str]]:
    output = {axis: list(timings) for axis, timings in defaults.items()}
    output.update(_parse_axis_timing_map(raw))
    return output


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Select a competition-aware RCM graph from candidate component-axis-sign-timing edges. "
            "Rank evidence narrows the candidate pool; budget, conflict rules, and local search choose the graph."
        )
    )
    p.add_argument("--name", default="rcm_selected_graph")
    p.add_argument(
        "--protocol",
        default="clean",
        choices=["clean", "manual"],
        help="clean uses fixed main-paper defaults; manual uses CLI numeric defaults and explicit overrides.",
    )
    p.add_argument("--source_summary_csv", type=Path, default=None)
    p.add_argument("--form_summary_csv", type=Path, default=None)
    p.add_argument("--commitment_summary_csv", type=Path, default=None)
    p.add_argument("--prior_suppression_summary_csv", type=Path, default=None)
    p.add_argument("--axis_summary_csv", action="append", default=[], help="Additional axis=path summary CSV.")
    p.add_argument(
        "--axis_candidate_timings",
        default="",
        help="Comma-separated axis=t1+t2 timing candidates. Empty uses each axis default timing.",
    )
    p.add_argument("--axis_alphas", default="", help="Comma-separated axis=float alpha overrides.")
    p.add_argument("--axis_weights", default="", help="Comma-separated axis=float objective weights.")
    p.add_argument(
        "--penalty_field_weights",
        default="",
        help="Comma-separated numeric-field=float penalty weights, e.g. mean_output_chars=0.001.",
    )
    p.add_argument("--budget", type=int, default=None)
    p.add_argument("--per_axis_quota", type=int, default=None)
    p.add_argument("--max_candidates_per_axis_sign", type=int, default=None)
    p.add_argument("--rrf_k", type=float, default=None)
    p.add_argument("--rank_weight", type=float, default=None)
    p.add_argument("--effect_weight", type=float, default=None)
    p.add_argument("--edge_cost", type=float, default=None)
    p.add_argument("--same_component_penalty", type=float, default=None)
    p.add_argument("--mixed_timing_penalty", type=float, default=None)
    p.add_argument("--local_search_rounds", type=int, default=None)
    p.add_argument("--allow_multiple_timings_per_component_axis_sign", action="store_true")
    p.add_argument(
        "--candidate_eval_csv",
        type=Path,
        default=None,
        help=(
            "Optional validation evaluation rows keyed by edge_id or axis/component_id/sign/timing. "
            "When provided, --objective_terms computes validation-J edge utility."
        ),
    )
    p.add_argument(
        "--candidate_eval_operation",
        default="target",
        help="When candidate eval contains multiple operations per edge, use this operation for graph selection.",
    )
    p.add_argument(
        "--objective_terms",
        default="",
        help="Comma-separated metric=weight terms for candidate validation-J, e.g. context_only=1,orig_hit=-1.",
    )
    p.add_argument(
        "--objective_preset",
        default="clean_composite",
        help="Objective preset used with --candidate_eval_csv when --objective_terms is empty.",
    )
    p.add_argument(
        "--rank_prior_weight",
        type=float,
        default=0.0,
        help="Blend weight for rank utility when candidate validation rows are available.",
    )
    p.add_argument(
        "--pairwise_eval_csv",
        type=Path,
        default=None,
        help="Optional pairwise rows with edge_id_a/edge_id_b and synergy/conflict metrics.",
    )
    p.add_argument(
        "--pairwise_terms",
        default="",
        help="Comma-separated metric=weight terms for pairwise set adjustment, e.g. synergy=1,conflict=-1.",
    )
    p.add_argument(
        "--pairwise_preset",
        default="clean_pairwise",
        help="Pairwise objective preset used with --pairwise_eval_csv when --pairwise_terms is empty.",
    )
    p.add_argument("--out_graph_json", type=Path, required=True)
    p.add_argument("--out_candidates_csv", type=Path, default=None)
    p.add_argument("--out_selected_edges_csv", type=Path, default=None)
    p.add_argument("--out_rejected_edges_csv", type=Path, default=None)
    p.add_argument("--out_trace_csv", type=Path, default=None)
    p.add_argument("--out_components_csv", type=Path, default=None)
    p.add_argument("--out_component_relations_csv", type=Path, default=None)
    p.add_argument("--out_axis_relations_csv", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    selector_defaults = CLEAN_SELECTOR_DEFAULTS if args.protocol == "clean" else {}
    config = SelectionConfig(
        budget=args.budget if args.budget is not None else int(selector_defaults.get("budget", 16)),
        per_axis_quota=args.per_axis_quota
        if args.per_axis_quota is not None
        else int(selector_defaults.get("per_axis_quota", 8)),
        max_candidates_per_axis_sign=args.max_candidates_per_axis_sign
        if args.max_candidates_per_axis_sign is not None
        else int(selector_defaults.get("max_candidates_per_axis_sign", 24)),
        rrf_k=args.rrf_k if args.rrf_k is not None else float(selector_defaults.get("rrf_k", 60.0)),
        rank_weight=args.rank_weight
        if args.rank_weight is not None
        else float(selector_defaults.get("rank_weight", 1.0)),
        effect_weight=args.effect_weight
        if args.effect_weight is not None
        else float(selector_defaults.get("effect_weight", 0.0)),
        edge_cost=args.edge_cost if args.edge_cost is not None else float(selector_defaults.get("edge_cost", 0.0)),
        same_component_penalty=args.same_component_penalty
        if args.same_component_penalty is not None
        else float(selector_defaults.get("same_component_penalty", 0.03)),
        mixed_timing_penalty=args.mixed_timing_penalty
        if args.mixed_timing_penalty is not None
        else float(selector_defaults.get("mixed_timing_penalty", 0.01)),
        one_timing_per_component_axis_sign=not args.allow_multiple_timings_per_component_axis_sign,
        local_search_rounds=args.local_search_rounds
        if args.local_search_rounds is not None
        else int(selector_defaults.get("local_search_rounds", 1)),
    )
    default_timings = CLEAN_AXIS_TIMINGS if args.protocol == "clean" else {}
    default_alphas = CLEAN_AXIS_ALPHAS if args.protocol == "clean" else {}
    default_axis_weights = CLEAN_AXIS_WEIGHTS if args.protocol == "clean" else {}
    candidates = load_candidate_edges_from_summaries(
        _collect_axis_summaries(args),
        axis_candidate_timings=_merge_timing_defaults(default_timings, args.axis_candidate_timings),
        axis_alphas=_merge_float_defaults(default_alphas, args.axis_alphas),
        axis_weights=_merge_float_defaults(default_axis_weights, args.axis_weights),
        penalty_field_weights=_parse_axis_float_map(args.penalty_field_weights),
        config=config,
    )
    if args.candidate_eval_csv is not None:
        objective_terms = _parse_float_map(args.objective_terms) or objective_preset(args.objective_preset)
        if not objective_terms:
            raise ValueError("--candidate_eval_csv requires --objective_terms.")
        candidates = apply_candidate_validation_scores(
            candidates,
            load_csv(args.candidate_eval_csv),
            objective_terms=objective_terms,
            rank_prior_weight=args.rank_prior_weight,
            operation=args.candidate_eval_operation,
        )
    pair_adjustments = None
    if args.pairwise_eval_csv is not None:
        pairwise_terms = _parse_float_map(args.pairwise_terms) or objective_preset(args.pairwise_preset)
        if not pairwise_terms:
            raise ValueError("--pairwise_eval_csv requires --pairwise_terms.")
        pair_adjustments = load_pair_adjustments(
            load_csv(args.pairwise_eval_csv),
            objective_terms=pairwise_terms,
        )
    result = select_edges_greedy(candidates, name=args.name, config=config, pair_adjustments=pair_adjustments)
    save_graph(result.graph, args.out_graph_json)
    if args.out_candidates_csv is not None:
        dump_csv(args.out_candidates_csv, [edge.to_dict() for edge in result.candidates])
    if args.out_selected_edges_csv is not None:
        dump_csv(args.out_selected_edges_csv, [edge.to_dict() for edge in result.selected])
    if args.out_rejected_edges_csv is not None:
        dump_csv(args.out_rejected_edges_csv, result.rejected_rows)
    if args.out_trace_csv is not None:
        dump_csv(args.out_trace_csv, result.trace_rows)
    if args.out_components_csv is not None:
        save_component_rows(result.graph, args.out_components_csv)
    if args.out_component_relations_csv is not None:
        save_component_relation_rows(result.graph, args.out_component_relations_csv)
    if args.out_axis_relations_csv is not None:
        save_axis_relation_rows(result.graph, args.out_axis_relations_csv)
    print(
        f"[rcm-select-graph] candidates={len(result.candidates)} selected={len(result.selected)} "
        f"axes={','.join(sorted(result.graph.axes))} out={args.out_graph_json}"
    )


if __name__ == "__main__":
    main()
