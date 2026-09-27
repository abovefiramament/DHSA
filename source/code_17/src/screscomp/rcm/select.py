from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from screscomp.data import load_csv

from .events import AxisSpec, default_axis_specs
from .graph import ComponentNode, RCMGraph


@dataclass(slots=True)
class CandidateEdge:
    edge_id: str
    axis: str
    component_id: str
    layer_idx: int
    component_type: str
    sign: str
    timing: str
    alpha: float
    rank: int
    rank_score: float
    selection_score: float
    utility: float
    source_path: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SelectionConfig:
    budget: int = 16
    per_axis_quota: int = 8
    max_candidates_per_axis_sign: int = 24
    rrf_k: float = 60.0
    rank_weight: float = 1.0
    effect_weight: float = 0.0
    edge_cost: float = 0.0
    same_component_penalty: float = 0.03
    mixed_timing_penalty: float = 0.01
    one_timing_per_component_axis_sign: bool = True
    local_search_rounds: int = 1


@dataclass(slots=True)
class SelectionResult:
    graph: RCMGraph
    candidates: list[CandidateEdge]
    selected: list[CandidateEdge]
    rejected_rows: list[dict[str, Any]]
    trace_rows: list[dict[str, Any]]


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value in ("", None):
        return default
    return float(value)


def _int(row: dict[str, Any], key: str, default: int = 0) -> int:
    value = row.get(key, default)
    if value in ("", None):
        return default
    return int(float(value))


def _rank_score(rank: int, total: int, rrf_k: float) -> float:
    if total <= 1:
        normalized = 1.0
    else:
        normalized = 1.0 - ((rank - 1) / (total - 1))
    rrf = 1.0 / (rrf_k + rank)
    # Keep the score in a readable range while retaining the top-heavy RRF bump.
    return normalized + rrf_k * rrf / (rrf_k + 1.0)


def _axis_rows(rows: list[dict[str, str]], axis: str) -> list[dict[str, str]]:
    selected = [row for row in rows if str(row.get("axis", axis)) == axis]
    return selected or rows


def _axis_timing_list(axis: str, axis_specs: dict[str, AxisSpec], axis_candidate_timings: dict[str, list[str]]) -> list[str]:
    timings = axis_candidate_timings.get(axis)
    if timings:
        return timings
    return [axis_specs[axis].default_timing]


def _axis_alpha(axis: str, axis_specs: dict[str, AxisSpec], axis_alphas: dict[str, float]) -> float:
    return axis_alphas.get(axis, axis_specs[axis].default_alpha)


def _candidate_utility(
    *,
    axis: str,
    rank_score: float,
    selection_score: float,
    axis_weights: dict[str, float],
    config: SelectionConfig,
    penalty_field_weights: dict[str, float],
    row: dict[str, Any],
) -> float:
    penalty = sum(weight * abs(_float(row, field_name)) for field_name, weight in penalty_field_weights.items())
    return (
        axis_weights.get(axis, 1.0) * config.rank_weight * rank_score
        + config.effect_weight * abs(selection_score)
        - penalty
        - config.edge_cost
    )


def load_candidate_edges_from_summaries(
    axis_summary_csvs: dict[str, Path],
    *,
    axis_candidate_timings: dict[str, list[str]] | None = None,
    axis_alphas: dict[str, float] | None = None,
    axis_weights: dict[str, float] | None = None,
    penalty_field_weights: dict[str, float] | None = None,
    config: SelectionConfig | None = None,
) -> list[CandidateEdge]:
    axis_specs = default_axis_specs()
    axis_candidate_timings = axis_candidate_timings or {}
    axis_alphas = axis_alphas or {}
    axis_weights = axis_weights or {}
    penalty_field_weights = penalty_field_weights or {}
    config = config or SelectionConfig()
    candidates: list[CandidateEdge] = []

    for axis, path in axis_summary_csvs.items():
        if axis not in axis_specs:
            raise ValueError(f"Unknown axis: {axis}")
        rows = _axis_rows(load_csv(path), axis)
        usable = [row for row in rows if row.get("component_id") or (row.get("layer_idx") and row.get("component_type"))]
        by_sign = {
            "up": sorted(
                [row for row in usable if _float(row, "selection_score") >= 0.0],
                key=lambda row: (-_float(row, "selection_score"), _int(row, "selection_rank", 10**9)),
            ),
            "down": sorted(
                [row for row in usable if _float(row, "selection_score") < 0.0],
                key=lambda row: (_float(row, "selection_score"), _int(row, "selection_rank", 10**9)),
            ),
        }
        timings = _axis_timing_list(axis, axis_specs, axis_candidate_timings)
        alpha = _axis_alpha(axis, axis_specs, axis_alphas)
        for sign, sign_rows in by_sign.items():
            trimmed = sign_rows[: config.max_candidates_per_axis_sign]
            total = len(trimmed)
            for rank, row in enumerate(trimmed, start=1):
                layer_idx = _int(row, "layer_idx")
                component_type = str(row.get("component_type", ""))
                component_id = str(row.get("component_id") or f"L{layer_idx}.{component_type}")
                selection_score = _float(row, "selection_score")
                rank_score = _rank_score(rank, total, config.rrf_k)
                for timing in timings:
                    utility = _candidate_utility(
                        axis=axis,
                        rank_score=rank_score,
                        selection_score=selection_score,
                        axis_weights=axis_weights,
                        config=config,
                        penalty_field_weights=penalty_field_weights,
                        row=row,
                    )
                    candidates.append(
                        CandidateEdge(
                            edge_id=f"{axis}:{component_id}:{sign}:{timing}",
                            axis=axis,
                            component_id=component_id,
                            layer_idx=layer_idx,
                            component_type=component_type,
                            sign=sign,
                            timing=timing,
                            alpha=alpha,
                            rank=rank,
                            rank_score=rank_score,
                            selection_score=selection_score,
                            utility=utility,
                            source_path=str(path),
                            metadata={
                                key: row[key]
                                for key in (
                                    "pair_name",
                                    "selection_rule",
                                    "selection_metric",
                                    "mean_context_only",
                                    "mean_prior_only",
                                    "mean_both",
                                    "mean_neither",
                                    "mean_form_score",
                                    "mean_commitment_score",
                                    "mean_source_score",
                                    "mean_joint_score",
                                    "mean_output_chars",
                                )
                                if key in row
                            },
                        )
                    )
    return candidates


def _hard_conflict(a: CandidateEdge, b: CandidateEdge, config: SelectionConfig) -> str | None:
    if a.edge_id == b.edge_id:
        return "same_edge"
    if (
        config.one_timing_per_component_axis_sign
        and a.component_id == b.component_id
        and a.axis == b.axis
        and a.sign == b.sign
        and a.timing != b.timing
    ):
        return "same_component_axis_sign_different_timing"
    if a.component_id == b.component_id and a.timing == b.timing and a.sign != b.sign:
        return "same_component_timing_opposite_sign"
    return None


def _pair_penalty(a: CandidateEdge, b: CandidateEdge, config: SelectionConfig) -> float:
    if a.component_id != b.component_id:
        return 0.0
    penalty = config.same_component_penalty
    if a.timing != b.timing:
        penalty += config.mixed_timing_penalty
    return penalty


def _is_feasible(
    selected: list[CandidateEdge],
    candidate: CandidateEdge,
    *,
    config: SelectionConfig,
    per_axis_counts: dict[str, int],
) -> tuple[bool, str]:
    for edge in selected:
        conflict = _hard_conflict(edge, candidate, config)
        if conflict:
            return False, conflict
    if per_axis_counts.get(candidate.axis, 0) >= config.per_axis_quota:
        return False, "per_axis_quota_exhausted"
    if len(selected) >= config.budget:
        return False, "budget_exhausted"
    return True, ""


def _set_score(edges: list[CandidateEdge], config: SelectionConfig) -> float:
    return _set_score_with_pairs(edges, config, pair_adjustments=None)


def _set_score_with_pairs(
    edges: list[CandidateEdge],
    config: SelectionConfig,
    pair_adjustments: dict[tuple[str, str], float] | None,
) -> float:
    score = sum(edge.utility for edge in edges)
    for idx, edge_a in enumerate(edges):
        for edge_b in edges[idx + 1 :]:
            conflict = _hard_conflict(edge_a, edge_b, config)
            if conflict:
                return float("-inf")
            score -= _pair_penalty(edge_a, edge_b, config)
            if pair_adjustments:
                score += pair_adjustments.get(tuple(sorted((edge_a.edge_id, edge_b.edge_id))), 0.0)
    return score


def _to_graph(name: str, selected: list[CandidateEdge], metadata: dict[str, Any]) -> RCMGraph:
    axis_specs = default_axis_specs()
    axes = {axis: axis_specs[axis] for axis in sorted({edge.axis for edge in selected})}
    components = [
        ComponentNode(
            axis=edge.axis,
            component_id=edge.component_id,
            layer_idx=edge.layer_idx,
            component_type=edge.component_type,
            sign=edge.sign,
            weight=max(edge.utility, 0.0),
            selection_score=edge.selection_score,
            selection_metric=str(edge.metadata.get("selection_metric") or axis_specs[edge.axis].selection_metric),
            timing=edge.timing,
            alpha=edge.alpha,
            source_path=edge.source_path,
            rank=edge.rank,
            metadata={**edge.metadata, "edge_id": edge.edge_id, "rank_score": edge.rank_score, "utility": edge.utility},
        )
        for edge in selected
    ]
    return RCMGraph(name=name, axes=axes, components=components, metadata=metadata)


def select_edges_greedy(
    candidates: list[CandidateEdge],
    *,
    name: str = "rcm_selected_graph",
    config: SelectionConfig | None = None,
    pair_adjustments: dict[tuple[str, str], float] | None = None,
) -> SelectionResult:
    config = config or SelectionConfig()
    pool = sorted(candidates, key=lambda edge: (-edge.utility, edge.axis, edge.component_id, edge.timing))
    selected: list[CandidateEdge] = []
    trace_rows: list[dict[str, Any]] = []
    per_axis_counts: dict[str, int] = {}

    while len(selected) < config.budget:
        best_edge: CandidateEdge | None = None
        best_gain = float("-inf")
        best_score = _set_score_with_pairs(selected, config, pair_adjustments)
        for edge in pool:
            if edge in selected:
                continue
            feasible, _reason = _is_feasible(selected, edge, config=config, per_axis_counts=per_axis_counts)
            if not feasible:
                continue
            candidate_score = _set_score_with_pairs([*selected, edge], config, pair_adjustments)
            gain = candidate_score - best_score
            if gain > best_gain:
                best_edge = edge
                best_gain = gain
        if best_edge is None or best_gain <= 0:
            break
        selected.append(best_edge)
        per_axis_counts[best_edge.axis] = per_axis_counts.get(best_edge.axis, 0) + 1
        trace_rows.append(
            {
                "step": len(trace_rows) + 1,
                "move": "add",
                "edge_id": best_edge.edge_id,
                "axis": best_edge.axis,
                "component_id": best_edge.component_id,
                "sign": best_edge.sign,
                "timing": best_edge.timing,
                "utility": best_edge.utility,
                "marginal_gain": best_gain,
                "set_score": _set_score(selected, config),
            }
        )

    for round_idx in range(config.local_search_rounds):
        improved = True
        while improved:
            improved = False
            current_score = _set_score_with_pairs(selected, config, pair_adjustments)
            for old in list(selected):
                base = [edge for edge in selected if edge is not old]
                base_counts: dict[str, int] = {}
                for edge in base:
                    base_counts[edge.axis] = base_counts.get(edge.axis, 0) + 1
                for new in pool:
                    if new in selected:
                        continue
                    feasible, _reason = _is_feasible(base, new, config=config, per_axis_counts=base_counts)
                    if not feasible:
                        continue
                    next_edges = [*base, new]
                    next_score = _set_score_with_pairs(next_edges, config, pair_adjustments)
                    if next_score > current_score + 1e-12:
                        selected = next_edges
                        trace_rows.append(
                            {
                                "step": len(trace_rows) + 1,
                                "move": "swap",
                                "round": round_idx + 1,
                                "dropped_edge_id": old.edge_id,
                                "edge_id": new.edge_id,
                                "axis": new.axis,
                                "component_id": new.component_id,
                                "sign": new.sign,
                                "timing": new.timing,
                                "utility": new.utility,
                                "marginal_gain": next_score - current_score,
                                "set_score": next_score,
                            }
                        )
                        improved = True
                        break
                if improved:
                    break

    selected_ids = {edge.edge_id for edge in selected}
    rejected_rows: list[dict[str, Any]] = []
    final_counts: dict[str, int] = {}
    for edge in selected:
        final_counts[edge.axis] = final_counts.get(edge.axis, 0) + 1
    for edge in pool:
        if edge.edge_id in selected_ids:
            continue
        feasible, reason = _is_feasible(selected, edge, config=config, per_axis_counts=final_counts)
        rejected_rows.append({**edge.to_dict(), "reject_reason": reason or "not_selected_positive_competition"})

    graph = _to_graph(
        name,
        selected,
        metadata={
            "selector": "rank_driven_conflict_greedy_local_search",
            "config": asdict(config),
            "candidate_count": len(candidates),
            "selected_count": len(selected),
            "set_score": _set_score_with_pairs(selected, config, pair_adjustments),
            "pair_adjustment_count": len(pair_adjustments or {}),
        },
    )
    return SelectionResult(
        graph=graph,
        candidates=pool,
        selected=selected,
        rejected_rows=rejected_rows,
        trace_rows=trace_rows,
    )
