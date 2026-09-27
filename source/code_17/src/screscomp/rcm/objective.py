from __future__ import annotations

from dataclasses import replace
from typing import Any

from .select import CandidateEdge


def edge_match_key(row: dict[str, Any]) -> str:
    edge_id = str(row.get("edge_id", "")).strip()
    if edge_id:
        return edge_id
    axis = str(row.get("axis", "")).strip()
    component_id = str(row.get("component_id", "")).strip()
    sign = str(row.get("sign", "")).strip()
    timing = str(row.get("timing", "")).strip()
    if axis and component_id and sign and timing:
        return f"{axis}:{component_id}:{sign}:{timing}"
    raise ValueError("Evaluation row needs edge_id or axis/component_id/sign/timing fields.")


def pair_match_key(row: dict[str, Any]) -> tuple[str, str]:
    edge_a = str(row.get("edge_id_a", row.get("edge_a", ""))).strip()
    edge_b = str(row.get("edge_id_b", row.get("edge_b", ""))).strip()
    if not edge_a or not edge_b:
        raise ValueError("Pairwise evaluation row needs edge_id_a and edge_id_b fields.")
    return tuple(sorted((edge_a, edge_b)))


def weighted_objective(row: dict[str, Any], terms: dict[str, float]) -> float:
    score = 0.0
    for field, weight in terms.items():
        value = row.get(field)
        if value in ("", None):
            continue
        score += weight * float(value)
    return score


def apply_candidate_validation_scores(
    candidates: list[CandidateEdge],
    eval_rows: list[dict[str, Any]],
    *,
    objective_terms: dict[str, float],
    rank_prior_weight: float = 0.0,
    operation: str | None = None,
) -> list[CandidateEdge]:
    """Blend ranking worth with true validation objective rows when available."""

    if operation is not None:
        eval_rows = [row for row in eval_rows if str(row.get("operation", operation)) == operation]
    by_edge = {edge_match_key(row): row for row in eval_rows}
    output: list[CandidateEdge] = []
    for edge in candidates:
        row = by_edge.get(edge.edge_id)
        if row is None:
            output.append(edge)
            continue
        validation_score = weighted_objective(row, objective_terms)
        output.append(
            replace(
                edge,
                utility=validation_score + rank_prior_weight * edge.utility,
                metadata={
                    **edge.metadata,
                    "validation_score": validation_score,
                    "rank_prior_utility": edge.utility,
                    "rank_prior_weight": rank_prior_weight,
                },
            )
        )
    return output


def load_pair_adjustments(
    pair_rows: list[dict[str, Any]],
    *,
    objective_terms: dict[str, float],
) -> dict[tuple[str, str], float]:
    output: dict[tuple[str, str], float] = {}
    for row in pair_rows:
        output[pair_match_key(row)] = weighted_objective(row, objective_terms)
    return output
