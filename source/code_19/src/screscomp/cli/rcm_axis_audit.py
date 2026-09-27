from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from screscomp.data import dump_csv, load_csv
from screscomp.rcm.graph import RCMGraph, load_graph


@dataclass(frozen=True, slots=True)
class AxisAuditSpec:
    audit_name: str
    axis: str
    target: str
    strict_scores: tuple[str, ...]
    calibrated_score: str = ""
    description: str = ""


AUDIT_SPECS = [
    AxisAuditSpec(
        audit_name="source_identity_context_side",
        axis="source_identity",
        target="label_context_only",
        strict_scores=(
            "source_identity_up_score",
            "source_identity_margin_score",
            "source_identity_up_down_score",
            "source_identity_momentum_score",
            "source_identity_balance_ratio_score",
            "source_identity_potential_score",
        ),
        calibrated_score="calibrated_label_context_only_score",
        description="Does the source axis read context adoption by itself?",
    ),
    AxisAuditSpec(
        audit_name="source_identity_prior_side",
        axis="source_identity",
        target="label_prior_only",
        strict_scores=(
            "anti_source_identity_score",
            "prior_failure_conjunction_score",
            "prior_failure_energy_score",
        ),
        calibrated_score="calibrated_label_prior_only_score",
        description="Does the source axis read prior adoption/leakage?",
    ),
    AxisAuditSpec(
        audit_name="prior_suppression_failure_side",
        axis="prior_suppression",
        target="label_prior_only",
        strict_scores=(
            "anti_prior_suppression_score",
            "prior_failure_conjunction_score",
            "prior_failure_energy_score",
        ),
        calibrated_score="calibrated_label_prior_only_score",
        description="Does the prior-suppression axis identify failure to suppress prior memory?",
    ),
    AxisAuditSpec(
        audit_name="form_short_side",
        axis="form",
        target="label_short_output",
        strict_scores=(
            "form_up_score",
            "form_margin_score",
            "form_up_down_score",
            "form_momentum_score",
            "form_balance_ratio_score",
            "form_potential_score",
        ),
        calibrated_score="calibrated_label_short_output_score",
        description="Does the form axis read short/concise output?",
    ),
    AxisAuditSpec(
        audit_name="form_exact_first_answer_side",
        axis="form",
        target="label_first_answer_cf_em",
        strict_scores=(
            "form_potential_score",
            "form_margin_score",
            "form_up_down_score",
            "form_momentum_score",
            "form_balance_ratio_score",
            "form_up_score",
        ),
        calibrated_score="calibrated_label_first_answer_cf_em_score",
        description="Does the form axis read exact first-answer behavior?",
    ),
    AxisAuditSpec(
        audit_name="commitment_single_side",
        axis="commitment",
        target="label_single_commitment",
        strict_scores=(
            "commitment_potential_score",
            "commitment_margin_score",
            "commitment_up_down_score",
            "commitment_momentum_score",
            "commitment_balance_ratio_score",
            "commitment_up_score",
        ),
        calibrated_score="calibrated_label_single_commitment_score",
        description="Does the commitment axis read single-answer commitment?",
    ),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Audit each RCM axis before multi-axis composition. The audit only accepts axis-native "
            "strict readouts; calibrated readouts are reported as secondary evidence."
        )
    )
    p.add_argument("--strict_summary_csv", type=Path, required=True)
    p.add_argument("--calibrated_summary_csv", type=Path, default=None)
    p.add_argument("--graph_json", type=Path, default=None)
    p.add_argument("--out_csv", type=Path, required=True)
    p.add_argument("--pass_auroc", type=float, default=0.65)
    p.add_argument("--provisional_auroc", type=float, default=0.58)
    return p.parse_args()


def _float(value: Any) -> float | None:
    if value in ("", None):
        return None
    return float(value)


def _index_summary(path: Path | None) -> dict[tuple[str, str], dict[str, str]]:
    if path is None:
        return {}
    return {(row["target"], row["score_name"]): row for row in load_csv(path)}


def _best_row(
    indexed: dict[tuple[str, str], dict[str, str]],
    *,
    target: str,
    score_names: tuple[str, ...],
) -> dict[str, str] | None:
    candidates = [indexed[(target, score)] for score in score_names if (target, score) in indexed]
    candidates = [row for row in candidates if _float(row.get("auroc")) is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda row: float(row["auroc"]))


def _status(auroc: float | None, *, pass_auroc: float, provisional_auroc: float, edge_count: int) -> str:
    if edge_count <= 0:
        return "not_ready_no_edges"
    if auroc is None:
        return "not_ready_no_score"
    if auroc >= pass_auroc:
        return "pass"
    if auroc >= provisional_auroc:
        return "provisional"
    return "fail"


def _axis_counts(graph: RCMGraph | None) -> dict[str, dict[str, Any]]:
    if graph is None:
        return {}
    output: dict[str, dict[str, Any]] = {}
    for axis in graph.axes:
        nodes = graph.by_axis(axis)
        output[axis] = {
            "edge_count": len(nodes),
            "up_edges": sum(1 for node in nodes if node.sign == "up"),
            "down_edges": sum(1 for node in nodes if node.sign == "down"),
            "timings": ",".join(sorted({node.timing for node in nodes})),
        }
    return output


def main() -> None:
    args = parse_args()
    strict = _index_summary(args.strict_summary_csv)
    calibrated = _index_summary(args.calibrated_summary_csv)
    graph = load_graph(args.graph_json) if args.graph_json is not None else None
    counts = _axis_counts(graph)

    rows: list[dict[str, Any]] = []
    for spec in AUDIT_SPECS:
        strict_best = _best_row(strict, target=spec.target, score_names=spec.strict_scores)
        calibrated_row = calibrated.get((spec.target, spec.calibrated_score)) if spec.calibrated_score else None
        axis_count = counts.get(spec.axis, {})
        edge_count = int(axis_count.get("edge_count", 0))
        strict_auroc = _float(strict_best.get("auroc")) if strict_best else None
        calibrated_auroc = _float(calibrated_row.get("auroc")) if calibrated_row else None
        rows.append(
            {
                "audit_name": spec.audit_name,
                "axis": spec.axis,
                "target": spec.target,
                "status": _status(
                    strict_auroc,
                    pass_auroc=args.pass_auroc,
                    provisional_auroc=args.provisional_auroc,
                    edge_count=edge_count,
                ),
                "strict_best_score": strict_best.get("score_name", "") if strict_best else "",
                "strict_auroc": strict_auroc,
                "strict_average_precision": strict_best.get("average_precision", "") if strict_best else "",
                "calibrated_score": spec.calibrated_score,
                "calibrated_auroc": calibrated_auroc,
                "calibrated_average_precision": calibrated_row.get("average_precision", "") if calibrated_row else "",
                "edge_count": edge_count,
                "up_edges": axis_count.get("up_edges", 0),
                "down_edges": axis_count.get("down_edges", 0),
                "timings": axis_count.get("timings", ""),
                "description": spec.description,
            }
        )
    dump_csv(args.out_csv, rows)
    print(f"[rcm-axis-audit] rows={len(rows)} out={args.out_csv}")


if __name__ == "__main__":
    main()
