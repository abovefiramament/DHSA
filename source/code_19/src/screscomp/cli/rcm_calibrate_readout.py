from __future__ import annotations

import argparse
from pathlib import Path
from statistics import median
from typing import Any

from screscomp.data import dump_csv, load_jsonl
from screscomp.rcm.prediction import auroc


DEFAULT_TARGETS = [
    "label_context_only",
    "label_prior_only",
    "label_failure_not_context_only",
    "label_composite_success",
    "label_first_answer_cf_em",
    "label_short_output",
    "label_single_commitment",
]

DEFAULT_TARGET_AXES = {
    "label_context_only": {"source_identity", "prior_suppression"},
    "label_prior_only": {"source_identity", "prior_suppression"},
    "label_failure_not_context_only": {"source_identity", "prior_suppression"},
    "label_composite_success": {"source_identity", "prior_suppression", "form", "commitment"},
    "label_first_answer_cf_em": {"form"},
    "label_short_output": {"form"},
    "label_single_commitment": {"commitment", "source_identity", "prior_suppression"},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Freeze a readout calibration from RCM edge raw scores. Control signs are left untouched; "
            "this only learns whether each edge's natural activation is positive or negative evidence "
            "for each event label."
        )
    )
    p.add_argument("--predictions_jsonl", type=Path, required=True)
    p.add_argument("--out_csv", type=Path, required=True)
    p.add_argument("--out_edge_stats_csv", type=Path, default=None)
    p.add_argument("--targets", type=str, default=",".join(DEFAULT_TARGETS))
    p.add_argument(
        "--edge_score_field",
        choices=["raw_score", "projection_score"],
        default="raw_score",
        help="Edge score field used for calibration and robust edge stats.",
    )
    p.add_argument("--min_directional_auroc", type=float, default=0.55)
    p.add_argument("--top_k_per_target", type=int, default=0, help="0 keeps all edges passing the threshold.")
    p.add_argument(
        "--weight_mode",
        choices=["readout", "causal_prior", "hybrid"],
        default="hybrid",
        help=(
            "readout uses only validation separability; causal_prior uses only the graph edge weight; "
            "hybrid multiplies both and is the default white-box readout."
        ),
    )
    p.add_argument(
        "--axis_policy",
        choices=["default", "none"],
        default="default",
        help=(
            "default restricts each target to its intended RCM axes; none allows every graph edge to predict every target."
        ),
    )
    return p.parse_args()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _mad(values: list[float]) -> float:
    if not values:
        return 0.0
    center = median(values)
    return float(median([abs(value - center) for value in values]))


def _edge_score(edge: dict[str, Any], score_field: str) -> float:
    if score_field in edge:
        return float(edge.get(score_field, 0.0) or 0.0)
    return float(edge.get("raw_score", 0.0) or 0.0)


def _control_sign_value(control_sign: str) -> float:
    if control_sign == "up":
        return 1.0
    if control_sign == "down":
        return -1.0
    return 0.0


def _edge_stats_rows(rows: list[dict[str, Any]], *, score_field: str) -> dict[str, dict[str, Any]]:
    edge_meta: dict[str, dict[str, Any]] = {}
    edge_scores: dict[str, list[float]] = {}
    for row in rows:
        for edge in row.get("edge_scores", []):
            edge_id = str(edge["edge_id"])
            edge_meta.setdefault(
                edge_id,
                {
                    "edge_id": edge_id,
                    "axis": edge.get("axis", ""),
                    "component_id": edge.get("component_id", ""),
                    "control_sign": edge.get("sign", ""),
                    "timing": edge.get("timing", ""),
                    "graph_weight": float(edge.get("weight", 1.0) or 1.0),
                },
            )
            edge_scores.setdefault(edge_id, []).append(_edge_score(edge, score_field))

    output: dict[str, dict[str, Any]] = {}
    for edge_id, scores in edge_scores.items():
        local_median = float(median(scores)) if scores else 0.0
        local_mad = _mad(scores)
        robust_scale = 1.4826 * local_mad
        if robust_scale <= 1e-12:
            robust_scale = 1.0
        output[edge_id] = {
            **edge_meta[edge_id],
            "n": len(scores),
            "median": local_median,
            "mad": local_mad,
            "robust_scale": robust_scale,
            "mean": _mean(scores),
            "min": min(scores) if scores else None,
            "max": max(scores) if scores else None,
            "score_field": score_field,
        }
    return output


def _edge_rows_for_target(
    rows: list[dict[str, Any]],
    target: str,
    *,
    score_field: str,
    edge_stats: dict[str, dict[str, Any]],
    allowed_axes: set[str] | None,
) -> list[dict[str, Any]]:
    labels = [_as_bool(row.get(target)) for row in rows]
    positives = sum(labels)
    if positives == 0 or positives == len(labels):
        return []

    edge_meta: dict[str, dict[str, Any]] = {}
    edge_scores: dict[str, list[float]] = {}
    for row in rows:
        for edge in row.get("edge_scores", []):
            edge_id = str(edge["edge_id"])
            edge_meta.setdefault(edge_id, edge_stats.get(edge_id, {"edge_id": edge_id}))
            axis = str(edge_meta[edge_id].get("axis", edge.get("axis", "")))
            if allowed_axes is not None and axis not in allowed_axes:
                continue
            edge_scores.setdefault(edge_id, []).append(_edge_score(edge, score_field))

    output: list[dict[str, Any]] = []
    for edge_id, scores in edge_scores.items():
        if len(scores) != len(labels):
            continue
        score_auroc = auroc(labels, scores)
        if score_auroc is None:
            continue
        directional_auroc = max(score_auroc, 1.0 - score_auroc)
        readout_sign = 1.0 if score_auroc >= 0.5 else -1.0
        readout_strength = max(0.0, 2.0 * (directional_auroc - 0.5))
        pos_scores = [score for score, label in zip(scores, labels) if label]
        neg_scores = [score for score, label in zip(scores, labels) if not label]
        output.append(
            {
                "target": target,
                **edge_meta[edge_id],
                "n": len(labels),
                "positives": positives,
                "positive_rate": positives / len(labels),
                "auroc": score_auroc,
                "directional_auroc": directional_auroc,
                "readout_sign": readout_sign,
                "readout_strength": readout_strength,
                "control_sign_value": _control_sign_value(str(edge_meta[edge_id].get("control_sign", ""))),
                "readout_control_agree": readout_sign
                == _control_sign_value(str(edge_meta[edge_id].get("control_sign", ""))),
                "graph_weight": edge_meta[edge_id]["graph_weight"],
                "median": edge_meta[edge_id].get("median", 0.0),
                "mad": edge_meta[edge_id].get("mad", 0.0),
                "robust_scale": edge_meta[edge_id].get("robust_scale", 1.0),
                "mean_score_positive": _mean(pos_scores),
                "mean_score_negative": _mean(neg_scores),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.predictions_jsonl)
    edge_stats = _edge_stats_rows(rows, score_field=args.edge_score_field)
    if args.out_edge_stats_csv is not None:
        dump_csv(args.out_edge_stats_csv, [edge_stats[key] for key in sorted(edge_stats)])
    targets = [target.strip() for target in args.targets.split(",") if target.strip()]
    output: list[dict[str, Any]] = []
    for target in targets:
        local = [
            row
            for row in _edge_rows_for_target(
                rows,
                target,
                score_field=args.edge_score_field,
                edge_stats=edge_stats,
                allowed_axes=(DEFAULT_TARGET_AXES.get(target) if args.axis_policy == "default" else None),
            )
            if float(row["directional_auroc"]) >= args.min_directional_auroc
        ]
        local.sort(key=lambda row: (-float(row["directional_auroc"]), str(row["edge_id"])))
        if args.top_k_per_target > 0:
            local = local[: args.top_k_per_target]
        for rank, row in enumerate(local, start=1):
            row["readout_rank"] = rank
            graph_weight = abs(float(row.get("graph_weight", 1.0) or 1.0))
            readout_strength = float(row.get("readout_strength", 0.0) or 0.0)
            if args.weight_mode == "readout":
                row["weight"] = readout_strength
            elif args.weight_mode == "causal_prior":
                row["weight"] = graph_weight
            else:
                row["weight"] = graph_weight * readout_strength
            row["weight_mode"] = args.weight_mode
        output.extend(local)
    dump_csv(args.out_csv, output)
    print(
        f"[rcm-calibrate-readout] rows={len(rows)} targets={len(targets)} "
        f"calibration_edges={len(output)} out={args.out_csv}"
    )


if __name__ == "__main__":
    main()
