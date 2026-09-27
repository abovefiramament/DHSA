from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from screscomp.cecm.audit import (
    ScreenRow,
    accepted,
    bootstrap_ci,
    ci,
    ci_excludes_zero,
    cosine,
    load_component_screen,
    paired_values,
    pearson,
    permutation_p,
    robust_status,
    spearman,
    split_status,
    topk_overlap_row,
)
from screscomp.data import dump_csv, dump_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Audit CECM component contribution confidence across splits and score operators."
    )
    p.add_argument("--avglogp-a", type=Path, required=True)
    p.add_argument("--avglogp-b", type=Path, required=True)
    p.add_argument("--margin-a", type=Path, required=True)
    p.add_argument("--margin-b", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--tau-delta", type=float, default=0.10)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--permutations", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _row_fields(prefix: str, row: ScreenRow | None, *, alpha: float, tau_delta: float) -> dict[str, object]:
    if row is None:
        return {
            f"{prefix}_mean_delta": "",
            f"{prefix}_ci_low": "",
            f"{prefix}_ci_high": "",
            f"{prefix}_ci_excludes_zero": "",
            f"{prefix}_accepted": "",
            f"{prefix}_positive_rate": "",
            f"{prefix}_negative_rate": "",
            f"{prefix}_sign_consistency": "",
        }
    low, high = ci(row, alpha=alpha)
    return {
        f"{prefix}_mean_delta": row.mean_delta,
        f"{prefix}_abs_mean_delta": abs(row.mean_delta),
        f"{prefix}_ci_low": low,
        f"{prefix}_ci_high": high,
        f"{prefix}_ci_excludes_zero": int(ci_excludes_zero(row, alpha=alpha)),
        f"{prefix}_accepted": int(accepted(row, alpha=alpha, tau_delta=tau_delta)),
        f"{prefix}_positive_rate": row.positive_rate,
        f"{prefix}_negative_rate": row.negative_rate,
        f"{prefix}_sign_consistency": row.sign_consistency,
    }


def _mean_delta(row_a: ScreenRow | None, row_b: ScreenRow | None) -> float:
    values = [row.mean_delta for row in (row_a, row_b) if row is not None]
    return sum(values) / len(values) if values else math.nan


def component_confidence_rows(
    *,
    avg_a: dict[str, ScreenRow],
    avg_b: dict[str, ScreenRow],
    margin_a: dict[str, ScreenRow],
    margin_b: dict[str, ScreenRow],
    alpha: float,
    tau_delta: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    component_ids = sorted(set(avg_a) | set(avg_b) | set(margin_a) | set(margin_b))
    confidence_rows: list[dict[str, object]] = []
    robust_core: list[dict[str, object]] = []
    stable_by_score: list[dict[str, object]] = []

    for component_id in component_ids:
        a1 = avg_a.get(component_id)
        b1 = avg_b.get(component_id)
        a2 = margin_a.get(component_id)
        b2 = margin_b.get(component_id)
        avg_status = split_status(a1, b1, alpha=alpha, tau_delta=tau_delta)
        margin_status = split_status(a2, b2, alpha=alpha, tau_delta=tau_delta)
        status, level = robust_status(
            avg_status,
            margin_status,
            avg_a=a1,
            avg_b=b1,
            margin_a=a2,
            margin_b=b2,
            tau_delta=tau_delta,
        )
        meta = next(row for row in (a1, b1, a2, b2) if row is not None)
        row = {
            "component_id": component_id,
            "layer_idx": meta.layer_idx,
            "component_type": meta.component_type,
            "alpha": alpha,
            "confidence": 1.0 - alpha,
            "tau_delta": tau_delta,
            "avglogp_split_status": avg_status,
            "answer_rest_margin_split_status": margin_status,
            "robust_status": status,
            "confidence_level": level,
            "avglogp_mean_delta_over_splits": _mean_delta(a1, b1),
            "answer_rest_margin_mean_delta_over_splits": _mean_delta(a2, b2),
            **_row_fields("avglogp_a", a1, alpha=alpha, tau_delta=tau_delta),
            **_row_fields("avglogp_b", b1, alpha=alpha, tau_delta=tau_delta),
            **_row_fields("margin_a", a2, alpha=alpha, tau_delta=tau_delta),
            **_row_fields("margin_b", b2, alpha=alpha, tau_delta=tau_delta),
        }
        confidence_rows.append(row)
        if avg_status in {"stable_positive", "stable_negative"}:
            stable_by_score.append({**row, "score_mode": "avglogp", "score_split_status": avg_status})
        if margin_status in {"stable_positive", "stable_negative"}:
            stable_by_score.append(
                {**row, "score_mode": "answer_rest_margin", "score_split_status": margin_status}
            )
        if level == "high":
            robust_core.append(row)

    confidence_rows.sort(
        key=lambda row: (
            {"high": 0, "medium": 1, "low": 2, "reject": 3}.get(str(row["confidence_level"]), 9),
            str(row["robust_status"]),
            -max(
                abs(float(row["avglogp_mean_delta_over_splits"] or 0.0)),
                abs(float(row["answer_rest_margin_mean_delta_over_splits"] or 0.0)),
            ),
            str(row["component_id"]),
        )
    )
    robust_core.sort(
        key=lambda row: (
            str(row["robust_status"]),
            -max(
                abs(float(row["avglogp_mean_delta_over_splits"] or 0.0)),
                abs(float(row["answer_rest_margin_mean_delta_over_splits"] or 0.0)),
            ),
            str(row["component_id"]),
        )
    )
    return confidence_rows, stable_by_score, robust_core


def correlation_rows(
    screens: dict[str, dict[str, ScreenRow]],
    *,
    n_bootstrap: int,
    n_permutations: int,
    seed: int,
) -> list[dict[str, object]]:
    comparisons = [
        ("avglogp_split", "avglogp_a", "avglogp_b"),
        ("answer_rest_margin_split", "margin_a", "margin_b"),
        ("score_operator_split_a", "avglogp_a", "margin_a"),
        ("score_operator_split_b", "avglogp_b", "margin_b"),
    ]
    metrics = [
        ("pearson", pearson),
        ("spearman", spearman),
        ("cosine", cosine),
    ]
    rows: list[dict[str, object]] = []
    for comparison, left_name, right_name in comparisons:
        ids, x, y = paired_values(screens[left_name], screens[right_name])
        for metric_name, metric_fn in metrics:
            rng = random.Random(seed + hash((comparison, metric_name)) % 1_000_000)
            observed = metric_fn(x, y)
            boot_low, boot_high = bootstrap_ci(
                x,
                y,
                metric_fn=metric_fn,
                n_bootstrap=n_bootstrap,
                rng=rng,
            )
            p_right, p_abs = permutation_p(
                x,
                y,
                observed=observed,
                metric_fn=metric_fn,
                n_permutations=n_permutations,
                rng=rng,
            )
            rows.append(
                {
                    "comparison": comparison,
                    "left": left_name,
                    "right": right_name,
                    "metric": metric_name,
                    "n_components": len(ids),
                    "estimate": observed,
                    "bootstrap_ci95_low": boot_low,
                    "bootstrap_ci95_high": boot_high,
                    "permutation_p_right": p_right,
                    "permutation_p_abs": p_abs,
                }
            )
    return rows


def topk_rows(screens: dict[str, dict[str, ScreenRow]], *, k: int) -> list[dict[str, object]]:
    comparisons = [
        ("avglogp_split", "avglogp_a", "avglogp_b"),
        ("answer_rest_margin_split", "margin_a", "margin_b"),
        ("score_operator_split_a", "avglogp_a", "margin_a"),
        ("score_operator_split_b", "avglogp_b", "margin_b"),
    ]
    rows: list[dict[str, object]] = []
    for comparison, left_name, right_name in comparisons:
        for direction in ("positive", "negative"):
            rows.append(
                topk_overlap_row(
                    comparison=comparison,
                    left=screens[left_name],
                    right=screens[right_name],
                    direction=direction,
                    k=k,
                )
            )
    return rows


def main() -> None:
    args = parse_args()
    screens = {
        "avglogp_a": load_component_screen(args.avglogp_a),
        "avglogp_b": load_component_screen(args.avglogp_b),
        "margin_a": load_component_screen(args.margin_a),
        "margin_b": load_component_screen(args.margin_b),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)

    confidence, stable_by_score, robust_core = component_confidence_rows(
        avg_a=screens["avglogp_a"],
        avg_b=screens["avglogp_b"],
        margin_a=screens["margin_a"],
        margin_b=screens["margin_b"],
        alpha=args.alpha,
        tau_delta=args.tau_delta,
    )
    corr = correlation_rows(
        screens,
        n_bootstrap=args.bootstrap,
        n_permutations=args.permutations,
        seed=args.seed,
    )
    topk = topk_rows(screens, k=args.top_k)

    dump_csv(args.out_dir / "component_confidence.csv", confidence)
    dump_csv(args.out_dir / "stable_edges_by_score.csv", stable_by_score)
    dump_csv(args.out_dir / "robust_core.csv", robust_core)
    dump_csv(args.out_dir / "correlation_audit.csv", corr)
    dump_csv(args.out_dir / "topk_overlap_audit.csv", topk)
    dump_csv(
        args.out_dir / "audit_summary.csv",
        [
            {"metric": "alpha", "value": args.alpha},
            {"metric": "confidence", "value": 1.0 - args.alpha},
            {"metric": "tau_delta", "value": args.tau_delta},
            {"metric": "top_k", "value": args.top_k},
            {"metric": "components_total", "value": len(confidence)},
            {"metric": "robust_core_size", "value": len(robust_core)},
            {"metric": "stable_edges_by_score", "value": len(stable_by_score)},
        ],
    )
    dump_json(
        args.out_dir / "audit_config.json",
        {
            "avglogp_a": str(args.avglogp_a),
            "avglogp_b": str(args.avglogp_b),
            "margin_a": str(args.margin_a),
            "margin_b": str(args.margin_b),
            "alpha": args.alpha,
            "confidence": 1.0 - args.alpha,
            "tau_delta": args.tau_delta,
            "top_k": args.top_k,
            "bootstrap": args.bootstrap,
            "permutations": args.permutations,
            "seed": args.seed,
            "edge_acceptance": (
                "A component edge is accepted for a score mode only if split A and split B "
                "both have confidence intervals excluding zero at the requested alpha, "
                "both exceed tau_delta in absolute mean contribution, and their signs agree."
            ),
        },
    )
    print(
        (
            f"[cecm-audit] components={len(confidence)} robust_core={len(robust_core)} "
            f"stable_edges_by_score={len(stable_by_score)} out={args.out_dir}"
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
