from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from screscomp.data import dump_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Select an RCM source-control configuration under pre-declared form-preservation constraints. "
            "This is intended for validation-only selection before freezing a test configuration."
        )
    )
    p.add_argument("--summary_csv", type=Path, required=True)
    p.add_argument("--out_csv", type=Path, required=True)
    p.add_argument("--target_method", default="ours_delta_prefill_context")
    p.add_argument("--baseline_method", default="strong_rag")
    p.add_argument(
        "--baseline_summary_csv",
        type=Path,
        default=None,
        help="Optional external summary containing the baseline row. Use this to avoid rerunning baselines.",
    )
    p.add_argument("--max_cf_em_drop", type=float, default=0.025)
    p.add_argument("--max_length_increase", type=float, default=6.0)
    p.add_argument("--max_neither_increase", type=float, default=0.015)
    p.add_argument("--max_po_increase", type=float, default=0.0)
    p.add_argument("--allowed_source_apply_modes", default="")
    p.add_argument(
        "--min_context_only",
        type=float,
        default=None,
        help="Optional absolute source-dominance target, e.g. CK best context_only_rate.",
    )
    p.add_argument(
        "--max_prior_only",
        type=float,
        default=None,
        help="Optional absolute source-dominance target, e.g. CK best prior_only_rate.",
    )
    p.add_argument("--max_po", type=float, default=None, help="Optional absolute po upper bound.")
    p.add_argument("--max_mr", type=float, default=None, help="Optional absolute mr upper bound.")
    p.add_argument(
        "--min_cf_em",
        type=float,
        default=None,
        help="Optional absolute cf_em lower bound, e.g. CK best cf_em_rate.",
    )
    p.add_argument(
        "--max_mean_output_chars",
        type=float,
        default=None,
        help="Optional absolute length upper bound, e.g. CK best mean_output_chars.",
    )
    return p.parse_args()


def _as_float(row: dict[str, Any], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else 0.0


def _source_score(row: dict[str, Any]) -> float:
    return _as_float(row, "context_only_rate") - _as_float(row, "prior_only_rate") - _as_float(row, "po")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    args = parse_args()
    rows = _read_rows(args.summary_csv)
    baseline_source_rows = rows
    if args.baseline_summary_csv is not None:
        baseline_source_rows = _read_rows(args.baseline_summary_csv)
    allowed_modes = {item.strip() for item in args.allowed_source_apply_modes.split(",") if item.strip()}

    baseline_rows = [row for row in baseline_source_rows if row.get("method") == args.baseline_method]
    if not baseline_rows:
        raise ValueError(f"Could not find baseline method in summary: {args.baseline_method}")

    # The baseline can be repeated for each run. Its metrics should be identical, but use the first row
    # and report deltas against it to keep the selection rule deterministic.
    baseline = baseline_rows[0]
    base_cf = _as_float(baseline, "cf_em_rate")
    base_len = _as_float(baseline, "mean_output_chars")
    base_neither = _as_float(baseline, "neither_rate")
    base_po = _as_float(baseline, "po")

    output: list[dict[str, Any]] = []
    for row in rows:
        if row.get("method") != args.target_method:
            continue
        mode = row.get("source_apply_mode", "")
        if allowed_modes and mode not in allowed_modes:
            continue
        cf_delta = _as_float(row, "cf_em_rate") - base_cf
        len_delta = _as_float(row, "mean_output_chars") - base_len
        neither_delta = _as_float(row, "neither_rate") - base_neither
        po_delta = _as_float(row, "po") - base_po
        constraints_pass = (
            cf_delta >= -args.max_cf_em_drop
            and len_delta <= args.max_length_increase
            and neither_delta <= args.max_neither_increase
            and po_delta <= args.max_po_increase
        )
        ck_bounded_checks = {
            "context_only": args.min_context_only is None
            or _as_float(row, "context_only_rate") >= args.min_context_only,
            "prior_only": args.max_prior_only is None
            or _as_float(row, "prior_only_rate") <= args.max_prior_only,
            "po": args.max_po is None or _as_float(row, "po") <= args.max_po,
            "mr": args.max_mr is None or _as_float(row, "mr") <= args.max_mr,
            "cf_em": args.min_cf_em is None or _as_float(row, "cf_em_rate") >= args.min_cf_em,
            "length": args.max_mean_output_chars is None
            or _as_float(row, "mean_output_chars") <= args.max_mean_output_chars,
        }
        ck_bounded_pass = all(ck_bounded_checks.values())
        ck_violation_count = sum(0 if passed else 1 for passed in ck_bounded_checks.values())
        source_target_gap = 0.0
        if args.min_context_only is not None:
            source_target_gap += max(0.0, args.min_context_only - _as_float(row, "context_only_rate"))
        if args.max_prior_only is not None:
            source_target_gap += max(0.0, _as_float(row, "prior_only_rate") - args.max_prior_only)
        if args.max_po is not None:
            source_target_gap += max(0.0, _as_float(row, "po") - args.max_po)
        if args.max_mr is not None:
            source_target_gap += max(0.0, _as_float(row, "mr") - args.max_mr)
        source_score = _source_score(row)
        output.append(
            {
                **row,
                "source_axis_score": source_score,
                "baseline_cf_em_rate": base_cf,
                "baseline_mean_output_chars": base_len,
                "baseline_neither_rate": base_neither,
                "baseline_po": base_po,
                "delta_cf_em_rate": cf_delta,
                "delta_mean_output_chars": len_delta,
                "delta_neither_rate": neither_delta,
                "delta_po": po_delta,
                "constraints_pass": constraints_pass,
                "ck_bounded_pass": ck_bounded_pass,
                "ck_violation_count": ck_violation_count,
                "source_target_gap": source_target_gap,
                "ck_check_context_only": ck_bounded_checks["context_only"],
                "ck_check_prior_only": ck_bounded_checks["prior_only"],
                "ck_check_po": ck_bounded_checks["po"],
                "ck_check_mr": ck_bounded_checks["mr"],
                "ck_check_cf_em": ck_bounded_checks["cf_em"],
                "ck_check_length": ck_bounded_checks["length"],
                "min_context_only": args.min_context_only,
                "max_prior_only": args.max_prior_only,
                "max_po": args.max_po,
                "max_mr": args.max_mr,
                "min_cf_em": args.min_cf_em,
                "max_mean_output_chars": args.max_mean_output_chars,
                "selection_objective": "maximize_source_axis_score_under_form_constraints",
                "max_cf_em_drop": args.max_cf_em_drop,
                "max_length_increase": args.max_length_increase,
                "max_neither_increase": args.max_neither_increase,
                "max_po_increase": args.max_po_increase,
            }
        )

    output.sort(
        key=lambda row: (
            not bool(row["ck_bounded_pass"]),
            int(row["ck_violation_count"]),
            float(row["source_target_gap"]),
            not bool(row["constraints_pass"]),
            -float(row["source_axis_score"]),
            -float(row.get("context_only_rate", 0.0)),
            float(row.get("mr", 0.0)),
            float(row.get("alpha", 999.0)),
            str(row.get("source_apply_mode", "")),
        )
    )
    dump_csv(args.out_csv, output)
    if output:
        best = output[0]
        print(
            "[select-rcm-source-config] best="
            f"mode={best.get('source_apply_mode', '')} alpha={best.get('alpha', '')} "
            f"source_score={float(best['source_axis_score']):.6f} "
            f"constraints_pass={best['constraints_pass']} "
            f"ck_bounded_pass={best.get('ck_bounded_pass', '')}"
        )


if __name__ == "__main__":
    main()
