from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from screscomp.data import dump_csv, load_csv


DEFAULT_SCORE_PICKS = {
    "label_context_only": [
        "composite_potential_score",
        "source_prior_consistency_score",
        "calibrated_label_context_only_score",
    ],
    "label_prior_only": ["anti_prior_suppression_score", "calibrated_label_prior_only_score"],
    "label_composite_success": [
        "composite_potential_score",
        "composite_gate_score",
        "composite_soft_product_score",
        "calibrated_label_composite_success_score",
    ],
    "label_first_answer_cf_em": [
        "form_up_score",
        "form_down_only_score",
        "form_up_down_score",
        "form_potential_score",
        "calibrated_label_first_answer_cf_em_score",
    ],
    "label_short_output": [
        "form_up_score",
        "form_down_only_score",
        "form_up_down_score",
        "calibrated_label_short_output_score",
    ],
    "label_single_commitment": ["commitment_potential_score", "calibrated_label_single_commitment_score"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect RCM prediction summaries into a compact comparison table.")
    p.add_argument(
        "--run",
        action="append",
        default=[],
        help="name=strict_csv=calibrated_csv. Example: rcm=out/strict/summary.csv=out/calibrated/summary.csv",
    )
    p.add_argument("--out_csv", type=Path, required=True)
    p.add_argument("--score_picks", default="", help="Optional target:score+score,target:score override.")
    return p.parse_args()


def _parse_runs(items: list[str]) -> list[tuple[str, Path, Path]]:
    output: list[tuple[str, Path, Path]] = []
    for item in items:
        parts = item.split("=", 2)
        if len(parts) != 3:
            raise ValueError(f"Expected name=strict_csv=calibrated_csv: {item}")
        output.append((parts[0], Path(parts[1]), Path(parts[2])))
    return output


def _parse_score_picks(raw: str) -> dict[str, list[str]]:
    picks = {key: list(value) for key, value in DEFAULT_SCORE_PICKS.items()}
    if not raw:
        return picks
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        target, scores = item.split(":", 1)
        picks[target.strip()] = [score.strip() for score in scores.split("+") if score.strip()]
    return picks


def _index(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(row["target"]), str(row["score_name"])): row for row in rows}


def _row_for(
    *,
    run_name: str,
    variant: str,
    source_row: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run": run_name,
        "variant": variant,
        "target": source_row.get("target", ""),
        "score_name": source_row.get("score_name", ""),
        "n": source_row.get("n", ""),
        "positives": source_row.get("positives", ""),
        "positive_rate": source_row.get("positive_rate", ""),
        "auroc": source_row.get("auroc", ""),
        "average_precision": source_row.get("average_precision", ""),
        "top_bottom_gap": source_row.get("top_bottom_gap", ""),
        "decile_monotonicity": source_row.get("decile_monotonicity", ""),
        "mean_score_positive": source_row.get("mean_score_positive", ""),
        "mean_score_negative": source_row.get("mean_score_negative", ""),
    }


def main() -> None:
    args = parse_args()
    picks = _parse_score_picks(args.score_picks)
    output: list[dict[str, Any]] = []
    for run_name, strict_csv, calibrated_csv in _parse_runs(args.run):
        indexed = {
            "strict": _index(load_csv(strict_csv)),
            "calibrated": _index(load_csv(calibrated_csv)),
        }
        for target, score_names in picks.items():
            for score_name in score_names:
                variant = "calibrated" if score_name.startswith("calibrated_") else "strict"
                row = indexed[variant].get((target, score_name))
                if row is not None:
                    output.append(_row_for(run_name=run_name, variant=variant, source_row=row))
    dump_csv(args.out_csv, output)
    print(f"[rcm-collect-prediction-table] rows={len(output)} out={args.out_csv}")


if __name__ == "__main__":
    main()
