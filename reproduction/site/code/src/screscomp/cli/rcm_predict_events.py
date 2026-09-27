from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from screscomp.data import dump_csv, load_jsonl
from screscomp.rcm.prediction import summarize_prediction_scores


DEFAULT_TARGET_SCORES = {
    "context_only": ["ci_score_context", "margin_score_context"],
    "prior_only": ["ci_score_prior", "margin_score_prior"],
    "cf_hit": ["ci_score_context", "margin_score_context"],
    "orig_hit": ["ci_score_prior", "margin_score_prior"],
    "failure_not_context_only": ["ci_score_prior", "ci_score_both", "ci_score_neither", "margin_score_prior"],
}


def _parse_target_scores(raw: str) -> dict[str, list[str]]:
    if not raw:
        return DEFAULT_TARGET_SCORES
    output: dict[str, list[str]] = {}
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Expected target=score,score item: {item}")
        target, scores = item.split("=", 1)
        output[target.strip()] = [score.strip() for score in scores.split(",") if score.strip()]
    return output


def _derive_default_labels(row: dict[str, Any]) -> dict[str, Any]:
    output = dict(row)
    outcome = str(output.get("outcome", ""))
    output.setdefault("context_only", outcome == "context_only")
    output.setdefault("prior_only", outcome == "prior_only")
    output.setdefault("both", outcome == "both")
    output.setdefault("neither", outcome == "neither")
    output.setdefault("failure_not_context_only", outcome != "context_only")
    return output


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize event prediction scores with AUROC and average precision.")
    p.add_argument("--predictions_jsonl", type=Path, required=True)
    p.add_argument("--out_summary_csv", type=Path, required=True)
    p.add_argument(
        "--target_scores",
        default="",
        help=(
            "Semicolon-separated mapping target=score,score. "
            "Empty uses CK/source defaults when the columns are present."
        ),
    )
    p.add_argument("--no_derive_default_labels", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.predictions_jsonl)
    if not args.no_derive_default_labels:
        rows = [_derive_default_labels(row) for row in rows]
    target_scores = _parse_target_scores(args.target_scores)
    available_target_scores = {
        target: [score for score in scores if any(score in row for row in rows)]
        for target, scores in target_scores.items()
        if any(target in row for row in rows)
    }
    metrics = summarize_prediction_scores(rows, target_scores=available_target_scores)
    dump_csv(args.out_summary_csv, [metric.to_dict() for metric in metrics])
    print(
        f"[rcm-predict-events] rows={len(rows)} targets={','.join(sorted(available_target_scores))} "
        f"metrics={len(metrics)} out={args.out_summary_csv}"
    )


if __name__ == "__main__":
    main()
