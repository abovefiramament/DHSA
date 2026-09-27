#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


ALPHA_RE = re.compile(r"^(?P<control>.+)_a(?P<alpha>[0-9]+(?:p[0-9]+)?)$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Select one fixed alpha per site-control from an alpha-dev open-generation "
            "summary. The output TSV can be passed to 10_run_pre_o_head_site_open_generation.sh "
            "as ALPHA_PLAN_TSV for held-out evaluation."
        )
    )
    p.add_argument("--summary-tsv", type=Path, required=True)
    p.add_argument("--out-tsv", type=Path, required=True)
    p.add_argument(
        "--confiqa-score",
        choices=("logic_score", "source_score"),
        default="logic_score",
        help="Primary ConFiQA selection score. PC and EM are reporting metrics, not selectors.",
    )
    return p.parse_args()


def as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in {"", None}:
        return default
    return float(value)


def parse_control_alpha(control_name: str) -> tuple[str, str] | None:
    match = ALPHA_RE.match(control_name)
    if not match:
        return None
    alpha = match.group("alpha").replace("p", ".")
    return match.group("control"), alpha


def confiqa_logic_score(row: dict[str, Any]) -> float:
    if row.get("logic_score", "") != "":
        return as_float(row, "logic_score")
    if row.get("score", "") != "":
        return as_float(row, "score")
    missing = [
        key
        for key in ("context_only", "short", "po", "neither", "mean_chars")
        if row.get(key, "") == ""
    ]
    if missing:
        raise ValueError(
            "ConFiQA rows must contain logic_score or enough fields to reconstruct it; "
            f"missing={missing} control_name={row.get('control_name', '')}"
        )
    return (
        as_float(row, "context_only")
        + as_float(row, "short")
        - as_float(row, "po")
        - as_float(row, "neither")
        - as_float(row, "mean_chars") / 400.0
    )


def confiqa_source_score(row: dict[str, Any]) -> float:
    missing = [key for key in ("context_only", "prior_only", "po") if row.get(key, "") == ""]
    if missing:
        raise ValueError(
            "source_score requires context_only, prior_only, and po fields; "
            f"missing={missing} control_name={row.get('control_name', '')}"
        )
    return as_float(row, "context_only") - as_float(row, "prior_only") - as_float(row, "po")


def row_score(row: dict[str, Any], *, confiqa_score: str) -> tuple[str, float]:
    task = str(row.get("task", ""))
    if task == "imdb":
        return "sentiment_score", as_float(row, "score")
    if task.startswith("confiqa_"):
        if confiqa_score == "source_score":
            return "source_score", confiqa_source_score(row)
        return "logic_score", confiqa_logic_score(row)
    raise ValueError(f"Unsupported task in alpha summary: {task}")


def sort_key(row: dict[str, Any], *, score_name: str) -> tuple[float, ...]:
    alpha = float(row["alpha"])
    if row["task"] == "imdb":
        return (
            -float(row["selection_score"]),
            as_float(row, "kl", default=1e9),
            -as_float(row, "positive_rate"),
            alpha,
        )
    if score_name == "source_score":
        return (
            -float(row["selection_score"]),
            -as_float(row, "pc"),
            as_float(row, "po", default=1e9),
            alpha,
        )
    return (
        -float(row["selection_score"]),
        -as_float(row, "em"),
        -as_float(row, "pc"),
        as_float(row, "po", default=1e9),
        as_float(row, "mean_chars", default=1e9),
        alpha,
    )


def main() -> None:
    args = parse_args()
    with args.summary_tsv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    enriched: list[dict[str, Any]] = []
    for row in rows:
        parsed = parse_control_alpha(str(row.get("control_name", "")))
        if parsed is None:
            continue
        control, alpha = parsed
        if float(alpha) <= 0.0:
            continue
        score_name, score = row_score(row, confiqa_score=args.confiqa_score)
        new_row = {
            **row,
            "control": control,
            "alpha": alpha,
            "selection_score_name": score_name,
            "selection_score": f"{score:.10g}",
        }
        grouped[(str(row.get("task", "")), control)].append(new_row)
        enriched.append(new_row)

    output: list[dict[str, Any]] = []
    for (_task, _control), group in sorted(grouped.items()):
        if not group:
            continue
        score_name = str(group[0]["selection_score_name"])
        group.sort(key=lambda row: sort_key(row, score_name=score_name))
        output.append(group[0])

    if not output:
        raise SystemExit(f"No alpha rows selected from {args.summary_tsv}")

    fields = [
        "task",
        "control",
        "alpha",
        "selection_score_name",
        "selection_score",
        "control_name",
        "n",
        "score",
        "kl",
        "positive_rate",
        "pc",
        "em",
        "short",
        "po",
        "mean_chars",
        "logic_score",
        "context_only",
        "prior_only",
        "both",
        "neither",
        "mr",
        "out_dir",
    ]
    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_tsv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output)

    for row in output:
        print(
            "[site-alpha] "
            f"{row['task']}/{row['control']} alpha={row['alpha']} "
            f"{row['selection_score_name']}={row['selection_score']} "
            f"pc={row.get('pc', '')} em={row.get('em', '')} kl={row.get('kl', '')}"
        )


if __name__ == "__main__":
    main()
