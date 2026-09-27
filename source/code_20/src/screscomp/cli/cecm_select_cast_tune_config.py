from __future__ import annotations

import argparse
import csv
import json
import shlex
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Select a CAST actuator configuration from a calibration generation_summary.csv. "
            "The selector keeps the component semantics fixed and only chooses inference-time "
            "control weights/timing from already-generated candidates."
        )
    )
    p.add_argument("--summary-csv", type=Path, required=True)
    p.add_argument("--control-plan-csv", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--min-n", type=int, default=1)
    p.add_argument("--exclude-controls", type=str, default="base,baseline,strong_prompt,context_dpo")
    p.add_argument("--require-control-prefix", type=str, default="")

    p.add_argument("--min-pc", type=float, default=None)
    p.add_argument("--max-po", type=float, default=None)
    p.add_argument("--max-mr", type=float, default=None)
    p.add_argument("--min-em", type=float, default=None)
    p.add_argument("--min-context-only", type=float, default=None)
    p.add_argument(
        "--max-pc-drop-from-best",
        type=float,
        default=0.04,
        help="Guardrail against selecting a high-EM config that loses too much context-hit rate.",
    )
    p.add_argument(
        "--max-context-only-drop-from-best",
        type=float,
        default=0.04,
        help="Guardrail against selecting a config that loses too much clean context-only behavior.",
    )

    p.add_argument("--weight-em", type=float, default=2.0)
    p.add_argument("--weight-pc", type=float, default=1.0)
    p.add_argument("--weight-context-only", type=float, default=1.0)
    p.add_argument("--weight-mr", type=float, default=1.0)
    p.add_argument("--weight-po", type=float, default=0.5)
    p.add_argument("--weight-neither", type=float, default=0.2)
    p.add_argument("--weight-harm", type=float, default=0.5)
    p.add_argument("--weight-chars", type=float, default=0.05)
    p.add_argument("--min-selection-gap", type=float, default=0.02)
    p.add_argument("--min-validation-n", type=int, default=50)
    return p.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except Exception:
        return default


def _int(row: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        value = row.get(key, "")
        if value is None or str(value).strip() == "":
            return default
        return int(float(value))
    except Exception:
        return default


def _control_parts(plan_csv: Path | None) -> dict[str, str]:
    if plan_csv is None or not plan_csv.exists():
        return {}
    rows = _read_csv(plan_csv)
    return {str(row.get("control_name", "")): str(row.get("parts", "")) for row in rows}


def _passes_guards(
    row: dict[str, Any],
    *,
    args: argparse.Namespace,
    best_pc: float,
    best_context_only: float,
) -> tuple[bool, str]:
    reasons: list[str] = []
    n = _int(row, "n")
    pc = _float(row, "pc")
    po = _float(row, "po")
    mr = _float(row, "mr")
    em = _float(row, "em")
    context_only = _float(row, "context_only_rate")

    if n < args.min_n:
        reasons.append(f"n<{args.min_n}")
    if args.min_pc is not None and pc < args.min_pc:
        reasons.append(f"pc<{args.min_pc}")
    if args.max_po is not None and po > args.max_po:
        reasons.append(f"po>{args.max_po}")
    if args.max_mr is not None and mr > args.max_mr:
        reasons.append(f"mr>{args.max_mr}")
    if args.min_em is not None and em < args.min_em:
        reasons.append(f"em<{args.min_em}")
    if args.min_context_only is not None and context_only < args.min_context_only:
        reasons.append(f"context_only<{args.min_context_only}")
    if args.max_pc_drop_from_best is not None and pc < best_pc - args.max_pc_drop_from_best:
        reasons.append(f"pc_drop>{args.max_pc_drop_from_best}")
    if (
        args.max_context_only_drop_from_best is not None
        and context_only < best_context_only - args.max_context_only_drop_from_best
    ):
        reasons.append(f"context_only_drop>{args.max_context_only_drop_from_best}")
    return not reasons, ";".join(reasons)


def _score(row: dict[str, Any], *, args: argparse.Namespace) -> float:
    mean_chars = min(_float(row, "mean_chars"), 400.0) / 400.0
    return (
        args.weight_em * _float(row, "em")
        + args.weight_pc * _float(row, "pc")
        + args.weight_context_only * _float(row, "context_only_rate")
        - args.weight_mr * _float(row, "mr")
        - args.weight_po * _float(row, "po")
        - args.weight_neither * _float(row, "neither_rate")
        - args.weight_harm * _float(row, "harm_from_context_only_rate")
        - args.weight_chars * mean_chars
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    excluded = {item.strip() for item in args.exclude_controls.split(",") if item.strip()}
    rows = []
    for row in _read_csv(args.summary_csv):
        name = str(row.get("control_name", ""))
        if not name or name in excluded:
            continue
        if args.require_control_prefix and not name.startswith(args.require_control_prefix):
            continue
        rows.append(row)
    if not rows:
        raise SystemExit(f"No candidate rows found in {args.summary_csv}")

    best_pc = max(_float(row, "pc") for row in rows)
    best_context_only = max(_float(row, "context_only_rate") for row in rows)
    parts_by_name = _control_parts(args.control_plan_csv)

    scored: list[dict[str, Any]] = []
    for row in rows:
        passes, guard_reasons = _passes_guards(
            row,
            args=args,
            best_pc=best_pc,
            best_context_only=best_context_only,
        )
        enriched = dict(row)
        enriched["score"] = _score(row, args=args)
        enriched["passes_guards"] = passes
        enriched["guard_reasons"] = guard_reasons
        enriched["control_parts"] = parts_by_name.get(str(row.get("control_name", "")), "")
        scored.append(enriched)

    passing = [row for row in scored if bool(row["passes_guards"])]
    pool = passing or scored
    selected_by = "guarded_score" if passing else "fallback_score_no_guarded_candidate"
    ranked = sorted(pool, key=lambda row: (_float(row, "score"), _float(row, "em"), _float(row, "pc")), reverse=True)
    best = ranked[0]
    all_ranked = sorted(scored, key=lambda row: (_float(row, "score"), _float(row, "em"), _float(row, "pc")), reverse=True)
    second_score = _float(ranked[1], "score") if len(ranked) > 1 else float("-inf")
    selection_gap = _float(best, "score") - second_score if second_score != float("-inf") else float("inf")
    needs_more_validation = _int(best, "n") < args.min_validation_n or selection_gap < args.min_selection_gap

    best_name = str(best["control_name"])
    best_parts = str(best.get("control_parts", ""))
    best_payload = {
        "selected_by": selected_by,
        "summary_csv": str(args.summary_csv),
        "control_plan_csv": str(args.control_plan_csv) if args.control_plan_csv else "",
        "best_control_name": best_name,
        "best_control_parts": best_parts,
        "cast_controls": f"{best_name}={best_parts}" if best_parts else best_name,
        "selection_gap": selection_gap,
        "needs_more_validation": needs_more_validation,
        "best_row": best,
        "guardrails": {
            "min_n": args.min_n,
            "min_pc": args.min_pc,
            "max_po": args.max_po,
            "max_mr": args.max_mr,
            "min_em": args.min_em,
            "min_context_only": args.min_context_only,
            "max_pc_drop_from_best": args.max_pc_drop_from_best,
            "max_context_only_drop_from_best": args.max_context_only_drop_from_best,
            "best_pc_among_candidates": best_pc,
            "best_context_only_among_candidates": best_context_only,
        },
        "weights": {
            "em": args.weight_em,
            "pc": args.weight_pc,
            "context_only": args.weight_context_only,
            "mr": args.weight_mr,
            "po": args.weight_po,
            "neither": args.weight_neither,
            "harm": args.weight_harm,
            "chars": args.weight_chars,
        },
    }

    (args.out_dir / "best_config.json").write_text(json.dumps(best_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    env_lines = [
        f"BEST_CAST_CONTROL_NAME={shlex.quote(best_name)}",
        f"BEST_CAST_CONTROL_PARTS={shlex.quote(best_parts)}",
        f"BEST_CAST_SELECTED_BY={shlex.quote(selected_by)}",
        f"CAST_TUNE_SELECTION_GAP={shlex.quote(f'{selection_gap:.8f}')}",
        f"CAST_TUNE_NEEDS_MORE_VALIDATION={'1' if needs_more_validation else '0'}",
    ]
    if best_parts:
        env_lines.append(f"CAST_CONTROLS={shlex.quote(best_name + '=' + best_parts)}")
    (args.out_dir / "best_config.env").write_text("\n".join(env_lines) + "\n", encoding="utf-8")
    _write_csv(args.out_dir / "selection_ranked.csv", all_ranked)

    print(
        "[cast-tune-select] "
        f"selected={best_name} score={_float(best, 'score'):.6f} "
        f"pc={_float(best, 'pc'):.4f} po={_float(best, 'po'):.4f} "
        f"mr={_float(best, 'mr'):.4f} em={_float(best, 'em'):.4f} "
        f"context_only={_float(best, 'context_only_rate'):.4f} selected_by={selected_by}",
        flush=True,
    )


if __name__ == "__main__":
    main()
