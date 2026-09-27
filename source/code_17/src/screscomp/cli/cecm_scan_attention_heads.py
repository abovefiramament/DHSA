from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from screscomp.cli.cecm_run_attention_head_refine import (
    AttentionHeadScalingRunner,
    _cache_baselines,
    _load_scan_pairs,
    _scan_heads,
)
from screscomp.cecm.objective import OPTION_SELECTION_MODES, option_selection_description
from screscomp.data import dump_csv, dump_json
from screscomp.modeling import TransformersABBackend


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Pair-only attention-head scan. It probes head scaling on the same y_plus/y_minus margin "
            "used by component discovery and actuator training."
        )
    )
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--pairs-csv", type=Path, required=True)
    p.add_argument("--event", type=str, default="step_correct_over_error")
    p.add_argument("--attn-layers", type=str, required=True)
    p.add_argument("--scan-factors", type=str, default="0.0,0.5,1.5")
    p.add_argument("--topk-heads", type=int, default=8)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--scan-start", type=int, default=0)
    p.add_argument("--scan-max-rows", type=int, default=24)
    p.add_argument(
        "--score-mode",
        type=str,
        default="avglogp",
        choices=["avglogp", "top_logit_gap", "answer_rest_margin"],
    )
    p.add_argument(
        "--score-apply-mode",
        type=str,
        default="decision_tokens",
        help=(
            "Score-time probe timing. Supports decision_tokens, boxed_decision, prompt_last, prompt, "
            "all, prefill, decode, first_decode, first_N_decode."
        ),
    )
    p.add_argument("--max-aliases-per-side", type=int, default=1)
    p.add_argument(
        "--option-selection-mode",
        type=str,
        default="model_max",
        choices=OPTION_SELECTION_MODES,
        help="How to choose among multiple continuation options for each endpoint.",
    )
    p.add_argument("--min-mean-margin-gain", type=float, default=0.0)
    p.add_argument(
        "--require-directional-ci",
        action="store_true",
        help="Require selected heads to have gain_ci95_low > 0.",
    )
    p.add_argument("--allow-nonpositive-selection", action="store_true")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--use-chat-template", action="store_true")
    p.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
    )
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args(argv)


def _parse_ints(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("--attn-layers is empty")
    return values


def _parse_floats(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("--scan-factors is empty")
    return values


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        raw = str(row.get(key, "")).strip()
        return float(raw) if raw else default
    except Exception:
        return default


def _select_heads(
    scan_rows: list[dict[str, object]],
    *,
    topk: int,
    min_mean_margin_gain: float,
    require_directional_ci: bool,
    allow_nonpositive: bool,
    role: str | None = None,
) -> tuple[list[dict[str, object]], str]:
    best_by_head: dict[str, dict[str, object]] = {}
    for row in scan_rows:
        if role is not None and str(row.get("role", "")) != role:
            continue
        head_id = str(row.get("head_id", ""))
        if not head_id:
            continue
        previous = best_by_head.get(head_id)
        if previous is None or _float(row, "mean_margin_gain") > _float(previous, "mean_margin_gain"):
            best_by_head[head_id] = row
    sorted_rows = sorted(
        best_by_head.values(),
        key=lambda row: (_float(row, "mean_margin_gain"), _float(row, "gain_positive_rate")),
        reverse=True,
    )
    positive = [
        row
        for row in sorted_rows
        if _float(row, "mean_margin_gain") >= min_mean_margin_gain
        and (not require_directional_ci or _float(row, "gain_ci95_low") > 0)
    ]
    if positive:
        suffix = "_ci_excludes_zero" if require_directional_ci else ""
        return positive[:topk], f"top_{role or 'any'}_positive_margin_gain{suffix}"
    if allow_nonpositive:
        return sorted_rows[:topk], f"fallback_top_{role or 'any'}_available_margin_gain"
    return [], f"no_{role or 'any'}_positive_margin_gain"


def _ranked_rows(rows: list[dict[str, object]], *, policy: str) -> list[dict[str, object]]:
    return [
        {
            "rank": rank,
            "selection_policy": policy,
            **row,
        }
        for rank, row in enumerate(rows, start=1)
    ]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    attn_layers = _parse_ints(args.attn_layers)
    factors = _parse_floats(args.scan_factors)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[cecm-scan-heads] loading model={args.model} layers={attn_layers}", flush=True)
    backend = TransformersABBackend(
        model_name_or_path=args.model,
        device=args.device,
        use_chat_template=args.use_chat_template,
        torch_dtype=args.torch_dtype,
    )
    runner = AttentionHeadScalingRunner(
        backend=backend,
        score_mode=args.score_mode,
        score_apply_mode=args.score_apply_mode,
        max_aliases_per_side=args.max_aliases_per_side,
        option_selection_mode=args.option_selection_mode,
    )
    pairs = _load_scan_pairs(
        args.pairs_csv,
        event=args.event,
        split=args.split,
        start=args.scan_start,
        max_rows=args.scan_max_rows,
    )
    dump_json(
        args.out_dir / "run_config.json",
        {
            "model": args.model,
            "pairs_csv": str(args.pairs_csv),
            "event": args.event,
            "attn_layers": attn_layers,
            "scan_factors": factors,
            "topk_heads": args.topk_heads,
            "split": args.split,
            "scan_start": args.scan_start,
            "scan_max_rows": args.scan_max_rows,
            "scan_pairs": len(pairs),
            "score_mode": args.score_mode,
            "score_apply_mode": args.score_apply_mode,
            "max_aliases_per_side": args.max_aliases_per_side,
            "option_selection_mode": args.option_selection_mode,
            "option_selection": option_selection_description(args.option_selection_mode),
            "min_mean_margin_gain": args.min_mean_margin_gain,
            "require_directional_ci": args.require_directional_ci,
            "allow_nonpositive_selection": bool(args.allow_nonpositive_selection),
            "competitive_margin": (
                "C(M;x)=S_M(y_plus|x)-S_M(y_minus|x) when y_minus options are present; "
                "when y_minus_mode=dynamic_max_non_gold_logic, y_minus is the strongest non-gold "
                "answer logic recomputed from the current forward logits."
            ),
            "probe_operator": "scale one attention head slice before the layer output projection",
        },
    )
    baselines = _cache_baselines(runner, pairs)
    scan_rows = _scan_heads(runner, pairs, baselines, attn_layers=attn_layers, factors=factors)
    selected_suppress, suppress_policy = _select_heads(
        scan_rows,
        topk=args.topk_heads,
        min_mean_margin_gain=args.min_mean_margin_gain,
        require_directional_ci=args.require_directional_ci,
        allow_nonpositive=args.allow_nonpositive_selection,
        role="suppress",
    )
    selected_boost, boost_policy = _select_heads(
        scan_rows,
        topk=args.topk_heads,
        min_mean_margin_gain=args.min_mean_margin_gain,
        require_directional_ci=args.require_directional_ci,
        allow_nonpositive=args.allow_nonpositive_selection,
        role="boost",
    )
    selected = selected_suppress + [
        row for row in selected_boost if str(row.get("head_id", "")) not in {str(item.get("head_id", "")) for item in selected_suppress}
    ]
    if args.topk_heads > 0 and not selected:
        raise SystemExit(
            "No heads selected with positive mean_margin_gain. "
            "Pass --allow-nonpositive-selection for an exploratory fallback."
        )

    selected_suppress_rows = _ranked_rows(selected_suppress, policy=suppress_policy)
    selected_boost_rows = _ranked_rows(selected_boost, policy=boost_policy)
    selected_rows = _ranked_rows(selected, policy=f"suppress:{suppress_policy};boost:{boost_policy}")
    dump_csv(args.out_dir / "head_scan.csv", scan_rows)
    dump_csv(args.out_dir / "selected_heads.csv", selected_rows)
    dump_csv(args.out_dir / "selected_suppress_heads.csv", selected_suppress_rows)
    dump_csv(args.out_dir / "selected_boost_heads.csv", selected_boost_rows)
    (args.out_dir / "selected_heads.txt").write_text(
        ",".join(str(row["head_id"]) for row in selected),
        encoding="utf-8",
    )
    (args.out_dir / "selected_suppress_heads.txt").write_text(
        ",".join(str(row["head_id"]) for row in selected_suppress),
        encoding="utf-8",
    )
    (args.out_dir / "selected_boost_heads.txt").write_text(
        ",".join(str(row["head_id"]) for row in selected_boost),
        encoding="utf-8",
    )
    dump_json(
        args.out_dir / "head_selection_manifest.json",
        {
            "event": args.event,
            "selection_policy": {
                "suppress": suppress_policy,
                "boost": boost_policy,
                "combined": "union of suppress and boost role selections",
            },
            "selected_heads": [str(row["head_id"]) for row in selected],
            "selected_suppress_heads": [str(row["head_id"]) for row in selected_suppress],
            "selected_boost_heads": [str(row["head_id"]) for row in selected_boost],
            "min_mean_margin_gain": args.min_mean_margin_gain,
            "require_directional_ci": args.require_directional_ci,
            "allow_nonpositive_selection": bool(args.allow_nonpositive_selection),
            "semantics": (
                "Head selection is a probe-stage component discovery step after attention layers "
                "are separated from the component pool. suppress heads are those whose attenuation "
                "improves the logic competition advantage; boost heads are those whose amplification "
                "improves it. Training remains vector-based and uses the same competitive margin."
            ),
        },
    )
    dump_csv(
        args.out_dir / "head_selection_summary.csv",
        [
            {"metric": "event", "value": args.event},
            {"metric": "scan_pairs", "value": len(pairs)},
            {"metric": "scan_rows", "value": len(scan_rows)},
            {"metric": "selected_heads", "value": len(selected)},
            {"metric": "selected_suppress_heads", "value": len(selected_suppress)},
            {"metric": "selected_boost_heads", "value": len(selected_boost)},
            {"metric": "suppress_policy", "value": suppress_policy},
            {"metric": "boost_policy", "value": boost_policy},
            {"metric": "require_directional_ci", "value": int(args.require_directional_ci)},
        ],
    )
    print(
        (
            f"[cecm-scan-heads] suppress={','.join(str(row['head_id']) for row in selected_suppress)} "
            f"boost={','.join(str(row['head_id']) for row in selected_boost)} out={args.out_dir}"
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
