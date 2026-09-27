from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from screscomp.data import dump_csv, dump_json, load_csv


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Select a simple component pool from component_screen.csv using signed competitive-margin deltas. "
            "This is a light-weight selector for new objectives that do not yet have a multi-clause spec."
        )
    )
    p.add_argument("--component-screen-csv", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--event", type=str, default="step_correct_over_error")
    p.add_argument("--topk-mlp", type=int, default=4)
    p.add_argument(
        "--topk-negative-mlp",
        type=int,
        default=None,
        help="How many negative-mean MLP components to write to mlp_negative_components.csv. Defaults to --topk-mlp.",
    )
    p.add_argument("--topk-attn-layers", type=int, default=4)
    p.add_argument(
        "--mlp-policy",
        choices=["positive", "negative", "absolute"],
        default="positive",
        help="positive means components whose zero-ablation reduces the target-vs-rejected margin.",
    )
    p.add_argument("--min-sign-consistency", type=float, default=0.0)
    p.add_argument(
        "--mlp-min-sign-consistency",
        type=float,
        default=None,
        help="Optional MLP-specific sign-consistency threshold. Defaults to --min-sign-consistency.",
    )
    p.add_argument(
        "--attn-min-sign-consistency",
        type=float,
        default=None,
        help="Optional attention-specific sign-consistency threshold. Defaults to --min-sign-consistency.",
    )
    p.add_argument(
        "--require-directional-ci",
        action="store_true",
        help="Require CI95 to stay on the selected direction: low>0 for positive, high<0 for negative.",
    )
    p.add_argument(
        "--mlp-require-directional-ci",
        action="store_true",
        help="Require directional CI only for MLP selection. If omitted, falls back to --require-directional-ci.",
    )
    p.add_argument(
        "--mlp-allow-ci-cross-zero",
        action="store_true",
        help="Allow MLP CI95 to cross zero even when --require-directional-ci is set.",
    )
    p.add_argument(
        "--attn-require-directional-ci",
        action="store_true",
        help="Require directional CI only for attention selection. If omitted, falls back to --require-directional-ci.",
    )
    p.add_argument(
        "--attn-allow-ci-cross-zero",
        action="store_true",
        help="Allow attention CI95 to cross zero even when --require-directional-ci is set.",
    )
    p.add_argument(
        "--min-mlp-abs-mean-delta",
        type=float,
        default=0.0,
        help="Minimum |mean_delta| for selected MLP components. Kept separate because MLP may be weaker.",
    )
    p.add_argument(
        "--min-attn-abs-mean-delta",
        type=float,
        default=0.0,
        help="Minimum |mean_delta| for selected attention layer components.",
    )
    p.add_argument(
        "--min-bidirectional-rate",
        type=float,
        default=0.0,
        help=(
            "Optional rollout-scan filter. When component_screen.csv has bidirectional_rate, require at least this "
            "fraction of prompts to satisfy target_drop>0 and source_rise>0."
        ),
    )
    p.add_argument(
        "--min-directional-transition-rate",
        type=float,
        default=None,
        help=(
            "Rollout-scan filter that respects the selected sign. Positive components use positive_transition_rate; "
            "negative components use negative_transition_rate. Defaults to --min-bidirectional-rate."
        ),
    )
    p.add_argument(
        "--min-ablated-format-ok",
        type=float,
        default=0.0,
        help="When format audit columns exist, require this mean ablated format-ok rate.",
    )
    p.add_argument(
        "--max-format-collapse-rate",
        type=float,
        default=1.0,
        help="When format audit columns exist, reject components above this full-ok to ablated-bad collapse rate.",
    )
    p.add_argument(
        "--min-abs-mean-target-drop",
        type=float,
        default=0.0,
        help="When rollout columns exist, require |mean_target_drop| at least this value.",
    )
    p.add_argument(
        "--min-abs-mean-source-rise",
        type=float,
        default=0.0,
        help="When rollout columns exist, require |mean_source_rise| at least this value.",
    )
    p.add_argument("--exclude-mlp-layers", type=str, default="")
    p.add_argument("--exclude-attn-layers", type=str, default="")
    p.add_argument("--allow-empty-attn", action="store_true")
    return p.parse_args(argv)


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        raw = str(row.get(key, "")).strip()
        return float(raw) if raw else default
    except Exception:
        return default


def _int(row: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        raw = str(row.get(key, "")).strip()
        return int(float(raw)) if raw else default
    except Exception:
        return default


def _parse_excluded(raw: str) -> set[int]:
    out: set[int] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        out.add(int(item))
    return out


def _component_row(row: dict[str, Any], *, rank: int, role: str, policy: str) -> dict[str, object]:
    out = {
        "component_id": row["component_id"],
        "layer_idx": row["layer_idx"],
        "component_type": row["component_type"],
        "rank": rank,
        "selection_role": role,
        "selection_policy": policy,
        "mean_delta": row.get("mean_delta", ""),
        "abs_mean_delta": row.get("abs_mean_delta", ""),
        "sign_consistency": row.get("sign_consistency", ""),
        "ci95_low": row.get("ci95_low", ""),
        "ci95_high": row.get("ci95_high", ""),
    }
    for key in (
        "positive_mass",
        "negative_mass",
        "mass_gap",
        "positive_active_mean",
        "negative_active_mean",
        "source_group_count",
        "group_mean_delta",
        "group_positive_rate",
        "group_negative_rate",
        "group_positive_mass",
        "group_negative_mass",
        "group_mass_gap",
        "group_positive_active_mean",
        "group_negative_active_mean",
        "group_sign_consistency",
        "support_direction",
        "mean_target_drop",
        "mean_source_rise",
        "bidirectional_rate",
        "positive_transition_rate",
        "negative_transition_rate",
        "directional_transition_rate",
        "mean_full_target_score",
        "mean_full_source_score",
        "mean_ablated_target_score",
        "mean_ablated_source_score",
        "state_transition_score",
        "mean_full_format_ok",
        "mean_ablated_format_ok",
        "format_collapse_rate",
    ):
        if key in row:
            out[key] = row.get(key, "")
    return out


def _finite_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = _float(row, key, default)
    return default if value != value else value


def _directional_mass(row: dict[str, Any], direction: str) -> float:
    if direction == "positive":
        if "group_positive_mass" in row:
            return _finite_float(row, "group_positive_mass")
        if "positive_mass" in row:
            return _finite_float(row, "positive_mass")
        return max(_finite_float(row, "mean_delta"), 0.0)
    if "group_negative_mass" in row:
        return _finite_float(row, "group_negative_mass")
    if "negative_mass" in row:
        return _finite_float(row, "negative_mass")
    return max(-_finite_float(row, "mean_delta"), 0.0)


def _directional_rate(row: dict[str, Any], direction: str) -> float:
    if direction == "positive":
        if "group_positive_rate" in row:
            return _finite_float(row, "group_positive_rate")
        if "positive_rate" in row:
            return _finite_float(row, "positive_rate")
        return 0.0
    if "group_negative_rate" in row:
        return _finite_float(row, "group_negative_rate")
    if "negative_rate" in row:
        return _finite_float(row, "negative_rate")
    return 0.0


def _directional_active_mean(row: dict[str, Any], direction: str) -> float:
    if direction == "positive":
        if "group_positive_active_mean" in row:
            return _finite_float(row, "group_positive_active_mean")
        if "positive_active_mean" in row:
            return _finite_float(row, "positive_active_mean")
        return max(_finite_float(row, "mean_delta"), 0.0)
    if "group_negative_active_mean" in row:
        return _finite_float(row, "group_negative_active_mean")
    if "negative_active_mean" in row:
        return _finite_float(row, "negative_active_mean")
    return max(-_finite_float(row, "mean_delta"), 0.0)


def _direction_ok(row: dict[str, Any], direction: str, *, require_directional_ci: bool, min_abs_mean_delta: float) -> bool:
    mean_delta = _float(row, "mean_delta")
    if direction == "positive":
        if mean_delta <= 0:
            return False
        if min_abs_mean_delta > 0 and mean_delta < min_abs_mean_delta:
            return False
        if require_directional_ci and _float(row, "ci95_low") <= 0:
            return False
        return True
    if direction == "negative":
        if mean_delta >= 0:
            return False
        if min_abs_mean_delta > 0 and abs(mean_delta) < min_abs_mean_delta:
            return False
        if require_directional_ci and _float(row, "ci95_high") >= 0:
            return False
        return True
    if mean_delta == 0:
        return False
    if min_abs_mean_delta > 0 and abs(mean_delta) < min_abs_mean_delta:
        return False
    if require_directional_ci:
        if mean_delta > 0 and _float(row, "ci95_low") <= 0:
            return False
        if mean_delta < 0 and _float(row, "ci95_high") >= 0:
            return False
    return True


def _directional_transition_rate(row: dict[str, Any], direction: str) -> float:
    if "directional_transition_rate" in row:
        return _float(row, "directional_transition_rate")
    if direction == "negative" and "negative_transition_rate" in row:
        return _float(row, "negative_transition_rate")
    if direction == "positive" and "positive_transition_rate" in row:
        return _float(row, "positive_transition_rate")
    if direction == "negative" and "group_negative_rate" in row:
        return _finite_float(row, "group_negative_rate")
    if direction == "positive" and "group_positive_rate" in row:
        return _finite_float(row, "group_positive_rate")
    if direction == "negative" and "negative_rate" in row:
        return _finite_float(row, "negative_rate")
    if direction == "positive" and "positive_rate" in row:
        return _finite_float(row, "positive_rate")
    if "bidirectional_rate" in row:
        return _float(row, "bidirectional_rate")
    return 1.0


def _quality_ok(
    row: dict[str, Any],
    *,
    direction: str,
    min_directional_transition_rate: float,
    min_ablated_format_ok: float,
    max_format_collapse_rate: float,
    min_abs_mean_target_drop: float,
    min_abs_mean_source_rise: float,
) -> bool:
    if _directional_transition_rate(row, direction) < min_directional_transition_rate:
        return False
    if "mean_ablated_format_ok" in row and _float(row, "mean_ablated_format_ok") < min_ablated_format_ok:
        return False
    if "format_collapse_rate" in row and _float(row, "format_collapse_rate") > max_format_collapse_rate:
        return False
    if "mean_target_drop" in row and abs(_float(row, "mean_target_drop")) < min_abs_mean_target_drop:
        return False
    if "mean_source_rise" in row and abs(_float(row, "mean_source_rise")) < min_abs_mean_source_rise:
        return False
    return True


def _select_mlp(
    rows: list[dict[str, str]],
    *,
    policy: str,
    topk: int,
    min_sign_consistency: float,
    require_directional_ci: bool,
    min_abs_mean_delta: float,
    excluded_layers: set[int],
    min_directional_transition_rate: float,
    min_ablated_format_ok: float,
    max_format_collapse_rate: float,
    min_abs_mean_target_drop: float,
    min_abs_mean_source_rise: float,
) -> tuple[list[dict[str, str]], str]:
    mlp_rows = [
        row
        for row in rows
        if row.get("component_type") == "mlp" and _float(row, "sign_consistency") >= min_sign_consistency
        and _int(row, "layer_idx") not in excluded_layers
    ]
    if not mlp_rows or topk <= 0:
        return [], policy
    if policy == "positive":
        signed = [
            row
            for row in mlp_rows
            if _direction_ok(
                row,
                "positive",
                require_directional_ci=require_directional_ci,
                min_abs_mean_delta=min_abs_mean_delta,
            )
            and _quality_ok(
                row,
                direction="positive",
                min_directional_transition_rate=min_directional_transition_rate,
                min_ablated_format_ok=min_ablated_format_ok,
                max_format_collapse_rate=max_format_collapse_rate,
                min_abs_mean_target_drop=min_abs_mean_target_drop,
                min_abs_mean_source_rise=min_abs_mean_source_rise,
            )
        ]
        if signed:
            return sorted(
                signed,
                key=lambda row: (
                    _directional_mass(row, "positive"),
                    _directional_rate(row, "positive"),
                    _directional_active_mean(row, "positive"),
                    _float(row, "mean_delta"),
                    _float(row, "sign_consistency"),
                ),
                reverse=True,
            )[:topk], policy
    if policy == "negative":
        signed = [
            row
            for row in mlp_rows
            if _direction_ok(
                row,
                "negative",
                require_directional_ci=require_directional_ci,
                min_abs_mean_delta=min_abs_mean_delta,
            )
            and _quality_ok(
                row,
                direction="negative",
                min_directional_transition_rate=min_directional_transition_rate,
                min_ablated_format_ok=min_ablated_format_ok,
                max_format_collapse_rate=max_format_collapse_rate,
                min_abs_mean_target_drop=min_abs_mean_target_drop,
                min_abs_mean_source_rise=min_abs_mean_source_rise,
            )
        ]
        if signed:
            return sorted(
                signed,
                key=lambda row: (
                    _directional_mass(row, "negative"),
                    _directional_rate(row, "negative"),
                    _directional_active_mean(row, "negative"),
                    -_float(row, "mean_delta"),
                    _float(row, "sign_consistency"),
                ),
                reverse=True,
            )[:topk], policy
    if require_directional_ci or min_abs_mean_delta > 0:
        return [], f"{policy}_strict_empty"
    fallback = sorted(
        mlp_rows,
        key=lambda row: (_float(row, "abs_mean_delta"), _float(row, "sign_consistency")),
        reverse=True,
    )[:topk]
    return fallback, f"{policy}_fallback_absolute"


def _select_attn_layers(
    rows: list[dict[str, str]],
    *,
    topk: int,
    min_sign_consistency: float,
    require_directional_ci: bool,
    min_abs_mean_delta: float,
    excluded_layers: set[int],
    min_directional_transition_rate: float,
    min_ablated_format_ok: float,
    max_format_collapse_rate: float,
    min_abs_mean_target_drop: float,
    min_abs_mean_source_rise: float,
) -> list[dict[str, str]]:
    if topk <= 0:
        return []
    attn_rows = [
        row
        for row in rows
        if row.get("component_type") == "attn"
        and _float(row, "sign_consistency") >= min_sign_consistency
        and _direction_ok(
            row,
            "absolute",
            require_directional_ci=require_directional_ci,
            min_abs_mean_delta=min_abs_mean_delta,
        )
        and _quality_ok(
            row,
            direction="positive" if _float(row, "mean_delta") >= 0 else "negative",
            min_directional_transition_rate=min_directional_transition_rate,
            min_ablated_format_ok=min_ablated_format_ok,
            max_format_collapse_rate=max_format_collapse_rate,
            min_abs_mean_target_drop=min_abs_mean_target_drop,
            min_abs_mean_source_rise=min_abs_mean_source_rise,
        )
        and _int(row, "layer_idx") not in excluded_layers
    ]
    selected: list[dict[str, str]] = []
    seen: set[int] = set()
    for row in sorted(
        attn_rows,
        key=lambda item: (
            _directional_mass(item, "positive" if _float(item, "mean_delta") >= 0 else "negative"),
            _directional_rate(item, "positive" if _float(item, "mean_delta") >= 0 else "negative"),
            _float(item, "abs_mean_delta"),
            _float(item, "sign_consistency"),
        ),
        reverse=True,
    ):
        layer_idx = _int(row, "layer_idx")
        if layer_idx in seen:
            continue
        seen.add(layer_idx)
        selected.append(row)
        if len(selected) >= topk:
            break
    return selected


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rows = load_csv(args.component_screen_csv)
    if not rows:
        raise SystemExit(f"No component rows found: {args.component_screen_csv}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    mlp_min_sign_consistency = (
        args.min_sign_consistency if args.mlp_min_sign_consistency is None else args.mlp_min_sign_consistency
    )
    attn_min_sign_consistency = (
        args.min_sign_consistency if args.attn_min_sign_consistency is None else args.attn_min_sign_consistency
    )
    mlp_require_directional_ci = args.require_directional_ci or args.mlp_require_directional_ci
    attn_require_directional_ci = args.require_directional_ci or args.attn_require_directional_ci
    if args.mlp_allow_ci_cross_zero:
        mlp_require_directional_ci = False
    if args.attn_allow_ci_cross_zero:
        attn_require_directional_ci = False
    min_directional_transition_rate = (
        args.min_bidirectional_rate
        if args.min_directional_transition_rate is None
        else args.min_directional_transition_rate
    )

    excluded_mlp_layers = _parse_excluded(args.exclude_mlp_layers)
    selected_mlp, actual_mlp_policy = _select_mlp(
        rows,
        policy=args.mlp_policy,
        topk=args.topk_mlp,
        min_sign_consistency=mlp_min_sign_consistency,
        require_directional_ci=mlp_require_directional_ci,
        min_abs_mean_delta=args.min_mlp_abs_mean_delta,
        excluded_layers=excluded_mlp_layers,
        min_directional_transition_rate=min_directional_transition_rate,
        min_ablated_format_ok=args.min_ablated_format_ok,
        max_format_collapse_rate=args.max_format_collapse_rate,
        min_abs_mean_target_drop=args.min_abs_mean_target_drop,
        min_abs_mean_source_rise=args.min_abs_mean_source_rise,
    )
    if args.topk_mlp > 0 and not selected_mlp:
        print(f"WARNING: No MLP components passed quality filters (topk={args.topk_mlp}). Continuing without MLP.")

    negative_topk = args.topk_mlp if args.topk_negative_mlp is None else args.topk_negative_mlp
    selected_mlp_positive, positive_mlp_policy = _select_mlp(
        rows,
        policy="positive",
        topk=args.topk_mlp,
        min_sign_consistency=mlp_min_sign_consistency,
        require_directional_ci=mlp_require_directional_ci,
        min_abs_mean_delta=args.min_mlp_abs_mean_delta,
        excluded_layers=excluded_mlp_layers,
        min_directional_transition_rate=min_directional_transition_rate,
        min_ablated_format_ok=args.min_ablated_format_ok,
        max_format_collapse_rate=args.max_format_collapse_rate,
        min_abs_mean_target_drop=args.min_abs_mean_target_drop,
        min_abs_mean_source_rise=args.min_abs_mean_source_rise,
    )
    selected_mlp_negative, negative_mlp_policy = _select_mlp(
        rows,
        policy="negative",
        topk=negative_topk,
        min_sign_consistency=mlp_min_sign_consistency,
        require_directional_ci=mlp_require_directional_ci,
        min_abs_mean_delta=args.min_mlp_abs_mean_delta,
        excluded_layers=excluded_mlp_layers,
        min_directional_transition_rate=min_directional_transition_rate,
        min_ablated_format_ok=args.min_ablated_format_ok,
        max_format_collapse_rate=args.max_format_collapse_rate,
        min_abs_mean_target_drop=args.min_abs_mean_target_drop,
        min_abs_mean_source_rise=args.min_abs_mean_source_rise,
    )

    excluded_layers = _parse_excluded(args.exclude_attn_layers)
    selected_attn = _select_attn_layers(
        rows,
        topk=args.topk_attn_layers,
        min_sign_consistency=attn_min_sign_consistency,
        require_directional_ci=attn_require_directional_ci,
        min_abs_mean_delta=args.min_attn_abs_mean_delta,
        excluded_layers=excluded_layers,
        min_directional_transition_rate=min_directional_transition_rate,
        min_ablated_format_ok=args.min_ablated_format_ok,
        max_format_collapse_rate=args.max_format_collapse_rate,
        min_abs_mean_target_drop=args.min_abs_mean_target_drop,
        min_abs_mean_source_rise=args.min_abs_mean_source_rise,
    )
    if args.topk_attn_layers > 0 and not selected_attn and not args.allow_empty_attn:
        raise SystemExit("No attention layers selected. Lower --min-sign-consistency or pass --allow-empty-attn.")

    mlp_component_rows = [
        _component_row(row, rank=rank, role="logic_support_mlp", policy=actual_mlp_policy)
        for rank, row in enumerate(selected_mlp, start=1)
    ]
    mlp_positive_rows = [
        _component_row(row, rank=rank, role="logic_support_mlp_positive", policy=positive_mlp_policy)
        for rank, row in enumerate(selected_mlp_positive, start=1)
    ]
    mlp_negative_rows = [
        _component_row(row, rank=rank, role="logic_resistance_mlp_negative", policy=negative_mlp_policy)
        for rank, row in enumerate(selected_mlp_negative, start=1)
    ]
    attn_layer_rows = [
        _component_row(row, rank=rank, role="attention_layer_pool", policy="top_unique_abs_delta")
        for rank, row in enumerate(selected_attn, start=1)
    ]
    dump_csv(args.out_dir / "components.csv", mlp_component_rows)
    dump_csv(args.out_dir / "mlp_positive_components.csv", mlp_positive_rows)
    dump_csv(args.out_dir / "mlp_negative_components.csv", mlp_negative_rows)
    dump_csv(args.out_dir / "attention_layers.csv", attn_layer_rows)
    (args.out_dir / "attention_layers.txt").write_text(
        ",".join(str(row["layer_idx"]) for row in selected_attn),
        encoding="utf-8",
    )
    dump_json(
        args.out_dir / "component_selection_manifest.json",
        {
            "event": args.event,
            "component_screen_csv": str(args.component_screen_csv),
            "mlp_policy_requested": args.mlp_policy,
            "mlp_policy_actual": actual_mlp_policy,
            "topk_mlp": args.topk_mlp,
            "topk_negative_mlp": negative_topk,
            "topk_attn_layers": args.topk_attn_layers,
            "min_sign_consistency": args.min_sign_consistency,
            "mlp_min_sign_consistency": mlp_min_sign_consistency,
            "attn_min_sign_consistency": attn_min_sign_consistency,
            "require_directional_ci": args.require_directional_ci,
            "mlp_require_directional_ci": mlp_require_directional_ci,
            "attn_require_directional_ci": attn_require_directional_ci,
            "min_mlp_abs_mean_delta": args.min_mlp_abs_mean_delta,
            "min_attn_abs_mean_delta": args.min_attn_abs_mean_delta,
            "min_bidirectional_rate": args.min_bidirectional_rate,
            "min_directional_transition_rate": min_directional_transition_rate,
            "min_ablated_format_ok": args.min_ablated_format_ok,
            "max_format_collapse_rate": args.max_format_collapse_rate,
            "min_abs_mean_target_drop": args.min_abs_mean_target_drop,
            "min_abs_mean_source_rise": args.min_abs_mean_source_rise,
            "exclude_mlp_layers": sorted(excluded_mlp_layers),
            "exclude_attn_layers": sorted(excluded_layers),
            "selected_mlp_components": [row["component_id"] for row in selected_mlp],
            "selected_mlp_positive_components": [row["component_id"] for row in selected_mlp_positive],
            "selected_mlp_negative_components": [row["component_id"] for row in selected_mlp_negative],
            "selected_attention_layers": [int(row["layer_idx"]) for row in selected_attn],
            "competitive_margin": "C(M;x)=S_M(y_plus|x)-S_M(y_minus|x)",
            "selection_semantics": (
                "Positive mean_delta means the component supports the configured target competitive margin "
                "under zero-ablation Delta(c,x)=C(M_full;x)-C(M_zero_c;x). Selection prefers components with "
                "larger one-sided gross support/resistance mass and direction rate so repeated within-source "
                "pairs do not collapse the selector down to a thin signed mean only."
            ),
        },
    )
    dump_csv(
        args.out_dir / "component_selection_summary.csv",
        [
            {"metric": "event", "value": args.event},
            {"metric": "component_screen_rows", "value": len(rows)},
            {"metric": "selected_mlp_components", "value": len(selected_mlp)},
            {"metric": "selected_mlp_positive_components", "value": len(selected_mlp_positive)},
            {"metric": "selected_mlp_negative_components", "value": len(selected_mlp_negative)},
            {"metric": "selected_attention_layers", "value": len(selected_attn)},
            {"metric": "mlp_policy_actual", "value": actual_mlp_policy},
            {"metric": "require_directional_ci", "value": int(args.require_directional_ci)},
            {"metric": "mlp_require_directional_ci", "value": int(mlp_require_directional_ci)},
            {"metric": "attn_require_directional_ci", "value": int(attn_require_directional_ci)},
            {"metric": "exclude_mlp_layers", "value": ",".join(str(value) for value in sorted(excluded_mlp_layers))},
            {"metric": "mlp_min_sign_consistency", "value": mlp_min_sign_consistency},
            {"metric": "attn_min_sign_consistency", "value": attn_min_sign_consistency},
            {"metric": "min_mlp_abs_mean_delta", "value": args.min_mlp_abs_mean_delta},
            {"metric": "min_attn_abs_mean_delta", "value": args.min_attn_abs_mean_delta},
            {"metric": "min_bidirectional_rate", "value": args.min_bidirectional_rate},
            {"metric": "min_directional_transition_rate", "value": min_directional_transition_rate},
            {"metric": "min_ablated_format_ok", "value": args.min_ablated_format_ok},
            {"metric": "max_format_collapse_rate", "value": args.max_format_collapse_rate},
            {"metric": "min_abs_mean_target_drop", "value": args.min_abs_mean_target_drop},
            {"metric": "min_abs_mean_source_rise", "value": args.min_abs_mean_source_rise},
        ],
    )
    print(
        (
            f"[cecm-select-margin] mlp={len(selected_mlp)} "
            f"attn_layers={','.join(str(row['layer_idx']) for row in selected_attn)} out={args.out_dir}"
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
