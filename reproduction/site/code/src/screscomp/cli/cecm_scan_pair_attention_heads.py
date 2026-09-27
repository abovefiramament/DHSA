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
        description="Scan and select attention heads from a fixed pair subset using teacher-forced pair margins."
    )
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--pairs-csv", type=Path, required=True)
    p.add_argument("--event", type=str, required=True)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--max-rows", type=int, default=600)
    p.add_argument("--attn-layers", type=str, required=True)
    p.add_argument("--scan-factors", type=str, default="0.0")
    p.add_argument("--topk", type=int, default=4)
    p.add_argument(
        "--score-mode",
        type=str,
        default="avglogp",
        choices=["avglogp", "top_logit_gap", "answer_rest_margin"],
    )
    p.add_argument(
        "--score-apply-mode",
        type=str,
        default="all",
        choices=["decision_tokens", "boxed_decision", "prompt_last", "prompt", "all"],
    )
    p.add_argument(
        "--option-selection-mode",
        type=str,
        default="first",
        choices=OPTION_SELECTION_MODES,
    )
    p.add_argument("--max-aliases-per-side", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use-chat-template", action="store_true")
    p.add_argument(
        "--torch-dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
    )
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args(argv)


def _parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def _parse_ints(raw: str) -> list[int]:
    return [int(item) for item in _parse_csv(raw)]


def _parse_floats(raw: str) -> list[float]:
    return [float(item) for item in _parse_csv(raw)]


def _zero_rows_by_head(scan_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    best: dict[str, dict[str, object]] = {}
    for row in scan_rows:
        try:
            factor = float(row.get("factor", 1.0))
        except Exception:
            continue
        if abs(factor) > 1e-12:
            continue
        head_id = str(row.get("head_id", ""))
        if not head_id:
            continue
        best[head_id] = row
    return list(best.values())


def _existence_delta(row: dict[str, object]) -> float:
    return -float(row.get("mean_margin_gain", 0.0))


def _selected_row(row: dict[str, object], *, rank: int, group_name: str) -> dict[str, object]:
    direction = "positive" if group_name == "head_positive" else "negative"
    return {
        "rank": rank,
        "group_name": group_name,
        "selection_direction": direction,
        "head_id": str(row["head_id"]),
        "component_id": str(row["head_id"]),
        "layer_idx": int(row["layer_idx"]),
        "head_idx": int(row["head_idx"]),
        "role": "zero",
        "factor": float(row["factor"]),
        "selection_metric": "mean_full_minus_zero_delta",
        "selection_score": _existence_delta(row),
        "mean_margin_gain": float(row["mean_margin_gain"]),
        "mean_full_minus_zero_delta": _existence_delta(row),
        "gain_positive_rate": float(row["gain_positive_rate"]),
        "pref_rate": float(row["pref_rate"]),
        "gain_ci95_low": float(row["gain_ci95_low"]),
        "gain_ci95_high": float(row["gain_ci95_high"]),
        "n": int(float(row["n"])),
    }


def _write_selected(
    *,
    out_dir: Path,
    stem: str,
    rows: list[dict[str, object]],
    alias_stem: str | None = None,
) -> None:
    dump_csv(out_dir / f"{stem}.csv", rows)
    head_ids = ",".join(str(row["head_id"]) for row in rows)
    (out_dir / f"{stem}.txt").write_text(head_ids, encoding="utf-8")
    if alias_stem is not None:
        dump_csv(out_dir / f"{alias_stem}.csv", rows)
        (out_dir / f"{alias_stem}.txt").write_text(head_ids, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    attn_layers = _parse_ints(args.attn_layers)
    scan_factors = _parse_floats(args.scan_factors)
    if any(abs(value) > 1e-12 for value in scan_factors):
        raise SystemExit("cecm_scan_pair_attention_heads is zero-only: --scan-factors must contain only 0.0")
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
        start=args.start,
        max_rows=args.max_rows,
    )
    baselines = _cache_baselines(runner, pairs)
    scan_rows = _scan_heads(
        runner,
        pairs,
        baselines,
        attn_layers=attn_layers,
        factors=scan_factors,
    )
    dump_csv(args.out_dir / "head_scan.csv", scan_rows)

    zero_rows = _zero_rows_by_head(scan_rows)
    positive_candidates = [row for row in zero_rows if _existence_delta(row) > 0]
    negative_candidates = [row for row in zero_rows if _existence_delta(row) < 0]
    positive_rows_src = sorted(positive_candidates, key=_existence_delta, reverse=True)[: args.topk]
    negative_rows_src = sorted(negative_candidates, key=_existence_delta)[: args.topk]
    if len(positive_rows_src) != args.topk or len(negative_rows_src) != args.topk:
        raise SystemExit(
            f"Not enough heads for topk={args.topk}: "
            f"positive={len(positive_rows_src)} negative={len(negative_rows_src)}"
        )

    positive_rows = [
        _selected_row(row, rank=index, group_name="head_positive")
        for index, row in enumerate(positive_rows_src, start=1)
    ]
    negative_rows = [
        _selected_row(row, rank=index, group_name="head_negative")
        for index, row in enumerate(negative_rows_src, start=1)
    ]

    _write_selected(
        out_dir=args.out_dir,
        stem="selected_positive_heads",
        alias_stem="head_positive_heads",
        rows=positive_rows,
    )
    _write_selected(
        out_dir=args.out_dir,
        stem="selected_negative_heads",
        alias_stem="head_negative_heads",
        rows=negative_rows,
    )

    dump_json(
        args.out_dir / "head_selection_manifest.json",
        {
            "model": args.model,
            "pairs_csv": str(args.pairs_csv),
            "event": args.event,
            "split": args.split,
            "start": args.start,
            "max_rows": args.max_rows,
            "attn_layers": attn_layers,
            "scan_factors": scan_factors,
            "topk": args.topk,
            "score_mode": args.score_mode,
            "score_apply_mode": args.score_apply_mode,
            "option_selection_mode": args.option_selection_mode,
            "option_selection": option_selection_description(args.option_selection_mode),
            "selected_positive_heads": [str(row["head_id"]) for row in positive_rows],
            "selected_negative_heads": [str(row["head_id"]) for row in negative_rows],
            "semantics": {
                "head_positive": "zeroing the head decreases the target margin the most, so the head most strongly supports the target behavior by existence",
                "head_negative": "zeroing the head increases the target margin the most, so the head most strongly resists the target behavior by existence",
            },
        },
    )
    print(
        f"[cecm-pair-head-scan] selected positive={len(positive_rows)} negative={len(negative_rows)} out={args.out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
