from __future__ import annotations

import argparse
from pathlib import Path

from screscomp.cecm.pairs import (
    DATASET_PRIOR_ANSWER_KEYS,
    MODEL_PRIOR_ANSWER_KEYS,
    SOURCE_CONTEXT_OVER_PRIOR,
    PairBuildConfig,
    build_preference_pairs,
    summarize_pair_build,
)
from screscomp.cecm.specs import load_objective_spec
from screscomp.data import dump_csv, dump_json, dump_jsonl, load_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build admitted CECM preference pairs from CK-PLUG open-generation rows."
    )
    p.add_argument("--input-jsonl", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--event",
        type=str,
        default="",
        help="Objective id. Defaults to source_context_over_prior when --spec-json is not provided.",
    )
    p.add_argument(
        "--spec-json",
        type=Path,
        default=None,
        help="Optional objective spec JSON. Use this for any new axis without changing the algorithm.",
    )
    p.add_argument("--prompt-key", type=str, default="", help="Override the prompt key in the objective spec.")
    p.add_argument("--val-mod", type=int, default=None, help="Override the validation modulus in the objective spec.")
    p.add_argument(
        "--prior-source",
        choices=["auto", "model_prior", "dataset_orig"],
        default="auto",
        help=(
            "For source objectives, choose the negative endpoint. auto prefers discovered "
            "model_prior_answer/model_prior_answers and falls back to dataset orig/prior fields."
        ),
    )
    p.add_argument(
        "--continuation-prefix",
        type=str,
        default=None,
        help="Override the continuation prefix in the objective spec.",
    )
    p.add_argument("--max-rows", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input_jsonl)
    if args.max_rows is not None:
        rows = rows[: args.max_rows]

    objective_id = args.event or (SOURCE_CONTEXT_OVER_PRIOR if args.spec_json is None else "")
    spec = load_objective_spec(objective_id=objective_id, spec_json=args.spec_json)
    prompt_key = args.prompt_key or spec.prompt_key
    y_minus_keys = spec.y_minus_keys
    y_minus_fallback_keys = spec.y_minus_fallback_keys
    if spec.objective_id == SOURCE_CONTEXT_OVER_PRIOR:
        if args.prior_source == "auto":
            y_minus_keys = MODEL_PRIOR_ANSWER_KEYS
            y_minus_fallback_keys = spec.y_minus_keys
        elif args.prior_source == "model_prior":
            y_minus_keys = MODEL_PRIOR_ANSWER_KEYS
            y_minus_fallback_keys = tuple()
        elif args.prior_source == "dataset_orig":
            y_minus_keys = DATASET_PRIOR_ANSWER_KEYS
            y_minus_fallback_keys = tuple()
    config = PairBuildConfig(
        event=spec.objective_id,
        prompt_key=prompt_key,
        y_plus_keys=spec.y_plus_keys,
        y_minus_keys=y_minus_keys,
        y_minus_fallback_keys=y_minus_fallback_keys,
        val_mod=args.val_mod if args.val_mod is not None else spec.val_mod,
        continuation_prefix=args.continuation_prefix if args.continuation_prefix is not None else spec.continuation_prefix,
    )
    pairs, rejected = build_preference_pairs(
        rows,
        source_path=str(args.input_jsonl),
        config=config,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dump_csv(args.out_dir / "pairs.csv", [pair.to_row() for pair in pairs])
    dump_csv(args.out_dir / "rejected_pairs.csv", [item.to_row() for item in rejected])
    dump_jsonl(
        args.out_dir / "pairs.jsonl",
        (
            {
                "pair": pair.to_row(),
                "raw_row": rows[pair.row_index],
            }
            for pair in pairs
        ),
    )
    dump_jsonl(
        args.out_dir / "rejected_pairs.jsonl",
        (
            {
                "rejected_pair": item.to_row(),
                "raw_row": rows[item.row_index],
            }
            for item in rejected
        ),
    )
    dump_csv(args.out_dir / "pair_admission_summary.csv", summarize_pair_build(pairs, rejected))
    dump_json(
        args.out_dir / "pair_build_manifest.json",
        {
            "input_jsonl": str(args.input_jsonl),
            "event": spec.objective_id,
            "spec_json": str(args.spec_json) if args.spec_json else "",
            "objective_spec": spec.to_dict(),
            "prompt_key": config.prompt_key,
            "prior_source": args.prior_source,
            "y_minus_keys": list(config.y_minus_keys),
            "y_minus_fallback_keys": list(config.y_minus_fallback_keys),
            "val_mod": config.val_mod,
            "continuation_prefix_repr": repr(config.continuation_prefix),
            "max_rows": args.max_rows,
            "admitted_pairs": len(pairs),
            "rejected_pairs": len(rejected),
            "semantics": (
                "y_plus is the context/counterfactual-supported answer; "
                "y_minus is the discovered model prior when available and requested, otherwise the configured "
                "fallback prior/original endpoint; prompt comes from the configured nested prompt key."
            ),
        },
    )
    print(
        f"[cecm] admitted={len(pairs)} rejected={len(rejected)} out={args.out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
