from __future__ import annotations

import argparse
import json
from pathlib import Path

from screscomp.cecm.round_executor import execute_round
from screscomp.cecm.rounds import load_round_config, materialize_round_pairs, round_dir, round_plan
from screscomp.data import dump_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run one generic CECM/CAST round from a centralized config. Task-specific code may only "
            "materialize prompts, continuations, and external option scores; actuator algorithms stay shared."
        )
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--round", type=int, default=None, dest="round_index")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument(
        "--plan-only",
        action="store_true",
        help="Write the effective config and planned phases without materializing pair CSVs.",
    )
    p.add_argument(
        "--execute",
        action="store_true",
        help="Run the configured shared-core scan/select/train/eval phases after materializing pairs.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Rerun completed scan/select/train phases instead of reusing existing phase outputs.",
    )
    p.add_argument(
        "--skip-eval",
        action="store_true",
        help="Execute scan/select/train but skip open-generation evaluation.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.plan_only and args.execute:
        raise SystemExit("--plan-only and --execute cannot be used together")
    config = load_round_config(args.config)
    if args.round_index is not None:
        config.setdefault("round", {})["index"] = int(args.round_index)
    out_dir = round_dir(config, round_index=args.round_index, out_dir=args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    effective_config_path = out_dir / "effective_round_config.json"
    if effective_config_path.exists() and not args.force:
        existing = json.loads(effective_config_path.read_text(encoding="utf-8"))
        if existing != config:
            raise SystemExit(
                f"Refusing to reuse dirty round directory with different config: {out_dir} "
                f"(use a new out-dirLOCAL_HOME or pass --force after cleaning target outputs)"
            )
    dump_json(effective_config_path, config)

    pairs_result = {"pairs_csv": str(out_dir / "pairs" / "pairs.csv"), "admitted_pairs": "", "rejected_pairs": ""}
    if not args.plan_only:
        pairs_result = materialize_round_pairs(config, out_dir=out_dir)
    plan = round_plan(config, out_dir=out_dir, pairs_csv=str(pairs_result["pairs_csv"]))
    plan["config_path"] = str(args.config)
    plan["pair_materialization"] = pairs_result
    dump_json(out_dir / "round_plan.json", plan)
    execution_result = None
    if args.execute:
        execution_result = execute_round(
            config=config,
            out_dir=out_dir,
            pairs_csv=str(pairs_result["pairs_csv"]),
            force=bool(args.force),
            skip_eval=bool(args.skip_eval),
        )
    print(
        (
            f"[cecm-round] out={out_dir} adapter={config['task']['adapter']} "
            f"pairs={pairs_result.get('admitted_pairs', '')} plan={out_dir / 'round_plan.json'}"
        ),
        flush=True,
    )
    if execution_result is not None:
        print(f"[cecm-round] execution={out_dir / 'round_execution.json'}", flush=True)


if __name__ == "__main__":
    main()
