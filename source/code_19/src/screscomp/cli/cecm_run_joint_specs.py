from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run cecm_run_joint_actuator_generation from exported joint_specs.json.")
    p.add_argument("--specs", type=Path, required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--eval-open-rows", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--generation-prompt-key", type=str, default="deepseek_math_cot")
    p.add_argument("--prior-source", type=str, default="model_prior")
    p.add_argument("--scoring-kind", choices=["gsm8k", "source"], default="gsm8k")
    p.add_argument("--split", type=str, default="all")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--val-mod", type=int, default=5)
    p.add_argument("--generation-apply-mode", type=str, default="prefill")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--stop-strings", type=str, default="\\nUser:")
    p.add_argument("--flush-every", type=int, default=1)
    p.add_argument("--summary-every", type=int, default=100)
    p.add_argument("--empty-cache-every", type=int, default=25)
    p.add_argument("--torch-dtype", type=str, default="auto")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--use-chat-template", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    specs = json.loads(args.specs.read_text(encoding="utf-8"))
    command = [
        sys.executable,
        "-m",
        "screscomp.cli.cecm_run_joint_actuator_generation",
        "--model",
        args.model,
        "--eval-open-rows",
        str(args.eval_open_rows),
        "--component-actuators",
        str(specs.get("component_actuators_text", "")),
        "--head-actuators",
        str(specs.get("head_actuators_text", "")),
        "--controls",
        str(specs.get("controls_text", "base=;")),
        "--generation-prompt-key",
        args.generation_prompt_key,
        "--prior-source",
        args.prior_source,
        "--scoring-kind",
        args.scoring_kind,
        "--split",
        args.split,
        "--start",
        str(args.start),
        "--max-rows",
        str(args.max_rows),
        "--val-mod",
        str(args.val_mod),
        "--generation-apply-mode",
        args.generation_apply_mode,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--stop-strings",
        args.stop_strings,
        "--flush-every",
        str(args.flush_every),
        "--summary-every",
        str(args.summary_every),
        "--empty-cache-every",
        str(args.empty_cache_every),
        "--torch-dtype",
        args.torch_dtype,
        "--device",
        args.device,
        "--out-dir",
        str(args.out_dir),
    ]
    if args.use_chat_template:
        command.append("--use-chat-template")
    if args.overwrite:
        command.append("--overwrite")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
