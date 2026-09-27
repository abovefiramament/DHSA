#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-data_gsm8k/cast_gsm8k_aligned_timing_probe_480_120_v0}"
MODEL="${MODEL:-LOCAL_HOME/.cache/huggingface/hub/models--deepseek-ai--deepseek-math-7b-instruct}"
TRAIN_SOURCE="${TRAIN_SOURCE:-data_gsm8k/raw/train.jsonl}"
TEST_SOURCE="${TEST_SOURCE:-data_gsm8k/raw/test.jsonl}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"

POOL_ROWS="${POOL_ROWS:-600}"
TRAIN_ROWS="${TRAIN_ROWS:-480}"
VAL_ROWS="${VAL_ROWS:-120}"
VAL_MOD="${VAL_MOD:-5}"
DISCOVERY_ROWS="${DISCOVERY_ROWS:-120}"
HEAD_SCAN_ROWS="${HEAD_SCAN_ROWS:-48}"
PROBE_EVAL_ROWS="${PROBE_EVAL_ROWS:-160}"

MIN_MLP_ABS_MEAN_DELTA="${MIN_MLP_ABS_MEAN_DELTA:-0.02}"
MIN_ATTN_ABS_MEAN_DELTA="${MIN_ATTN_ABS_MEAN_DELTA:-0.25}"
SCAN_MIN_SIGN_CONSISTENCY="${SCAN_MIN_SIGN_CONSISTENCY:-0.50}"
SELECT_MIN_SIGN_CONSISTENCY="${SELECT_MIN_SIGN_CONSISTENCY:-0.50}"

MLP_ALPHAS="${MLP_ALPHAS:-0.03,0.05,0.075}"
HEAD_ALPHAS="${HEAD_ALPHAS:-0.25,0.5,0.75}"
TRAIN_ALPHA_SWEEP="${TRAIN_ALPHA_SWEEP:-0.0,0.5,1.0}"
HEAD_SCAN_FACTORS="${HEAD_SCAN_FACTORS:-0.0,1.5}"

# Context-mainline timing. Each item is tag:teacher_forced_apply:open_generation_apply.
# The default intervention timing is intentionally identical to the ConFiQA/context
# CAST mainline:
# - component vectors use generation timing "prefill";
# - head vectors use generation timing "all".
#
# In this generation runner, "prefill" means the prompt prefill forward only, at
# the last prompt position. "all" means every generation forward, at that
# forward's last position. This is the historical context timing.
#
# Non-context timing ablations, such as all -> all_positions, are refused by
# default. Use ALLOW_NON_CONTEXT_TIMING=1 only for explicit diagnostics, e.g.
# TIMING_SPECS=boxed_decision_probe:boxed_decision:decode. The boxed_decision
# training mode means the state immediately after "\boxed{" and before the gold
# answer token span in the teacher-forced continuation.
TIMING_SPECS="${TIMING_SPECS:-context_mainline:prompt_last:prefill}"
ALLOW_NON_CONTEXT_TIMING="${ALLOW_NON_CONTEXT_TIMING:-0}"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

mkdir -p "$ROOT/configs" "$ROOT/logs"

line_count() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo 0
    return
  fi
  python - "$path" <<'PY'
import pathlib, sys
path = pathlib.Path(sys.argv[1])
with path.open("r", encoding="utf-8") as f:
    print(sum(1 for line in f if line.strip()))
PY
}

POOL_DIR="$ROOT/trajectory_pool/base"
POOL_FILE="$POOL_DIR/generation_rows.jsonl"
have_pool="$(line_count "$POOL_FILE")"
if [[ "$have_pool" -lt "$POOL_ROWS" ]]; then
  mkdir -p "$POOL_DIR"
  echo "[$(date -Is)] collect base trajectory pool have=$have_pool target=$POOL_ROWS"
  python -m screscomp.cli.cecm_run_joint_actuator_generation \
    --model "$MODEL" \
    --eval-open-rows "$TRAIN_SOURCE" \
    --component-actuators "" \
    --head-actuators "" \
    --controls "base=;" \
    --generation-prompt-key deepseek_math_cot \
    --prior-source model_prior \
    --scoring-kind gsm8k \
    --split all \
    --start 0 \
    --max-rows "$POOL_ROWS" \
    --val-mod "$VAL_MOD" \
    --generation-apply-mode prefill \
    --max-new-tokens 512 \
    --stop-strings "\\nUser:" \
    --flush-every 1 \
    --summary-every 100 \
    --empty-cache-every 25 \
    --out-dir "$POOL_DIR" \
    --torch-dtype auto \
    --device auto
fi

IFS=',' read -r -a SPECS <<< "$TIMING_SPECS"
ROUND_DIRS=()
for spec in "${SPECS[@]}"; do
  IFS=':' read -r tag mlp_score_mode mlp_generation_mode <<< "$spec"
  if [[ -z "${tag:-}" || -z "${mlp_score_mode:-}" || -z "${mlp_generation_mode:-}" ]]; then
    echo "Bad TIMING_SPECS item: $spec" >&2
    exit 2
  fi
  if [[ "$ALLOW_NON_CONTEXT_TIMING" != "1" ]]; then
    context_pair=0
    if [[ "$mlp_score_mode" == "prompt_last" && "$mlp_generation_mode" == "prefill" ]]; then
      context_pair=1
    fi
    if [[ "$context_pair" != "1" ]]; then
      echo "Refusing non-context intervention timing: $spec" >&2
      echo "Context mainline requires MLP prompt_last -> prefill and head generation all." >&2
      echo "Use ALLOW_NON_CONTEXT_TIMING=1 only for explicit diagnostics." >&2
      exit 2
    fi
  fi
  cfg="$ROOT/configs/${tag}.json"
  round_dir="$ROOT/round0_${tag}"
  ROUND_DIRS+=("$round_dir")
  python - "$cfg" "$MODEL" "$TRAIN_SOURCE" "$POOL_FILE" "$TEST_SOURCE" "$tag" "$mlp_score_mode" "$mlp_generation_mode" \
    "$TRAIN_ROWS" "$VAL_ROWS" "$VAL_MOD" "$DISCOVERY_ROWS" "$HEAD_SCAN_ROWS" "$PROBE_EVAL_ROWS" \
    "$MIN_MLP_ABS_MEAN_DELTA" "$MIN_ATTN_ABS_MEAN_DELTA" "$SCAN_MIN_SIGN_CONSISTENCY" "$SELECT_MIN_SIGN_CONSISTENCY" \
    "$MLP_ALPHAS" "$HEAD_ALPHAS" "$TRAIN_ALPHA_SWEEP" "$HEAD_SCAN_FACTORS" "$POOL_ROWS" <<'PY'
import json
import sys
from pathlib import Path

(
    cfg_path,
    model,
    train_source,
    pool_file,
    test_source,
    tag,
    mlp_score_mode,
    mlp_generation_mode,
    train_rows,
    val_rows,
    val_mod,
    discovery_rows,
    head_scan_rows,
    probe_eval_rows,
    min_mlp_delta,
    min_attn_delta,
    scan_min_sign,
    select_min_sign,
    mlp_alphas,
    head_alphas,
    train_alpha_sweep,
    head_scan_factors,
    pool_rows,
) = sys.argv[1:]

def floats(raw):
    return [float(item.strip()) for item in raw.split(",") if item.strip()]

head_generation_mode = "all_positions" if mlp_generation_mode == "all_positions" else "all"

config = {
    "schema_version": 1,
    "round": {"index": 0},
    "model": {
        "path": model,
        "device": "auto",
        "torch_dtype": "auto",
        "use_chat_template": False,
    },
    "output": {"root": str(Path(cfg_path).parent.parent / "unused_rounds")},
    "task": {
        "adapter": "gsm8k_full_trajectory",
        "name": f"gsm8k_deepseek_{tag}",
        "event": "gsm8k_gold_answer_logic_advantage",
        "source_jsonl": train_source,
        "rollouts_jsonl": pool_file,
        "prompt_key": "deepseek_math_cot",
        "prompt_semantics": "DeepSeek Math CoT prompt: Please reason step by step, and put your final answer within \\boxed{}.",
        "val_mod": int(val_mod),
    },
    "pairs": {
        "max_rows": int(pool_rows),
        "continuation_policy": "single_full_open_generation",
    },
    "discovery": {
        "component_source": "reselect",
        "reselect_components": True,
        "component_types": ["attn", "mlp"],
        "component_type_apply_modes": {"attn": "all", "mlp": mlp_score_mode},
        "allow_ci_cross_zero": False,
        "require_directional_ci": True,
        "rows": int(discovery_rows),
        "mlp_topk": 4,
        "mlp_negative_topk": 4,
        "attn_layer_topk": 4,
        "exclude_attn_layers": "0,1",
        "head_scan_rows": int(head_scan_rows),
        "head_scan_factors": floats(head_scan_factors),
        "head_topk": 4,
        "head_require_directional_ci": True,
        "scan_min_sign_consistency": float(scan_min_sign),
        "min_sign_consistency": float(select_min_sign),
        "min_mlp_abs_mean_delta": float(min_mlp_delta),
        "min_attn_abs_mean_delta": float(min_attn_delta),
    },
    "training": {
        "score_mode": "answer_rest_margin",
        "option_selection_mode": "first",
        "max_aliases_per_side": 1,
        "train_split": "train",
        "val_split": "val",
        "max_train_rows": int(train_rows),
        "max_val_rows": int(val_rows),
        "epochs": 2,
        "lr": 0.05,
        "lambda_norm": 1e-4,
        "alpha_train": 1.0,
        "state_margin_weight": 0.0,
        "gain_weight": 1.0,
        "target_margin": 0.0,
        "target_gain": 0.0,
        "empty_cache_every": 25,
        "mlp": {
            "enabled": True,
            "train_apply_mode": mlp_score_mode,
            "generation_apply_mode": mlp_generation_mode,
        },
        "attention": {
            "enabled": True,
            "train_apply_mode": "all",
            "generation_apply_mode": head_generation_mode,
        },
    },
    "alpha": {
        "scan": True,
        "mlp": floats(mlp_alphas),
        "head": floats(head_alphas),
        "train_sweep": floats(train_alpha_sweep),
    },
    "evaluation": {
        "enabled": True,
        "name": "probe_eval",
        "eval_open_rows": test_source,
        "generation_prompt_key": "deepseek_math_cot",
        "scoring_kind": "gsm8k",
        "split": "all",
        "start": 0,
        "max_rows": int(probe_eval_rows),
        "generation_apply_mode": "prefill",
        "max_new_tokens": 512,
        "stop_strings": "\\nUser:",
        "flush_every": 1,
        "summary_every": 50,
        "empty_cache_every": 25,
    },
}
Path(cfg_path).write_text(json.dumps(config, indent=2), encoding="utf-8")
PY
  echo "[$(date -Is)] run timing=$tag mlp_score=$mlp_score_mode mlp_generation=$mlp_generation_mode out=$round_dir"
  python -m screscomp.cli.cecm_round \
    --config "$cfg" \
    --out-dir "$round_dir" \
    --execute
done

python -m screscomp.cli.cecm_select_round_controls \
  --round-dirs "${ROUND_DIRS[@]}" \
  --eval-name probe_eval \
  --top-per-category 2 \
  --avoid-boundary \
  --out-dir "$ROOT/selection"

echo "[$(date -Is)] done root=$ROOT selection=$ROOT/selection/selected_controls.csv"
