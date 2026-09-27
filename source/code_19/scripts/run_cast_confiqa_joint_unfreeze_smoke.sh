#!/usr/bin/env bash
set -euo pipefail

# Global joint-refinement stage for an existing role-wise CAST ConFiQA run.
# The base model stays frozen. Existing suppress-head and boost-head actuator
# groups initialize trainable vectors; the prior-MLP group is included by
# default and can be disabled with USE_MLP=0 for attention-only refinement.
# Bounded group gates and all included actuator vectors are refined together.

cd "$(dirname "${BASH_SOURCE[0]}")/.."

GPU="${GPU:-3}"
MODEL="${MODEL:-LOCAL_HOME/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct}"
ROOT="${ROOT:-data_ckplug/cast_confiqa_train_size_stability_qa_v0/n300}"
OUT="${OUT:-data_ckplug/cast_confiqa_joint_unfreeze_smoke_qa_n300_v0}"

TRAIN_ROWS="${TRAIN_ROWS:-240}"
VAL_ROWS="${VAL_ROWS:-60}"
EPOCHS="${EPOCHS:-2}"
GATE_LR="${GATE_LR:-0.03}"
VECTOR_LR="${VECTOR_LR:-0.005}"
STATE_MARGIN_WEIGHT="${STATE_MARGIN_WEIGHT:-0.0}"
GAIN_WEIGHT="${GAIN_WEIGHT:-1.0}"
TARGET_MARGIN="${TARGET_MARGIN:-0.0}"
TARGET_GAIN="${TARGET_GAIN:-0.0}"
LAMBDA_GATE="${LAMBDA_GATE:-1e-3}"
LAMBDA_VECTOR="${LAMBDA_VECTOR:-1e-4}"
LAMBDA_DRIFT="${LAMBDA_DRIFT:-1e-4}"
HEAD_GATE_MAX="${HEAD_GATE_MAX:-1.0}"
COMPONENT_GATE_MAX="${COMPONENT_GATE_MAX:-0.2}"
INIT_GATES="${INIT_GATES:-prior_mlp=0.055475719;suppress=0.48858309;boost=0.2481145}"
EVAL_ROWS="${EVAL_ROWS:-500}"
EVAL_SPLIT="${EVAL_SPLIT:-all}"
EMPTY_CACHE_EVERY="${EMPTY_CACHE_EVERY:-25}"
USE_MLP="${USE_MLP:-1}"

MLP_OUT="${MLP_OUT:-$ROOT/train_decision_tokens_prior_mlp}"
HEAD_SUPPRESS_OUT="${HEAD_SUPPRESS_OUT:-$ROOT/train_decision_tokens_head_suppress}"
HEAD_BOOST_OUT="${HEAD_BOOST_OUT:-$ROOT/train_decision_tokens_head_boost}"

mkdir -p "$OUT"
MASTER_LOG="$OUT/joint_unfreeze_smoke.log"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  deactivate || true
fi
source LOCAL_HOME/anaconda3/etc/profile.d/conda.sh
conda activate screscomp
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

test -f "$ROOT/pairs/pairs.csv"
test -f "$ROOT/eval_open_rows.jsonl"
if [[ "$USE_MLP" == "1" ]]; then
  test -f "$MLP_OUT/fixed_actuator.pt"
fi
test -f "$HEAD_SUPPRESS_OUT/head_actuator.pt"
test -f "$HEAD_BOOST_OUT/head_actuator.pt"

TRAIN_OUT="$OUT/train_joint_unfreeze"
GEN_OUT="$OUT/eval_joint_unfreeze_base_rag"
mkdir -p "$TRAIN_OUT" "$GEN_OUT"
REUSE_TRAIN_IF_PRESENT="${REUSE_TRAIN_IF_PRESENT:-1}"

COMPONENT_ACTUATORS=""
COMPONENT_APPLY_MODES=""
if [[ "$USE_MLP" == "1" ]]; then
  COMPONENT_ACTUATORS="prior_mlp=$MLP_OUT/fixed_actuator.pt"
  COMPONENT_APPLY_MODES="prior_mlp=prompt_last"
fi

{
  echo "[$(date -Is)] root=$ROOT out=$OUT model=$MODEL"
  echo "[$(date -Is)] use_mlp=$USE_MLP component_actuators=${COMPONENT_ACTUATORS:-none}"
  echo "[$(date -Is)] train_rows=$TRAIN_ROWS val_rows=$VAL_ROWS epochs=$EPOCHS gate_lr=$GATE_LR vector_lr=$VECTOR_LR"
  echo "[$(date -Is)] state_margin_weight=$STATE_MARGIN_WEIGHT gain_weight=$GAIN_WEIGHT target_margin=$TARGET_MARGIN target_gain=$TARGET_GAIN"
  echo "[$(date -Is)] init_gates=$INIT_GATES head_gate_max=$HEAD_GATE_MAX component_gate_max=$COMPONENT_GATE_MAX"
} | tee -a "$MASTER_LOG"

if [[ "$REUSE_TRAIN_IF_PRESENT" == "1" && -f "$TRAIN_OUT/best_config.env" ]]; then
  echo "[$(date -Is)] reuse joint training payload=$TRAIN_OUT/best_config.env" | tee -a "$MASTER_LOG"
else
  CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.cecm_train_joint_actuator_unfreeze \
    --model "$MODEL" \
    --pairs-csv "$ROOT/pairs/pairs.csv" \
    --event source_context_over_prior \
    --train-split train \
    --val-split val \
    --max-train-rows "$TRAIN_ROWS" \
    --max-val-rows "$VAL_ROWS" \
    --epochs "$EPOCHS" \
    --gate-lr "$GATE_LR" \
    --vector-lr "$VECTOR_LR" \
    --state-margin-weight "$STATE_MARGIN_WEIGHT" \
    --gain-weight "$GAIN_WEIGHT" \
    --target-margin "$TARGET_MARGIN" \
    --target-gain "$TARGET_GAIN" \
    --lambda-gate "$LAMBDA_GATE" \
    --lambda-vector "$LAMBDA_VECTOR" \
    --lambda-drift "$LAMBDA_DRIFT" \
    --score-mode answer_rest_margin \
    --component-actuators "$COMPONENT_ACTUATORS" \
    --head-actuators "suppress=$HEAD_SUPPRESS_OUT/head_actuator.pt;boost=$HEAD_BOOST_OUT/head_actuator.pt" \
    --component-apply-modes "$COMPONENT_APPLY_MODES" \
    --head-apply-modes "suppress=all;boost=all" \
    --component-gate-max "$COMPONENT_GATE_MAX" \
    --head-gate-max "$HEAD_GATE_MAX" \
    --init-gates "$INIT_GATES" \
    --empty-cache-every "$EMPTY_CACHE_EVERY" \
    --torch-dtype bfloat16 \
    --device cuda \
    --out-dir "$TRAIN_OUT" \
    > "$TRAIN_OUT/train.log" 2>&1
fi

# shellcheck disable=SC1090
source "$TRAIN_OUT/best_config.env"
echo "[$(date -Is)] learned $CAST_CONTROLS" | tee -a "$MASTER_LOG"

CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.cecm_run_joint_actuator_generation \
  --model "$MODEL" \
  --eval-open-rows "$ROOT/eval_open_rows.jsonl" \
  --component-actuators "$CAST_COMPONENT_ACTUATORS" \
  --head-actuators "$CAST_HEAD_ACTUATORS" \
  --controls "$CAST_CONTROLS" \
  --generation-prompt-key base_rag \
  --prior-source dataset_orig \
  --split "$EVAL_SPLIT" \
  --start 0 \
  --max-rows "$EVAL_ROWS" \
  --generation-apply-mode prefill \
  --max-new-tokens 64 \
  --stop-strings "Q:" \
  --empty-cache-every "$EMPTY_CACHE_EVERY" \
  --torch-dtype bfloat16 \
  --device cuda \
  --out-dir "$GEN_OUT" \
  > "$GEN_OUT/generation.log" 2>&1

find "$OUT" -maxdepth 4 \( -name gate_summary.csv -o -name train_history.csv -o -name vector_summary.csv -o -name gates.json -o -name best_config.env -o -name generation_summary.csv -o -name run_summary.csv -o -name payload_paths.json \) -print | sort
cat "$GEN_OUT/generation_summary.csv"
