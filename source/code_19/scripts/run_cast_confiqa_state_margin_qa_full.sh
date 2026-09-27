#!/usr/bin/env bash
set -euo pipefail

# Full QA route with completed-state training loss:
#   1) rebuild QA n300 CAST actuators with state-margin completion as the main loss
#   2) joint-unfreeze vectors + gates with the same loss
#   3) evaluate on the held-out QA rows

cd "$(dirname "${BASH_SOURCE[0]}")/.."

GPU="${GPU:-3}"
BASE_ROOT="${BASE_ROOT:-data_ckplug/cast_confiqa_state_margin_qa_n300_v0}"
OUT="${OUT:-data_ckplug/cast_confiqa_state_margin_joint_unfreeze_qa_n300_full_v0}"

TRAIN_SOURCE_ROWS="${TRAIN_SOURCE_ROWS:-300}"
EVAL_SOURCE_START="${EVAL_SOURCE_START:-1000}"
EVAL_SOURCE_ROWS="${EVAL_SOURCE_ROWS:-}"
TRAIN_ROWS="${TRAIN_ROWS:-240}"
VAL_ROWS="${VAL_ROWS:-60}"
EPOCHS="${EPOCHS:-2}"
DISCOVERY_ROWS="${DISCOVERY_ROWS:-60}"
HEAD_SCAN_ROWS="${HEAD_SCAN_ROWS:-24}"

STATE_MARGIN_WEIGHT="${STATE_MARGIN_WEIGHT:-1.0}"
GAIN_WEIGHT="${GAIN_WEIGHT:-0.25}"
TARGET_MARGIN="${TARGET_MARGIN:-0.0}"
TARGET_GAIN="${TARGET_GAIN:-0.0}"

JOINT_EVAL_ROWS="${JOINT_EVAL_ROWS:-5000}"
EMPTY_CACHE_EVERY="${EMPTY_CACHE_EVERY:-25}"

mkdir -p "$OUT"

echo "[$(date -Is)] base_root=$BASE_ROOT out=$OUT gpu=$GPU"
echo "[$(date -Is)] state_margin_weight=$STATE_MARGIN_WEIGHT gain_weight=$GAIN_WEIGHT target_margin=$TARGET_MARGIN target_gain=$TARGET_GAIN"

GPU="$GPU" \
TASK=qa \
ROOT="$BASE_ROOT" \
TRAIN_SOURCE_ROWS="$TRAIN_SOURCE_ROWS" \
EVAL_SOURCE_START="$EVAL_SOURCE_START" \
EVAL_SOURCE_ROWS="$EVAL_SOURCE_ROWS" \
TRAIN_ROWS="$TRAIN_ROWS" \
VAL_ROWS="$VAL_ROWS" \
MLP_TRAIN_ROWS="$TRAIN_ROWS" \
MLP_VAL_ROWS="$VAL_ROWS" \
HEAD_TRAIN_ROWS="$TRAIN_ROWS" \
HEAD_VAL_ROWS="$VAL_ROWS" \
EPOCHS="$EPOCHS" \
MLP_EPOCHS="$EPOCHS" \
HEAD_EPOCHS="$EPOCHS" \
DISCOVERY_ROWS="$DISCOVERY_ROWS" \
HEAD_SCAN_ROWS="$HEAD_SCAN_ROWS" \
STATE_MARGIN_WEIGHT="$STATE_MARGIN_WEIGHT" \
GAIN_WEIGHT="$GAIN_WEIGHT" \
TARGET_MARGIN="$TARGET_MARGIN" \
TARGET_GAIN="$TARGET_GAIN" \
RUN_BASE_PROMPTS=0 \
RUN_CONTEXT_DPO=0 \
AUTO_TUNE=0 \
EVAL_ROWS=1 \
EVAL_SPLIT=all \
EMPTY_CACHE_EVERY="$EMPTY_CACHE_EVERY" \
bash scripts/run_cast_confiqa_min_loop.sh

GPU="$GPU" \
ROOT="$BASE_ROOT" \
OUT="$OUT" \
TRAIN_ROWS="$TRAIN_ROWS" \
VAL_ROWS="$VAL_ROWS" \
EPOCHS="$EPOCHS" \
STATE_MARGIN_WEIGHT="$STATE_MARGIN_WEIGHT" \
GAIN_WEIGHT="$GAIN_WEIGHT" \
TARGET_MARGIN="$TARGET_MARGIN" \
TARGET_GAIN="$TARGET_GAIN" \
EVAL_ROWS="$JOINT_EVAL_ROWS" \
EVAL_SPLIT=all \
EMPTY_CACHE_EVERY="$EMPTY_CACHE_EVERY" \
bash scripts/run_cast_confiqa_joint_unfreeze_smoke.sh

echo "[$(date -Is)] done"
cat "$OUT/eval_joint_unfreeze_base_rag/generation_summary.csv"
