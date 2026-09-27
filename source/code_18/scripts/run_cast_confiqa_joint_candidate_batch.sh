#!/usr/bin/env bash
set -euo pipefail

# Overnight shortlist for strong CAST / ConFiQA joint-refinement candidates:
#   - head4 attention-only joint from the full-QA fixed anchor
#   - head6 attention-only joint from a fresh fixed top-k role-wise init
#   - full MLP4/head4 joint from the full-QA fixed anchor
#   - full MLP4/head6 joint from the same head6 role-wise init
#   - full MLP2/head4 joint as the lighter-MLP hedge
#
# The top-k init roots use 1-row fixed generation only to materialize the
# role-wise actuator payloads and held-out rows. Full generation happens in the
# joint candidates below.

cd "$(dirname "${BASH_SOURCE[0]}")/.."

GPU="${GPU:-3}"
TASK="${TASK:-qa}"
MODEL="${MODEL:-LOCAL_HOME/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct}"
BASE_OUT="${BASE_OUT:-data_ckplug/cast_confiqa_joint_candidate_batch_${TASK}_v0}"
ANCHOR_ROOT="${ANCHOR_ROOT:-data_ckplug/cast_confiqa_all3_smalltrain_heldout5000_v0/qa}"
INIT_BASE="${INIT_BASE:-$BASE_OUT/rolewise_init}"

TRAIN_SOURCE_ROWS="${TRAIN_SOURCE_ROWS:-300}"
EVAL_SOURCE_START="${EVAL_SOURCE_START:-1000}"
EVAL_SOURCE_ROWS="${EVAL_SOURCE_ROWS:-}"
TRAIN_ROWS="${TRAIN_ROWS:-240}"
VAL_ROWS="${VAL_ROWS:-60}"
EPOCHS="${EPOCHS:-2}"
FULL_EVAL_ROWS="${FULL_EVAL_ROWS:-5000}"
EVAL_SPLIT="${EVAL_SPLIT:-all}"
EMPTY_CACHE_EVERY="${EMPTY_CACHE_EVERY:-25}"
EXCLUDE_HEAD_LAYERS="${EXCLUDE_HEAD_LAYERS:-0}"

STATE_MARGIN_WEIGHT="${STATE_MARGIN_WEIGHT:-0}"
GAIN_WEIGHT="${GAIN_WEIGHT:-1}"
HEAD_INIT_GATES="${HEAD_INIT_GATES:-suppress=0.5;boost=0.5}"
FULL_INIT_GATES="${FULL_INIT_GATES:-prior_mlp=0.055475719;suppress=0.48858309;boost=0.2481145}"

mkdir -p "$BASE_OUT" "$INIT_BASE"
MASTER_LOG="$BASE_OUT/joint_candidate_batch.log"

log() {
  echo "[$(date -Is)] $*" | tee -a "$MASTER_LOG"
}

has_rolewise_payloads() {
  local root="$1"
  test -f "$root/train_decision_tokens_prior_mlp/fixed_actuator.pt" \
    && test -f "$root/train_decision_tokens_head_suppress/head_actuator.pt" \
    && test -f "$root/train_decision_tokens_head_boost/head_actuator.pt" \
    && test -f "$root/pairs/pairs.csv" \
    && test -f "$root/eval_open_rows.jsonl"
}

eval_complete() {
  local eval_dir="$1"
  local run_summary="$eval_dir/run_summary.csv"
  [[ -f "$run_summary" ]] || return 1
  RUN_SUMMARY="$run_summary" python - <<'PY'
import csv
import os
from pathlib import Path

metrics = {
    row.get("metric", ""): int(float(row.get("value", "0") or 0))
    for row in csv.DictReader(Path(os.environ["RUN_SUMMARY"]).open("r", encoding="utf-8", newline=""))
}
expected = metrics.get("eval_rows", 0) * metrics.get("control_specs", 0)
done = metrics.get("generation_rows", 0)
raise SystemExit(0 if expected > 0 and done >= expected else 1)
PY
}

ensure_topk_inits() {
  local head6_root="$INIT_BASE/mlp4_head6_init"
  local mlp2_root="$INIT_BASE/mlp2_head4_init"
  if has_rolewise_payloads "$head6_root" && has_rolewise_payloads "$mlp2_root"; then
    log "reuse role-wise init roots=$head6_root,$mlp2_root"
    return
  fi
  log "build role-wise init roots for mlp4_head6 and mlp2_head4"
  GPU="$GPU" \
  TASK="$TASK" \
  MODEL="$MODEL" \
  BASE_OUT="$INIT_BASE" \
  SPECS="mlp4_head6_init:4:6 mlp2_head4_init:2:4" \
  TRAIN_SOURCE_ROWS="$TRAIN_SOURCE_ROWS" \
  EVAL_SOURCE_START="$EVAL_SOURCE_START" \
  EVAL_SOURCE_ROWS="$EVAL_SOURCE_ROWS" \
  TRAIN_ROWS="$TRAIN_ROWS" \
  VAL_ROWS="$VAL_ROWS" \
  EPOCHS="$EPOCHS" \
  STATE_MARGIN_WEIGHT="$STATE_MARGIN_WEIGHT" \
  GAIN_WEIGHT="$GAIN_WEIGHT" \
  EXCLUDE_HEAD_LAYERS="$EXCLUDE_HEAD_LAYERS" \
  EVAL_ROWS=1 \
  EVAL_SPLIT="$EVAL_SPLIT" \
  RUN_JOINT_UNFREEZE=0 \
  EMPTY_CACHE_EVERY="$EMPTY_CACHE_EVERY" \
  bash scripts/run_cast_confiqa_topk_window_fast.sh \
    > "$INIT_BASE/rolewise_init.log" 2>&1
}

run_head_candidate() {
  local name="$1"
  local root="$2"
  local out="$BASE_OUT/$name"
  local eval_dir="$out/eval_joint_unfreeze_base_rag"
  if eval_complete "$eval_dir"; then
    log "skip completed head candidate=$name eval_dir=$eval_dir"
    return
  fi
  log "run head candidate=$name root=$root out=$out"
  mkdir -p "$out"
  GPU="$GPU" \
  MODEL="$MODEL" \
  ROOT="$root" \
  OUT="$out" \
  TRAIN_ROWS="$TRAIN_ROWS" \
  VAL_ROWS="$VAL_ROWS" \
  EPOCHS="$EPOCHS" \
  STATE_MARGIN_WEIGHT="$STATE_MARGIN_WEIGHT" \
  GAIN_WEIGHT="$GAIN_WEIGHT" \
  INIT_GATES="$HEAD_INIT_GATES" \
  EVAL_ROWS="$FULL_EVAL_ROWS" \
  EVAL_SPLIT="$EVAL_SPLIT" \
  EMPTY_CACHE_EVERY="$EMPTY_CACHE_EVERY" \
  bash scripts/run_cast_confiqa_head_joint_refine.sh \
    > "$out/runner.log" 2>&1
}

run_full_candidate() {
  local name="$1"
  local root="$2"
  local out="$BASE_OUT/$name"
  local eval_dir="$out/eval_joint_unfreeze_base_rag"
  if eval_complete "$eval_dir"; then
    log "skip completed full candidate=$name eval_dir=$eval_dir"
    return
  fi
  log "run full candidate=$name root=$root out=$out"
  mkdir -p "$out"
  GPU="$GPU" \
  MODEL="$MODEL" \
  ROOT="$root" \
  OUT="$out" \
  USE_MLP=1 \
  TRAIN_ROWS="$TRAIN_ROWS" \
  VAL_ROWS="$VAL_ROWS" \
  EPOCHS="$EPOCHS" \
  STATE_MARGIN_WEIGHT="$STATE_MARGIN_WEIGHT" \
  GAIN_WEIGHT="$GAIN_WEIGHT" \
  INIT_GATES="$FULL_INIT_GATES" \
  EVAL_ROWS="$FULL_EVAL_ROWS" \
  EVAL_SPLIT="$EVAL_SPLIT" \
  EMPTY_CACHE_EVERY="$EMPTY_CACHE_EVERY" \
  bash scripts/run_cast_confiqa_joint_unfreeze_smoke.sh \
    > "$out/runner.log" 2>&1
}

log "start gpu=$GPU base_out=$BASE_OUT full_eval_rows=$FULL_EVAL_ROWS"
log "gain objective state_margin_weight=$STATE_MARGIN_WEIGHT gain_weight=$GAIN_WEIGHT exclude_head_layers=$EXCLUDE_HEAD_LAYERS"

run_head_candidate "head4_attn_joint" "$ANCHOR_ROOT"
ensure_topk_inits
run_head_candidate "head6_attn_joint" "$INIT_BASE/mlp4_head6_init"
run_full_candidate "mlp4_head4_full_joint" "$ANCHOR_ROOT"
run_full_candidate "mlp4_head6_full_joint" "$INIT_BASE/mlp4_head6_init"
run_full_candidate "mlp2_head4_full_joint" "$INIT_BASE/mlp2_head4_init"

log "done"
find "$BASE_OUT" -maxdepth 5 \( -name generation_summary.csv -o -name gates.json -o -name best_config.env \) -print | sort
