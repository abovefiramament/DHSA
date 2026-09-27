#!/usr/bin/env bash
set -euo pipefail

# Supplementary pre-O head random-control pipeline.
#
# Purpose:
#   - same_layer_random_retrained: keep each selected layer fixed but randomize
#     the head index, testing whether head identity matters.
#   - same_head_outside_band_retrained: keep each selected head index fixed but
#     move to an outside-band layer, testing whether layer/interface range matters.
#   - outside_band_random_retrained: sample heads outside the selected +/- max
#     shift layer neighborhood, testing whether useful head layers are
#     effectively unrestricted.
#
# The selected condition is not retrained. Shift controls are not retrained here.
# This script only trains/evaluates the two random retrained controls.

export LC_ALL=C.UTF-8
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

REPO_ROOT="${REPO_ROOT:-LOCAL_HOME/RPEC/projects/screscomp}"
cd "$REPO_ROOT"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

PY="${PY:-LOCAL_HOME/anaconda3/envs/screscomp/bin/python}"
GPU="${GPU:-1}"
SERIAL_ROOT="${SERIAL_ROOT:-/dev/shm/screscomp_runs/pre_o_head_random_controls_after_site_$(date -u +%Y%m%d_%H%M%S)}"
WAIT_FOR_SERIAL_PID="${WAIT_FOR_SERIAL_PID:-}"
WAIT_FOR_STATUS_ROOT="${WAIT_FOR_STATUS_ROOT:-}"
DISCOVERY_BASE="${DISCOVERY_BASE:-data_ckplug/cast_confiqa_all3_disc120_heldout500_v0}"
SHIFTS="${SHIFTS:-1 2 3}"
RANDOM_CONTROLS="${RANDOM_CONTROLS:-same_layer_random_retrained same_head_outside_band_retrained outside_band_random_retrained}"
ALPHA_SWEEP="${ALPHA_SWEEP:-0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}"
MAX_GPU_USED_MIB="${MAX_GPU_USED_MIB:-256}"

mkdir -p "$SERIAL_ROOT/logs"
STATUS="$SERIAL_ROOT/status.tsv"
MASTER_LOG="$SERIAL_ROOT/random_controls_pipeline.log"
printf 'time\tstage\tstatus\tnote\n' > "$STATUS"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$MASTER_LOG"
}

stage() {
  local name="$1"
  local status="$2"
  local note="${3:-}"
  printf '%s\t%s\t%s\t%s\n' "$(date -Is)" "$name" "$status" "$note" >> "$STATUS"
  log "stage=$name status=$status ${note}"
}

gpu_used_mib() {
  nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '
}

wait_gpu_free() {
  local name="$1"
  while true; do
    local used
    used="$(gpu_used_mib)"
    if [[ "$used" =~ ^[0-9]+$ ]] && (( used <= MAX_GPU_USED_MIB )); then
      stage "gpu_${name}" free "gpu=$GPU used_mib=$used"
      return
    fi
    stage "gpu_${name}" waiting "gpu=$GPU used_mib=$used"
    sleep 60
  done
}

wait_prior_run() {
  if [[ -n "$WAIT_FOR_SERIAL_PID" ]]; then
    stage wait_prior_pid running "pid=$WAIT_FOR_SERIAL_PID"
    while kill -0 "$WAIT_FOR_SERIAL_PID" 2>/dev/null; do
      sleep 120
    done
    stage wait_prior_pid done "pid=$WAIT_FOR_SERIAL_PID exited"
  fi

  if [[ -n "$WAIT_FOR_STATUS_ROOT" ]]; then
    local prior_status="$WAIT_FOR_STATUS_ROOT/status.tsv"
    stage wait_prior_status running "$prior_status"
    while true; do
      if [[ -f "$prior_status" ]] && grep -q $'\tcomplete\tdone\t' "$prior_status"; then
        stage wait_prior_status done "$prior_status"
        return
      fi
      sleep 120
    done
  fi
}

run_confiqa_random_controls() {
  local train_root="$SERIAL_ROOT/confiqa_random_head_train"
  local dev_root="$SERIAL_ROOT/confiqa_random_head_alpha_dev"
  local test_root="$SERIAL_ROOT/confiqa_random_head_test"
  local alpha_plan="$dev_root/alpha_plan.tsv"

  wait_gpu_free confiqa_random_train
  stage confiqa_random_train running "$train_root"
  env GPU="$GPU" RUN_ROOT="$train_root" TASKS="qa mr mc" SHIFTS="$SHIFTS" \
    TRAIN_SHIFT_CONTROLS=0 RANDOM_HEAD_CONTROLS="$RANDOM_CONTROLS" DISCOVERY_BASE="$DISCOVERY_BASE" \
    bash scripts/paper/09_run_pre_o_head_site_shift.sh \
    >"$SERIAL_ROOT/logs/confiqa_random_train.log" 2>&1
  stage confiqa_random_train done "$train_root"

  wait_gpu_free confiqa_random_alpha_dev
  stage confiqa_random_alpha_dev running "$dev_root"
  env GPU="$GPU" SITE_RUN_ROOT="$train_root" RUN_ROOT="$dev_root" TASKS="qa mr mc" \
    INCLUDE_BASE=0 INCLUDE_SELECTED=0 INCLUDE_SHIFT_CONTROLS=0 EXTRA_OPEN_CONTROLS="$RANDOM_CONTROLS" \
    DISCOVERY_BASE="$DISCOVERY_BASE" \
    CONFIQA_EVAL_SOURCE_START=301 CONFIQA_EVAL_SOURCE_ROWS=120 CONFIQA_EVAL_ROWS=120 \
    ALPHA_SWEEP="$ALPHA_SWEEP" \
    bash scripts/paper/10_run_pre_o_head_site_open_generation.sh \
    >"$SERIAL_ROOT/logs/confiqa_random_alpha_dev.log" 2>&1
  "$PY" scripts/paper/12_select_site_alpha.py \
    --summary-tsv "$dev_root/open_generation_summary.tsv" \
    --out-tsv "$alpha_plan" \
    --confiqa-score logic_score \
    >"$SERIAL_ROOT/logs/confiqa_random_alpha_select.log" 2>&1
  stage confiqa_random_alpha_dev done "$alpha_plan"

  wait_gpu_free confiqa_random_test
  stage confiqa_random_test running "$test_root"
  env GPU="$GPU" SITE_RUN_ROOT="$train_root" RUN_ROOT="$test_root" TASKS="qa mr mc" \
    INCLUDE_BASE=0 INCLUDE_SELECTED=0 INCLUDE_SHIFT_CONTROLS=0 EXTRA_OPEN_CONTROLS="$RANDOM_CONTROLS" \
    DISCOVERY_BASE="$DISCOVERY_BASE" \
    CONFIQA_EVAL_SOURCE_START=1000 CONFIQA_EVAL_SOURCE_ROWS=500 CONFIQA_EVAL_ROWS=500 \
    ALPHA_PLAN_TSV="$alpha_plan" ALPHA_SWEEP="$ALPHA_SWEEP" \
    bash scripts/paper/10_run_pre_o_head_site_open_generation.sh \
    >"$SERIAL_ROOT/logs/confiqa_random_test.log" 2>&1
  stage confiqa_random_test done "$test_root"
}

run_imdb_random_controls() {
  local train_root="$SERIAL_ROOT/imdb_random_head_train"
  local test_root="$SERIAL_ROOT/imdb_random_head_test_alpha_curve"

  wait_gpu_free imdb_random_train
  stage imdb_random_train running "$train_root"
  env GPU="$GPU" RUN_ROOT="$train_root" TASKS="imdb" SHIFTS="$SHIFTS" \
    TRAIN_SHIFT_CONTROLS=0 RANDOM_HEAD_CONTROLS="$RANDOM_CONTROLS" \
    bash scripts/paper/09_run_pre_o_head_site_shift.sh \
    >"$SERIAL_ROOT/logs/imdb_random_train.log" 2>&1
  stage imdb_random_train done "$train_root"

  wait_gpu_free imdb_random_test_alpha_curve
  stage imdb_random_test_alpha_curve running "$test_root"
  env GPU="$GPU" SITE_RUN_ROOT="$train_root" RUN_ROOT="$test_root" TASKS="imdb" \
    INCLUDE_BASE=0 INCLUDE_SELECTED=0 INCLUDE_SHIFT_CONTROLS=0 EXTRA_OPEN_CONTROLS="$RANDOM_CONTROLS" \
    IMDB_PROMPTS_SPLIT=test IMDB_EVAL_SPLIT=eval IMDB_EVAL_ROWS=2048 \
    ALPHA_SWEEP="$ALPHA_SWEEP" \
    bash scripts/paper/10_run_pre_o_head_site_open_generation.sh \
    >"$SERIAL_ROOT/logs/imdb_random_test.log" 2>&1
  stage imdb_random_test_alpha_curve done "$test_root"
}

stage preflight running "$SERIAL_ROOT"
"$PY" -m py_compile \
  scripts/paper/12_select_site_alpha.py \
  src/screscomp/cli/cecm_train_attention_head_actuator.py \
  src/screscomp/cli/cecm_run_joint_actuator_generation.py \
  src/screscomp/cli/run_imdb_sentiment_actuator_generation.py
stage preflight done "$SERIAL_ROOT"

wait_prior_run
run_confiqa_random_controls
run_imdb_random_controls

stage complete done "$SERIAL_ROOT"
log "done serial_root=$SERIAL_ROOT"
