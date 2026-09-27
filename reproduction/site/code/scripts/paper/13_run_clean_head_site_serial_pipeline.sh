#!/usr/bin/env bash
set -euo pipefail

# Single-GPU serial clean head-site pipeline.
#
# Order:
#   1. Wait for the clean ConFiQA disc120 root if requested.
#   2. Train and evaluate ConFiQA QA/MR/MC pre-O true-head site controls.
#   3. Train and evaluate IMDb pre-O true-head site controls.
#
# This script intentionally keeps all heavy stages serial on one GPU.

export LC_ALL=C.UTF-8
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

REPO_ROOT="${REPO_ROOT:-LOCAL_HOME/RPEC/projects/screscomp}"
cd "$REPO_ROOT"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

PY="${PY:-LOCAL_HOME/anaconda3/envs/screscomp/bin/python}"
GPU="${GPU:-1}"
SERIAL_ROOT="${SERIAL_ROOT:-/dev/shm/screscomp_runs/clean_head_site_serial_$(date -u +%Y%m%d_%H%M%S)}"
WAIT_CONFIQA_BOOTSTRAP_PID="${WAIT_CONFIQA_BOOTSTRAP_PID:-}"
CONFIQA_DISCOVERY_ROOT="${CONFIQA_DISCOVERY_ROOT:-data_ckplug/cast_confiqa_all3_disc120_heldout500_v0}"
SHIFTS="${SHIFTS:-1 2 3}"
ALPHA_SWEEP="${ALPHA_SWEEP:-0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}"
MAX_GPU_USED_MIB="${MAX_GPU_USED_MIB:-256}"

mkdir -p "$SERIAL_ROOT/logs"
STATUS="$SERIAL_ROOT/status.tsv"
MASTER_LOG="$SERIAL_ROOT/serial_pipeline.log"
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

wait_confiqa_root() {
  if [[ -n "$WAIT_CONFIQA_BOOTSTRAP_PID" ]]; then
    stage wait_confiqa_bootstrap running "pid=$WAIT_CONFIQA_BOOTSTRAP_PID"
    while kill -0 "$WAIT_CONFIQA_BOOTSTRAP_PID" 2>/dev/null; do
      sleep 120
    done
    stage wait_confiqa_bootstrap done "pid=$WAIT_CONFIQA_BOOTSTRAP_PID exited"
  fi

  stage wait_confiqa_artifacts running "$CONFIQA_DISCOVERY_ROOT"
  for task in qa mr mc; do
    test -f "$CONFIQA_DISCOVERY_ROOT/$task/discovery/selected/selected.env"
    test -f "$CONFIQA_DISCOVERY_ROOT/$task/train_decision_tokens_head_suppress/head_actuator.pt"
    test -f "$CONFIQA_DISCOVERY_ROOT/$task/train_decision_tokens_head_boost/head_actuator.pt"
    test -f "$CONFIQA_DISCOVERY_ROOT/$task/pairs/pairs.csv"
  done
  stage wait_confiqa_artifacts done "$CONFIQA_DISCOVERY_ROOT"
}

run_confiqa_site() {
  local train_root="$SERIAL_ROOT/confiqa_site_train"
  local dev_root="$SERIAL_ROOT/confiqa_site_alpha_dev"
  local test_root="$SERIAL_ROOT/confiqa_site_test"
  local alpha_plan="$dev_root/alpha_plan.tsv"

  wait_confiqa_root

  wait_gpu_free confiqa_train
  stage confiqa_train running "$train_root"
  env GPU="$GPU" RUN_ROOT="$train_root" TASKS="qa mr mc" SHIFTS="$SHIFTS" \
    DISCOVERY_BASE="$CONFIQA_DISCOVERY_ROOT" \
    bash scripts/paper/09_run_pre_o_head_site_shift.sh \
    >"$SERIAL_ROOT/logs/confiqa_train.log" 2>&1
  stage confiqa_train done "$train_root"

  wait_gpu_free confiqa_alpha_dev
  stage confiqa_alpha_dev running "$dev_root"
  env GPU="$GPU" SITE_RUN_ROOT="$train_root" RUN_ROOT="$dev_root" TASKS="qa mr mc" OPEN_SHIFTS="$SHIFTS" \
    DISCOVERY_BASE="$CONFIQA_DISCOVERY_ROOT" \
    CONFIQA_EVAL_SOURCE_START=301 CONFIQA_EVAL_SOURCE_ROWS=120 CONFIQA_EVAL_ROWS=120 \
    ALPHA_SWEEP="$ALPHA_SWEEP" \
    bash scripts/paper/10_run_pre_o_head_site_open_generation.sh \
    >"$SERIAL_ROOT/logs/confiqa_alpha_dev.log" 2>&1
  "$PY" scripts/paper/12_select_site_alpha.py \
    --summary-tsv "$dev_root/open_generation_summary.tsv" \
    --out-tsv "$alpha_plan" \
    --confiqa-score logic_score \
    >"$SERIAL_ROOT/logs/confiqa_alpha_select.log" 2>&1
  stage confiqa_alpha_dev done "$alpha_plan"

  wait_gpu_free confiqa_test
  stage confiqa_test running "$test_root"
  env GPU="$GPU" SITE_RUN_ROOT="$train_root" RUN_ROOT="$test_root" TASKS="qa mr mc" OPEN_SHIFTS="$SHIFTS" \
    DISCOVERY_BASE="$CONFIQA_DISCOVERY_ROOT" \
    CONFIQA_EVAL_SOURCE_START=1000 CONFIQA_EVAL_SOURCE_ROWS=500 CONFIQA_EVAL_ROWS=500 \
    ALPHA_PLAN_TSV="$alpha_plan" ALPHA_SWEEP="$ALPHA_SWEEP" \
    bash scripts/paper/10_run_pre_o_head_site_open_generation.sh \
    >"$SERIAL_ROOT/logs/confiqa_test.log" 2>&1
  stage confiqa_test done "$test_root"
}

run_imdb_site() {
  local train_root="$SERIAL_ROOT/imdb_site_train"
  local test_root="$SERIAL_ROOT/imdb_site_test_alpha_curve"

  wait_gpu_free imdb_train
  stage imdb_train running "$train_root"
  env GPU="$GPU" RUN_ROOT="$train_root" TASKS="imdb" SHIFTS="$SHIFTS" \
    bash scripts/paper/09_run_pre_o_head_site_shift.sh \
    >"$SERIAL_ROOT/logs/imdb_train.log" 2>&1
  stage imdb_train done "$train_root"

  wait_gpu_free imdb_test_alpha_curve
  stage imdb_test_alpha_curve running "$test_root"
  env GPU="$GPU" SITE_RUN_ROOT="$train_root" RUN_ROOT="$test_root" TASKS="imdb" OPEN_SHIFTS="$SHIFTS" \
    IMDB_PROMPTS_SPLIT=test IMDB_EVAL_SPLIT=eval IMDB_EVAL_ROWS=2048 \
    ALPHA_SWEEP="$ALPHA_SWEEP" \
    bash scripts/paper/10_run_pre_o_head_site_open_generation.sh \
  >"$SERIAL_ROOT/logs/imdb_test.log" 2>&1
  stage imdb_test_alpha_curve done "$test_root"
}

stage preflight running "$SERIAL_ROOT"
"$PY" -m py_compile \
  scripts/paper/12_select_site_alpha.py \
  src/screscomp/cli/cecm_train_attention_head_actuator.py \
  src/screscomp/cli/cecm_run_joint_actuator_generation.py \
  src/screscomp/cli/run_imdb_sentiment_actuator_generation.py
stage preflight done "$SERIAL_ROOT"

run_confiqa_site
run_imdb_site

stage complete done "$SERIAL_ROOT"
log "done serial_root=$SERIAL_ROOT"
