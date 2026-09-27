#!/usr/bin/env bash
set -euo pipefail

# Focused smoke for heterogeneous timing:
#   head_sep uses persistent all-step control
#   MLP uses prefill-only control
# This is the fair test for whether source-control heads and answer-boundary MLP
# are complementary instead of forcing both through one global generation timing.

cd "$(dirname "${BASH_SOURCE[0]}")/.."

GPU="${GPU:-3}"
MODEL="${MODEL:-LOCAL_HOME/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct}"
DATA="${DATA:-data_ckplug/confiqa_large_ck_default_test5000/02_prompts/confiqa_open_rows.jsonl}"
ROOT="${ROOT:-data_ckplug/cecm_per_actuator_timing_smoke_v0}"

MLP_ACTUATOR="${MLP_ACTUATOR:-data_ckplug/cecm_hard_vector_smoke_v0/train_prior_mlp/fixed_actuator.pt}"
HEAD_SUPPRESS_ACTUATOR="${HEAD_SUPPRESS_ACTUATOR:-data_ckplug/cecm_timing_retrain_smoke_v0/train_all_head_suppress/head_actuator.pt}"
HEAD_BOOST_ACTUATOR="${HEAD_BOOST_ACTUATOR:-data_ckplug/cecm_timing_retrain_smoke_v0/train_all_head_boost/head_actuator.pt}"

GEN_PROMPT="${GEN_PROMPT:-base_rag}"
PRIOR_SOURCE="${PRIOR_SOURCE:-dataset_orig}"
MAX_ROWS="${MAX_ROWS:-40}"
MLP_ALPHA="${MLP_ALPHA:-0.5}"
HEAD_ALPHA="${HEAD_ALPHA:-0.5}"
GLOBAL_APPLY_MODE="${GLOBAL_APPLY_MODE:-prefill}"

mkdir -p "$ROOT"
MASTER_LOG="$ROOT/master.log"
STATUS="$ROOT/status.tsv"

test -d "$MODEL"
test -f "$DATA"
test -f "$MLP_ACTUATOR"
test -f "$HEAD_SUPPRESS_ACTUATOR"
test -f "$HEAD_BOOST_ACTUATOR"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  deactivate || true
fi
source LOCAL_HOME/anaconda3/etc/profile.d/conda.sh
conda activate screscomp
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

{
  echo -e "time\tstage\tstatus"
  echo -e "$(date -Is)\tinit\tstarting"
} > "$STATUS"

COMP_ACTUATORS="prior_mlp=$MLP_ACTUATOR"
HEAD_ACTUATORS="suppress=$HEAD_SUPPRESS_ACTUATOR;boost=$HEAD_BOOST_ACTUATOR"
CONTROLS="baseline=;\
head_sep_all=head_act:suppress:${HEAD_ALPHA}:all+head_act:boost:${HEAD_ALPHA}:all;\
mlp_prefill=comp:prior_mlp:${MLP_ALPHA}:prefill;\
head_all_mlp_prefill=head_act:suppress:${HEAD_ALPHA}:all+head_act:boost:${HEAD_ALPHA}:all+comp:prior_mlp:${MLP_ALPHA}:prefill"

echo "[$(date -Is)] root=$ROOT max_rows=$MAX_ROWS mlp_alpha=$MLP_ALPHA head_alpha=$HEAD_ALPHA" | tee -a "$MASTER_LOG"
echo "[$(date -Is)] controls=$CONTROLS" | tee -a "$MASTER_LOG"
echo -e "$(date -Is)\trun\trunning" >> "$STATUS"

CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.cecm_run_joint_actuator_generation \
  --model "$MODEL" \
  --eval-open-rows "$DATA" \
  --component-actuators "$COMP_ACTUATORS" \
  --head-actuators "$HEAD_ACTUATORS" \
  --controls "$CONTROLS" \
  --generation-prompt-key "$GEN_PROMPT" \
  --prior-source "$PRIOR_SOURCE" \
  --split val \
  --start 0 \
  --max-rows "$MAX_ROWS" \
  --generation-apply-mode "$GLOBAL_APPLY_MODE" \
  --max-new-tokens 64 \
  --stop-strings "Q:" \
  --torch-dtype bfloat16 \
  --device cuda \
  --out-dir "$ROOT" \
  > "$ROOT/run.log" 2>&1

echo "[$(date -Is)] done" | tee -a "$MASTER_LOG"
echo -e "$(date -Is)\trun\tdone" >> "$STATUS"
find "$ROOT" -maxdepth 1 \( -name generation_summary.csv -o -name control_plan.csv -o -name run_summary.csv \) -print | sort
