#!/usr/bin/env bash
set -euo pipefail

# Quick smoke: separately trained positive/negative head actuators used together.
# No retraining; reuses the overnight matrix payloads.

cd "$(dirname "${BASH_SOURCE[0]}")/.."

GPU="${GPU:-3}"
MODEL="${MODEL:-LOCAL_HOME/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct}"
DATA="${DATA:-data_ckplug/confiqa_large_ck_default_test5000/02_prompts/confiqa_open_rows.jsonl}"
ROOT="${ROOT:-data_ckplug/cecm_separate_head_joint_smoke_v0}"

PRIOR_MLP_ACTUATOR="${PRIOR_MLP_ACTUATOR:-data_ckplug/cecm_hard_vector_smoke_v0/train_prior_mlp/fixed_actuator.pt}"
HEAD_SUPPRESS_ACTUATOR="${HEAD_SUPPRESS_ACTUATOR:-data_ckplug/cecm_joint_control_matrix_v0/train_head_suppress/head_actuator.pt}"
HEAD_BOOST_ACTUATOR="${HEAD_BOOST_ACTUATOR:-data_ckplug/cecm_joint_control_matrix_v0/train_head_boost/head_actuator.pt}"
HEAD_MIXED_ACTUATOR="${HEAD_MIXED_ACTUATOR:-data_ckplug/cecm_joint_control_matrix_v0/train_head_mixed/head_actuator.pt}"

GEN_PROMPT="${GEN_PROMPT:-base_rag}"
PRIOR_SOURCE="${PRIOR_SOURCE:-dataset_orig}"
MAX_ROWS="${MAX_ROWS:-80}"
APPLY_MODE="${APPLY_MODE:-prefill}"
MLP_ALPHA="${MLP_ALPHA:-0.5}"
HEAD_ALPHA="${HEAD_ALPHA:-0.5}"

mkdir -p "$ROOT"
MASTER_LOG="$ROOT/master.log"
STATUS="$ROOT/status.tsv"

test -d "$MODEL"
test -f "$DATA"
test -f "$PRIOR_MLP_ACTUATOR"
test -f "$HEAD_SUPPRESS_ACTUATOR"
test -f "$HEAD_BOOST_ACTUATOR"
test -f "$HEAD_MIXED_ACTUATOR"

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

COMP_ACTUATORS="prior_mlp=$PRIOR_MLP_ACTUATOR"
HEAD_ACTUATORS="suppress=$HEAD_SUPPRESS_ACTUATOR;boost=$HEAD_BOOST_ACTUATOR;mixed=$HEAD_MIXED_ACTUATOR"
CONTROLS="baseline=;\
ha_suppress=head_act:suppress:${HEAD_ALPHA};\
ha_boost=head_act:boost:${HEAD_ALPHA};\
ha_sep=head_act:suppress:${HEAD_ALPHA}+head_act:boost:${HEAD_ALPHA};\
ha_mixed=head_act:mixed:${HEAD_ALPHA};\
mlp=comp:prior_mlp:${MLP_ALPHA};\
mlp_ha_suppress=comp:prior_mlp:${MLP_ALPHA}+head_act:suppress:${HEAD_ALPHA};\
mlp_ha_boost=comp:prior_mlp:${MLP_ALPHA}+head_act:boost:${HEAD_ALPHA};\
mlp_ha_sep=comp:prior_mlp:${MLP_ALPHA}+head_act:suppress:${HEAD_ALPHA}+head_act:boost:${HEAD_ALPHA};\
mlp_ha_mixed=comp:prior_mlp:${MLP_ALPHA}+head_act:mixed:${HEAD_ALPHA}"

echo "[$(date -Is)] root=$ROOT max_rows=$MAX_ROWS apply_mode=$APPLY_MODE mlp_alpha=$MLP_ALPHA head_alpha=$HEAD_ALPHA" | tee -a "$MASTER_LOG"
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
  --generation-apply-mode "$APPLY_MODE" \
  --max-new-tokens 64 \
  --stop-strings "Q:" \
  --torch-dtype bfloat16 \
  --device cuda \
  --out-dir "$ROOT" \
  > "$ROOT/run.log" 2>&1

echo "[$(date -Is)] done" | tee -a "$MASTER_LOG"
echo -e "$(date -Is)\trun\tdone" >> "$STATUS"
find "$ROOT" -maxdepth 1 \( -name generation_summary.csv -o -name control_plan.csv -o -name run_summary.csv \) -print | sort
