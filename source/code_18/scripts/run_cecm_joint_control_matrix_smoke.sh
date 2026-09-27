#!/usr/bin/env bash
set -euo pipefail

# Overnight matrix for refined attention heads + prior_mlp hard actuator.
# Includes:
#   - head scaling: suppress / boost / mixed
#   - head trained vectors: suppress / boost / mixed
#   - prior_mlp trained vector alone
#   - prior_mlp + each head control

cd "$(dirname "${BASH_SOURCE[0]}")/.."

GPU="${GPU:-2}"
MODEL="${MODEL:-LOCAL_HOME/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct}"
PAIRS="${PAIRS:-data_ckplug/cecm_source_context_over_prior_base_step1/pairs.csv}"
DATA="${DATA:-data_ckplug/confiqa_large_ck_default_test5000/02_prompts/confiqa_open_rows.jsonl}"
ROOT="${ROOT:-data_ckplug/cecm_joint_control_matrix_v0}"

PRIOR_MLP_ACTUATOR="${PRIOR_MLP_ACTUATOR:-data_ckplug/cecm_hard_vector_smoke_v0/train_prior_mlp/fixed_actuator.pt}"
SUPPRESS_HEADS="${SUPPRESS_HEADS:-L9.attn.h17,L31.attn.h11,L9.attn.h12,L9.attn.h10}"
BOOST_HEADS="${BOOST_HEADS:-L31.attn.h14,L31.attn.h5,L9.attn.h3,L9.attn.h29}"
MIXED_HEADS="${MIXED_HEADS:-${SUPPRESS_HEADS},${BOOST_HEADS}}"

EVENT="${EVENT:-source_context_over_prior}"
GEN_PROMPT="${GEN_PROMPT:-base_rag}"
PRIOR_SOURCE="${PRIOR_SOURCE:-dataset_orig}"
TRAIN_ROWS="${TRAIN_ROWS:-80}"
VAL_ROWS="${VAL_ROWS:-40}"
EPOCHS="${EPOCHS:-2}"
LR="${LR:-0.05}"
LAMBDA_NORM="${LAMBDA_NORM:-1e-4}"
TRAIN_APPLY_MODE="${TRAIN_APPLY_MODE:-decision_tokens}"
SCORE_MODE="${SCORE_MODE:-answer_rest_margin}"

MAX_ROWS="${MAX_ROWS:-200}"
APPLY_MODES="${APPLY_MODES:-prefill,first_decode}"
MLP_ALPHA="${MLP_ALPHA:-0.5}"
HEAD_ALPHA="${HEAD_ALPHA:-0.5}"
SKIP_HEAD_TRAIN="${SKIP_HEAD_TRAIN:-0}"

mkdir -p "$ROOT"
MASTER_LOG="$ROOT/master.log"
STATUS="$ROOT/status.tsv"

test -d "$MODEL"
test -f "$PAIRS"
test -f "$DATA"
test -f "$PRIOR_MLP_ACTUATOR"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  deactivate || true
fi
source LOCAL_HOME/anaconda3/etc/profile.d/conda.sh
conda activate screscomp
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

{
  echo -e "time\tstage\tname\tstatus"
  echo -e "$(date -Is)\tinit\tall\tstarting"
} > "$STATUS"

echo "[$(date -Is)] root=$ROOT max_rows=$MAX_ROWS apply_modes=$APPLY_MODES mlp_alpha=$MLP_ALPHA head_alpha=$HEAD_ALPHA" | tee -a "$MASTER_LOG"
echo "[$(date -Is)] suppress=$SUPPRESS_HEADS boost=$BOOST_HEADS" | tee -a "$MASTER_LOG"

train_head_set() {
  local name="$1"
  local heads="$2"
  local out="$ROOT/train_head_${name}"
  mkdir -p "$out"
  if [[ "$SKIP_HEAD_TRAIN" == "1" ]]; then
    test -f "$out/head_actuator.pt"
    echo "[$(date -Is)] skip head train name=$name using=$out/head_actuator.pt" | tee -a "$MASTER_LOG"
    echo -e "$(date -Is)\ttrain_head\t$name\tskipped" >> "$STATUS"
    return
  fi
  echo "[$(date -Is)] train head actuator name=$name heads=$heads" | tee -a "$MASTER_LOG"
  echo -e "$(date -Is)\ttrain_head\t$name\trunning" >> "$STATUS"
  CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.cecm_train_attention_head_actuator \
    --model "$MODEL" \
    --pairs-csv "$PAIRS" \
    --event "$EVENT" \
    --heads "$heads" \
    --train-split train \
    --val-split val \
    --max-train-rows "$TRAIN_ROWS" \
    --max-val-rows "$VAL_ROWS" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --lambda-norm "$LAMBDA_NORM" \
    --alpha-train 1.0 \
    --apply-mode "$TRAIN_APPLY_MODE" \
    --score-mode "$SCORE_MODE" \
    --alpha-sweep "0,0.25,0.5,1.0,1.5" \
    --torch-dtype bfloat16 \
    --device cuda \
    --out-dir "$out" \
    > "$out/train.log" 2>&1
  echo "[$(date -Is)] done head train name=$name" | tee -a "$MASTER_LOG"
  echo -e "$(date -Is)\ttrain_head\t$name\tdone" >> "$STATUS"
}

train_head_set suppress "$SUPPRESS_HEADS"
train_head_set boost "$BOOST_HEADS"
train_head_set mixed "$MIXED_HEADS"

HEAD_SCALINGS="suppress=${SUPPRESS_HEADS//,/:0,}:0;boost=${BOOST_HEADS//,/:1.5,}:1.5;mixed=${SUPPRESS_HEADS//,/:0,}:0,${BOOST_HEADS//,/:1.5,}:1.5"
HEAD_ACTUATORS="suppress=$ROOT/train_head_suppress/head_actuator.pt;boost=$ROOT/train_head_boost/head_actuator.pt;mixed=$ROOT/train_head_mixed/head_actuator.pt"
COMP_ACTUATORS="prior_mlp=$PRIOR_MLP_ACTUATOR"
CONTROLS="baseline=;\
mlp=comp:prior_mlp:${MLP_ALPHA};\
hs_suppress=head_scale:suppress;\
hs_boost=head_scale:boost;\
hs_mixed=head_scale:mixed;\
ha_suppress=head_act:suppress:${HEAD_ALPHA};\
ha_boost=head_act:boost:${HEAD_ALPHA};\
ha_mixed=head_act:mixed:${HEAD_ALPHA};\
mlp_hs_suppress=comp:prior_mlp:${MLP_ALPHA}+head_scale:suppress;\
mlp_hs_boost=comp:prior_mlp:${MLP_ALPHA}+head_scale:boost;\
mlp_hs_mixed=comp:prior_mlp:${MLP_ALPHA}+head_scale:mixed;\
mlp_ha_suppress=comp:prior_mlp:${MLP_ALPHA}+head_act:suppress:${HEAD_ALPHA};\
mlp_ha_boost=comp:prior_mlp:${MLP_ALPHA}+head_act:boost:${HEAD_ALPHA};\
mlp_ha_mixed=comp:prior_mlp:${MLP_ALPHA}+head_act:mixed:${HEAD_ALPHA};\
ha_mixed_hs_mixed=head_act:mixed:${HEAD_ALPHA}+head_scale:mixed;\
mlp_ha_mixed_hs_mixed=comp:prior_mlp:${MLP_ALPHA}+head_act:mixed:${HEAD_ALPHA}+head_scale:mixed"

IFS=',' read -ra MODES <<< "$APPLY_MODES"
for mode in "${MODES[@]}"; do
  mode="$(echo "$mode" | xargs)"
  [[ -n "$mode" ]] || continue
  out="$ROOT/gen_${mode}"
  mkdir -p "$out"
  echo "[$(date -Is)] joint generation mode=$mode out=$out" | tee -a "$MASTER_LOG"
  echo -e "$(date -Is)\tgeneration\t$mode\trunning" >> "$STATUS"
  CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.cecm_run_joint_actuator_generation \
    --model "$MODEL" \
    --eval-open-rows "$DATA" \
    --component-actuators "$COMP_ACTUATORS" \
    --head-actuators "$HEAD_ACTUATORS" \
    --head-scalings "$HEAD_SCALINGS" \
    --controls "$CONTROLS" \
    --generation-prompt-key "$GEN_PROMPT" \
    --prior-source "$PRIOR_SOURCE" \
    --split val \
    --start 0 \
    --max-rows "$MAX_ROWS" \
    --generation-apply-mode "$mode" \
    --max-new-tokens 64 \
    --stop-strings "Q:" \
    --torch-dtype bfloat16 \
    --device cuda \
    --out-dir "$out" \
    > "$out/generation.log" 2>&1
  echo "[$(date -Is)] done joint generation mode=$mode" | tee -a "$MASTER_LOG"
  echo -e "$(date -Is)\tgeneration\t$mode\tdone" >> "$STATUS"
done

echo "[$(date -Is)] all done" | tee -a "$MASTER_LOG"
find "$ROOT" -maxdepth 2 \( -name alpha_summary.csv -o -name generation_summary.csv -o -name control_plan.csv \) -print | sort
