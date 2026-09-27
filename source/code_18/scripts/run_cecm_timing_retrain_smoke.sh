#!/usr/bin/env bash
set -euo pipefail

# Timing-only retrain smoke.
# Keeps target sets fixed and retrains prior_mlp + separated head actuators under
# different training apply modes, then evaluates the same controls under selected
# generation apply modes.

cd "$(dirname "${BASH_SOURCE[0]}")/.."

GPU="${GPU:-3}"
MODEL="${MODEL:-LOCAL_HOME/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct}"
PAIRS="${PAIRS:-data_ckplug/cecm_source_context_over_prior_base_step1/pairs.csv}"
DATA="${DATA:-data_ckplug/confiqa_large_ck_default_test5000/02_prompts/confiqa_open_rows.jsonl}"
ROOT="${ROOT:-data_ckplug/cecm_timing_retrain_smoke_v0}"

PRIOR_MLP_COMPONENTS="${PRIOR_MLP_COMPONENTS:-configs/cecm_hard_prior_mlp_components.csv}"
SUPPRESS_HEADS="${SUPPRESS_HEADS:-L9.attn.h17,L31.attn.h11,L9.attn.h12,L9.attn.h10}"
BOOST_HEADS="${BOOST_HEADS:-L31.attn.h14,L31.attn.h5,L9.attn.h3,L9.attn.h29}"

EVENT="${EVENT:-source_context_over_prior}"
GEN_PROMPT="${GEN_PROMPT:-base_rag}"
PRIOR_SOURCE="${PRIOR_SOURCE:-dataset_orig}"
TRAIN_ROWS="${TRAIN_ROWS:-80}"
VAL_ROWS="${VAL_ROWS:-40}"
EPOCHS="${EPOCHS:-2}"
LR="${LR:-0.05}"
LAMBDA_NORM="${LAMBDA_NORM:-1e-4}"
SCORE_MODE="${SCORE_MODE:-answer_rest_margin}"
ALPHA_SWEEP="${ALPHA_SWEEP:-0}"

# Priority timing-only smoke:
# - decision_tokens is the old anchor on the same rows.
# - prompt_last matches the generation prefill boundary.
# - all tests whether a persistent actuator needs matching all-position training.
# Optional wider sweep: TRAIN_MODES=decision_tokens,prompt_last,prompt,all
TRAIN_MODES="${TRAIN_MODES:-decision_tokens,prompt_last,all}"
GEN_MODES="${GEN_MODES:-prefill,all,decode}"
MAX_ROWS="${MAX_ROWS:-40}"
MLP_ALPHA="${MLP_ALPHA:-0.5}"
HEAD_ALPHA="${HEAD_ALPHA:-0.5}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

mkdir -p "$ROOT"
MASTER_LOG="$ROOT/master.log"
STATUS="$ROOT/status.tsv"

test -d "$MODEL"
test -f "$PAIRS"
test -f "$DATA"
test -f "$PRIOR_MLP_COMPONENTS"

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

echo "[$(date -Is)] root=$ROOT train_modes=$TRAIN_MODES gen_modes=$GEN_MODES max_rows=$MAX_ROWS alpha_sweep=$ALPHA_SWEEP" | tee -a "$MASTER_LOG"
echo "[$(date -Is)] suppress=$SUPPRESS_HEADS boost=$BOOST_HEADS mlp_alpha=$MLP_ALPHA head_alpha=$HEAD_ALPHA" | tee -a "$MASTER_LOG"

train_mlp() {
  local mode="$1"
  local out="$ROOT/train_${mode}_prior_mlp"
  mkdir -p "$out"
  if [[ "$SKIP_TRAIN" == "1" ]]; then
    test -f "$out/fixed_actuator.pt"
    echo "[$(date -Is)] skip mlp train mode=$mode" | tee -a "$MASTER_LOG"
    echo -e "$(date -Is)\ttrain_mlp_${mode}\tprior_mlp\tskipped" >> "$STATUS"
    return
  fi
  echo "[$(date -Is)] train mlp mode=$mode out=$out" | tee -a "$MASTER_LOG"
  echo -e "$(date -Is)\ttrain_mlp_${mode}\tprior_mlp\trunning" >> "$STATUS"
  CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.cecm_train_fixed_actuator \
    --model "$MODEL" \
    --pairs-csv "$PAIRS" \
    --components-csv "$PRIOR_MLP_COMPONENTS" \
    --event "$EVENT" \
    --train-split train \
    --val-split val \
    --max-train-rows "$TRAIN_ROWS" \
    --max-val-rows "$VAL_ROWS" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --lambda-norm "$LAMBDA_NORM" \
    --alpha-train 1.0 \
    --apply-mode "$mode" \
    --score-mode "$SCORE_MODE" \
    --alpha-sweep "$ALPHA_SWEEP" \
    --torch-dtype bfloat16 \
    --device cuda \
    --out-dir "$out" \
    > "$out/train.log" 2>&1
  echo "[$(date -Is)] done mlp train mode=$mode" | tee -a "$MASTER_LOG"
  echo -e "$(date -Is)\ttrain_mlp_${mode}\tprior_mlp\tdone" >> "$STATUS"
}

train_heads() {
  local mode="$1"
  local name="$2"
  local heads="$3"
  local out="$ROOT/train_${mode}_head_${name}"
  mkdir -p "$out"
  if [[ "$SKIP_TRAIN" == "1" ]]; then
    test -f "$out/head_actuator.pt"
    echo "[$(date -Is)] skip head train mode=$mode name=$name" | tee -a "$MASTER_LOG"
    echo -e "$(date -Is)\ttrain_head_${mode}\t$name\tskipped" >> "$STATUS"
    return
  fi
  echo "[$(date -Is)] train head mode=$mode name=$name heads=$heads" | tee -a "$MASTER_LOG"
  echo -e "$(date -Is)\ttrain_head_${mode}\t$name\trunning" >> "$STATUS"
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
    --apply-mode "$mode" \
    --score-mode "$SCORE_MODE" \
    --alpha-sweep "$ALPHA_SWEEP" \
    --torch-dtype bfloat16 \
    --device cuda \
    --out-dir "$out" \
    > "$out/train.log" 2>&1
  echo "[$(date -Is)] done head train mode=$mode name=$name" | tee -a "$MASTER_LOG"
  echo -e "$(date -Is)\ttrain_head_${mode}\t$name\tdone" >> "$STATUS"
}

IFS=',' read -ra TRAIN_MODE_ARRAY <<< "$TRAIN_MODES"
IFS=',' read -ra GEN_MODE_ARRAY <<< "$GEN_MODES"

for train_mode in "${TRAIN_MODE_ARRAY[@]}"; do
  train_mode="$(echo "$train_mode" | xargs)"
  [[ -n "$train_mode" ]] || continue
  train_mlp "$train_mode"
  train_heads "$train_mode" suppress "$SUPPRESS_HEADS"
  train_heads "$train_mode" boost "$BOOST_HEADS"

  comp_actuators="prior_mlp=$ROOT/train_${train_mode}_prior_mlp/fixed_actuator.pt"
  head_actuators="suppress=$ROOT/train_${train_mode}_head_suppress/head_actuator.pt;boost=$ROOT/train_${train_mode}_head_boost/head_actuator.pt"
  controls="baseline=;mlp=comp:prior_mlp:${MLP_ALPHA};ha_sep=head_act:suppress:${HEAD_ALPHA}+head_act:boost:${HEAD_ALPHA};mlp_ha_sep=comp:prior_mlp:${MLP_ALPHA}+head_act:suppress:${HEAD_ALPHA}+head_act:boost:${HEAD_ALPHA}"

  for gen_mode in "${GEN_MODE_ARRAY[@]}"; do
    gen_mode="$(echo "$gen_mode" | xargs)"
    [[ -n "$gen_mode" ]] || continue
    out="$ROOT/gen_train_${train_mode}_gen_${gen_mode}"
    mkdir -p "$out"
    echo "[$(date -Is)] generation train_mode=$train_mode gen_mode=$gen_mode out=$out" | tee -a "$MASTER_LOG"
    echo -e "$(date -Is)\tgeneration_${train_mode}\t$gen_mode\trunning" >> "$STATUS"
    CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.cecm_run_joint_actuator_generation \
      --model "$MODEL" \
      --eval-open-rows "$DATA" \
      --component-actuators "$comp_actuators" \
      --head-actuators "$head_actuators" \
      --controls "$controls" \
      --generation-prompt-key "$GEN_PROMPT" \
      --prior-source "$PRIOR_SOURCE" \
      --split val \
      --start 0 \
      --max-rows "$MAX_ROWS" \
      --generation-apply-mode "$gen_mode" \
      --max-new-tokens 64 \
      --stop-strings "Q:" \
      --torch-dtype bfloat16 \
      --device cuda \
      --out-dir "$out" \
      > "$out/generation.log" 2>&1
    echo "[$(date -Is)] done generation train_mode=$train_mode gen_mode=$gen_mode" | tee -a "$MASTER_LOG"
    echo -e "$(date -Is)\tgeneration_${train_mode}\t$gen_mode\tdone" >> "$STATUS"
  done
done

echo "[$(date -Is)] all done" | tee -a "$MASTER_LOG"
find "$ROOT" -maxdepth 2 \( -name alpha_summary.csv -o -name generation_summary.csv -o -name control_plan.csv \) -print | sort
