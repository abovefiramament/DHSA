#!/usr/bin/env bash
set -euo pipefail

# Staged control smoke:
#   1) train/use prior_mlp as a frozen prefill-only state offset
#   2) train attention head actuators on top of that conditioned state
#   3) evaluate MLP-prefill, conditioned heads, and MLP-prefill + conditioned heads

cd "$(dirname "${BASH_SOURCE[0]}")/.."

GPU="${GPU:-3}"
MODEL="${MODEL:-LOCAL_HOME/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct}"
PAIRS="${PAIRS:-data_ckplug/cecm_source_context_over_prior_base_step1/pairs.csv}"
DATA="${DATA:-data_ckplug/confiqa_large_ck_default_test5000/02_prompts/confiqa_open_rows.jsonl}"
ROOT="${ROOT:-data_ckplug/cecm_mlp_prefill_then_head_smoke_v0}"

PRIOR_MLP_COMPONENTS="${PRIOR_MLP_COMPONENTS:-configs/cecm_hard_prior_mlp_components.csv}"
SUPPRESS_HEADS="${SUPPRESS_HEADS:-L9.attn.h17,L31.attn.h11,L9.attn.h12,L9.attn.h10}"
BOOST_HEADS="${BOOST_HEADS:-L31.attn.h14,L31.attn.h5,L9.attn.h3,L9.attn.h29}"

EVENT="${EVENT:-source_context_over_prior}"
GEN_PROMPT="${GEN_PROMPT:-base_rag}"
PRIOR_SOURCE="${PRIOR_SOURCE:-dataset_orig}"
TRAIN_ROWS="${TRAIN_ROWS:-40}"
VAL_ROWS="${VAL_ROWS:-10}"
EPOCHS="${EPOCHS:-2}"
LR="${LR:-0.05}"
LAMBDA_NORM="${LAMBDA_NORM:-1e-4}"
SCORE_MODE="${SCORE_MODE:-answer_rest_margin}"
ALPHA_SWEEP="${ALPHA_SWEEP:-0}"

MLP_TRAIN_APPLY_MODE="${MLP_TRAIN_APPLY_MODE:-decision_tokens}"
MLP_GEN_APPLY_MODE="${MLP_GEN_APPLY_MODE:-prefill}"
HEAD_TRAIN_APPLY_MODE="${HEAD_TRAIN_APPLY_MODE:-decision_tokens}"
HEAD_GEN_APPLY_MODE="${HEAD_GEN_APPLY_MODE:-all}"
MLP_ALPHA="${MLP_ALPHA:-0.5}"
HEAD_ALPHA="${HEAD_ALPHA:-0.5}"
MAX_ROWS="${MAX_ROWS:-60}"
SKIP_MLP_TRAIN="${SKIP_MLP_TRAIN:-0}"
SKIP_HEAD_TRAIN="${SKIP_HEAD_TRAIN:-0}"
MLP_ACTUATOR="${MLP_ACTUATOR:-}"

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

echo "[$(date -Is)] root=$ROOT train_rows=$TRAIN_ROWS val_rows=$VAL_ROWS max_rows=$MAX_ROWS" | tee -a "$MASTER_LOG"
echo "[$(date -Is)] mlp_train=$MLP_TRAIN_APPLY_MODE mlp_gen=$MLP_GEN_APPLY_MODE head_train=$HEAD_TRAIN_APPLY_MODE head_gen=$HEAD_GEN_APPLY_MODE mlp_alpha=$MLP_ALPHA head_alpha=$HEAD_ALPHA" | tee -a "$MASTER_LOG"

MLP_OUT="$ROOT/train_mlp_${MLP_TRAIN_APPLY_MODE}"
if [[ -z "$MLP_ACTUATOR" ]]; then
  MLP_ACTUATOR="$MLP_OUT/fixed_actuator.pt"
fi

if [[ "$SKIP_MLP_TRAIN" != "1" ]]; then
  mkdir -p "$MLP_OUT"
  echo "[$(date -Is)] train mlp out=$MLP_OUT" | tee -a "$MASTER_LOG"
  echo -e "$(date -Is)\ttrain_mlp\tprior_mlp\trunning" >> "$STATUS"
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
    --apply-mode "$MLP_TRAIN_APPLY_MODE" \
    --score-mode "$SCORE_MODE" \
    --alpha-sweep "$ALPHA_SWEEP" \
    --torch-dtype bfloat16 \
    --device cuda \
    --out-dir "$MLP_OUT" \
    > "$MLP_OUT/train.log" 2>&1
  echo -e "$(date -Is)\ttrain_mlp\tprior_mlp\tdone" >> "$STATUS"
fi
test -f "$MLP_ACTUATOR"

train_conditioned_head() {
  local name="$1"
  local heads="$2"
  local out="$ROOT/train_head_${name}_conditioned"
  mkdir -p "$out"
  if [[ "$SKIP_HEAD_TRAIN" == "1" ]]; then
    test -f "$out/head_actuator.pt"
    echo -e "$(date -Is)\ttrain_head\t$name\tskipped" >> "$STATUS"
    return
  fi
  echo "[$(date -Is)] train conditioned head name=$name heads=$heads" | tee -a "$MASTER_LOG"
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
    --apply-mode "$HEAD_TRAIN_APPLY_MODE" \
    --score-mode "$SCORE_MODE" \
    --alpha-sweep "$ALPHA_SWEEP" \
    --frozen-component-actuator "$MLP_ACTUATOR" \
    --frozen-component-alpha "$MLP_ALPHA" \
    --frozen-component-apply-mode "$MLP_GEN_APPLY_MODE" \
    --torch-dtype bfloat16 \
    --device cuda \
    --out-dir "$out" \
    > "$out/train.log" 2>&1
  echo -e "$(date -Is)\ttrain_head\t$name\tdone" >> "$STATUS"
}

train_conditioned_head suppress "$SUPPRESS_HEADS"
train_conditioned_head boost "$BOOST_HEADS"

COMP_ACTUATORS="prior_mlp=$MLP_ACTUATOR"
HEAD_ACTUATORS="suppress=$ROOT/train_head_suppress_conditioned/head_actuator.pt;boost=$ROOT/train_head_boost_conditioned/head_actuator.pt"
CONTROLS="baseline=;\
mlp_prefill=comp:prior_mlp:${MLP_ALPHA}:${MLP_GEN_APPLY_MODE};\
conditioned_head_only=head_act:suppress:${HEAD_ALPHA}:${HEAD_GEN_APPLY_MODE}+head_act:boost:${HEAD_ALPHA}:${HEAD_GEN_APPLY_MODE};\
mlp_prefill_then_head=comp:prior_mlp:${MLP_ALPHA}:${MLP_GEN_APPLY_MODE}+head_act:suppress:${HEAD_ALPHA}:${HEAD_GEN_APPLY_MODE}+head_act:boost:${HEAD_ALPHA}:${HEAD_GEN_APPLY_MODE}"

echo "[$(date -Is)] generation controls=$CONTROLS" | tee -a "$MASTER_LOG"
echo -e "$(date -Is)\tgeneration\tall\trunning" >> "$STATUS"
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
  --generation-apply-mode "$HEAD_GEN_APPLY_MODE" \
  --max-new-tokens 64 \
  --stop-strings "Q:" \
  --torch-dtype bfloat16 \
  --device cuda \
  --out-dir "$ROOT" \
  > "$ROOT/generation.log" 2>&1
echo -e "$(date -Is)\tgeneration\tall\tdone" >> "$STATUS"

echo "[$(date -Is)] done" | tee -a "$MASTER_LOG"
find "$ROOT" -maxdepth 1 \( -name generation_summary.csv -o -name control_plan.csv -o -name run_summary.csv \) -print | sort
