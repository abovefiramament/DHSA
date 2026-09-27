#!/usr/bin/env bash
set -euo pipefail

# Train component-specific hard vectors, then evaluate them as reusable cross-sample
# actuators on open generation. Eval does not run target/high prompts.

cd "$(dirname "${BASH_SOURCE[0]}")/.."

GPU="${GPU:-2}"
MODEL="${MODEL:-LOCAL_HOME/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct}"
PAIRS="${PAIRS:-data_ckplug/cecm_source_context_over_prior_base_step1/pairs.csv}"
DATA="${DATA:-data_ckplug/confiqa_large_ck_default_test5000/02_prompts/confiqa_open_rows.jsonl}"
ROOT="${ROOT:-data_ckplug/cecm_hard_vector_smoke_v0}"

HARD_GROUPS="${HARD_GROUPS:-ctx_attn,prior_mlp,all8}"
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

MAX_ROWS="${MAX_ROWS:-20}"
ALPHAS="${ALPHAS:-0,0.5,1.0,1.5}"
APPLY_MODES="${APPLY_MODES:-all}"
CONTROLS="${CONTROLS:-force_target,force_start}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_GENERATION="${SKIP_GENERATION:-0}"

mkdir -p "$ROOT"
MASTER_LOG="$ROOT/master.log"
STATUS="$ROOT/status.tsv"

test -d "$MODEL"
test -f "$PAIRS"
test -f "$DATA"

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

echo "[$(date -Is)] root=$ROOT groups=$HARD_GROUPS train_rows=$TRAIN_ROWS val_rows=$VAL_ROWS eval_rows=$MAX_ROWS gen=$GEN_PROMPT alphas=$ALPHAS apply_modes=$APPLY_MODES" | tee -a "$MASTER_LOG"

IFS=',' read -ra GROUP_ARRAY <<< "$HARD_GROUPS"
ACTUATORS=()
for group in "${GROUP_ARRAY[@]}"; do
  group="$(echo "$group" | xargs)"
  [[ -n "$group" ]] || continue
  components_csv="configs/cecm_hard_${group}_components.csv"
  out="$ROOT/train_${group}"
  test -f "$components_csv"
  mkdir -p "$out"
  if [[ "$SKIP_TRAIN" == "1" ]]; then
    test -f "$out/fixed_actuator.pt"
    echo "[$(date -Is)] skip train group=$group using=$out/fixed_actuator.pt" | tee -a "$MASTER_LOG"
    echo -e "$(date -Is)\ttrain\t$group\tskipped" >> "$STATUS"
  else
    echo "[$(date -Is)] train group=$group out=$out" | tee -a "$MASTER_LOG"
    echo -e "$(date -Is)\ttrain\t$group\trunning" >> "$STATUS"

    CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.cecm_train_fixed_actuator \
      --model "$MODEL" \
      --pairs-csv "$PAIRS" \
      --components-csv "$components_csv" \
      --event "$EVENT" \
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
      --alpha-sweep "$ALPHAS" \
      --torch-dtype bfloat16 \
      --device cuda \
      --out-dir "$out" \
      > "$out/train.log" 2>&1

    echo "[$(date -Is)] done train group=$group" | tee -a "$MASTER_LOG"
    echo -e "$(date -Is)\ttrain\t$group\tdone" >> "$STATUS"
  fi
  ACTUATORS+=("${group}=${out}/fixed_actuator.pt")
done

ACTUATORS_ARG="$(IFS=';'; echo "${ACTUATORS[*]}")"
if [[ "$SKIP_GENERATION" == "1" ]]; then
  echo "[$(date -Is)] skip generation" | tee -a "$MASTER_LOG"
  exit 0
fi

IFS=',' read -ra MODES <<< "$APPLY_MODES"
for mode in "${MODES[@]}"; do
  mode="$(echo "$mode" | xargs)"
  [[ -n "$mode" ]] || continue
  out="$ROOT/gen_${mode}"
  mkdir -p "$out"
  echo "[$(date -Is)] generation apply_mode=$mode out=$out" | tee -a "$MASTER_LOG"
  echo -e "$(date -Is)\tgeneration\t$mode\trunning" >> "$STATUS"

  CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.cecm_run_fixed_actuator_generation \
    --model "$MODEL" \
    --eval-open-rows "$DATA" \
    --actuators "$ACTUATORS_ARG" \
    --generation-prompt-key "$GEN_PROMPT" \
    --prior-source "$PRIOR_SOURCE" \
    --controls "$CONTROLS" \
    --split val \
    --start 0 \
    --max-rows "$MAX_ROWS" \
    --alpha-sweep "$ALPHAS" \
    --generation-apply-mode "$mode" \
    --max-new-tokens 64 \
    --stop-strings "Q:" \
    --torch-dtype bfloat16 \
    --device cuda \
    --out-dir "$out" \
    > "$out/generation.log" 2>&1

  echo "[$(date -Is)] done generation apply_mode=$mode" | tee -a "$MASTER_LOG"
  echo -e "$(date -Is)\tgeneration\t$mode\tdone" >> "$STATUS"
done

echo "[$(date -Is)] all done" | tee -a "$MASTER_LOG"
find "$ROOT" -maxdepth 2 \( -name alpha_summary.csv -o -name generation_summary.csv \) -print | sort
