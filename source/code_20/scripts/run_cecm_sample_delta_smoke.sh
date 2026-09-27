#!/usr/bin/env bash
set -euo pipefail

# Old-RCM-style same-sample delta smoke.
# This is intentionally small: if it moves metrics, scale rows/alphas afterward.

cd "$(dirname "${BASH_SOURCE[0]}")/.."

GPU="${GPU:-2}"
MODEL="${MODEL:-LOCAL_HOME/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct}"
DATA="${DATA:-data_ckplug/confiqa_large_ck_default_test5000/02_prompts/confiqa_open_rows.jsonl}"
ROOT="${ROOT:-data_ckplug/cecm_sample_delta_smoke30_v0}"

START_PROMPT="${START_PROMPT:-prior_objective_rag}"
TARGET_PROMPT="${TARGET_PROMPT:-strong_rag}"
GEN_PROMPT="${GEN_PROMPT:-base_rag}"
PRIOR_SOURCE="${PRIOR_SOURCE:-dataset_orig}"

MAX_ROWS="${MAX_ROWS:-30}"
ALPHAS="${ALPHAS:-0,0.5,1.0,1.5}"
APPLY_MODES="${APPLY_MODES:-prefill,first_decode,all}"
CONTROLS="${CONTROLS:-force_target,force_start}"
COMPONENTS="${COMPONENTS:-ctx_attn=L31.attn,L27.attn,L9.attn,L25.attn;prior_mlp=L6.mlp,L20.mlp,L1.mlp,L22.mlp;all8=L31.attn,L27.attn,L9.attn,L25.attn,L6.mlp,L20.mlp,L1.mlp,L22.mlp}"
RANDOM_BASELINES="${RANDOM_BASELINES:-}"
RANDOM_TRIALS="${RANDOM_TRIALS:-0}"

mkdir -p "$ROOT"
MASTER_LOG="$ROOT/master.log"
STATUS="$ROOT/status.tsv"

test -d "$MODEL"
test -f "$DATA"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  deactivate || true
fi
source LOCAL_HOME/anaconda3/etc/profile.d/conda.sh
conda activate screscomp
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

{
  echo -e "time\tapply_mode\tstatus"
  echo -e "$(date -Is)\tinit\tstarting"
} > "$STATUS"

echo "[$(date -Is)] root=$ROOT start=$START_PROMPT target=$TARGET_PROMPT gen=$GEN_PROMPT max_rows=$MAX_ROWS alphas=$ALPHAS components=$COMPONENTS" | tee -a "$MASTER_LOG"

IFS=',' read -ra MODES <<< "$APPLY_MODES"
for mode in "${MODES[@]}"; do
  mode="$(echo "$mode" | xargs)"
  [[ -n "$mode" ]] || continue
  out="$ROOT/$mode"
  mkdir -p "$out"
  rm -f "$out/run.log"
  echo "[$(date -Is)] start apply_mode=$mode out=$out" | tee -a "$MASTER_LOG"
  echo -e "$(date -Is)\t$mode\trunning" >> "$STATUS"

  CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.cecm_run_sample_delta_smoke \
    --model "$MODEL" \
    --eval-open-rows "$DATA" \
    --start-prompt-key "$START_PROMPT" \
    --target-prompt-key "$TARGET_PROMPT" \
    --generation-prompt-key "$GEN_PROMPT" \
    --prior-source "$PRIOR_SOURCE" \
    --components "$COMPONENTS" \
    --controls "$CONTROLS" \
    --random-baselines "$RANDOM_BASELINES" \
    --random-trials "$RANDOM_TRIALS" \
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
    > "$out/run.log" 2>&1

  echo "[$(date -Is)] done apply_mode=$mode" | tee -a "$MASTER_LOG"
  echo -e "$(date -Is)\t$mode\tdone" >> "$STATUS"
done

echo "[$(date -Is)] all done" | tee -a "$MASTER_LOG"
find "$ROOT" -maxdepth 2 -name generation_summary.csv -print | sort
