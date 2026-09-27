#!/usr/bin/env bash
set -euo pipefail

# Smoke-test three high-minus-low prompt-delta actuator sources.
# Run this script itself under nohup if you want the terminal back.

GPU="${GPU:-3}"
MODEL="${MODEL:-LOCAL_HOME/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct}"
DATA="${DATA:-data_ckplug/confiqa_large_ck_default_test5000/02_prompts/confiqa_open_rows.jsonl}"
SRC="${SRC:-data_ckplug/cecm_source_modelprior_main1000_v1}"
ROOT="${ROOT:-data_ckplug/cecm_source_prompt_delta_vector3_smoke20_v2}"

VECTOR_ROWS="${VECTOR_ROWS:-70}"
MAX_ROWS="${MAX_ROWS:-20}"
ALPHAS="${ALPHAS:-0,0.5,1.0}"
CONTROLS="${CONTROLS:-force_context}"
RANDOM_BASELINES="${RANDOM_BASELINES:-}"
RANDOM_TRIALS="${RANDOM_TRIALS:-0}"
COMPONENTS="${COMPONENTS:-l9=L9.attn;l31=L31.attn;l9_l31=L9.attn,L31.attn;all4=L9.attn,L31.attn,L27.attn,L25.attn}"

mkdir -p "$ROOT"
MASTER_LOG="$ROOT/master.log"
STATUS="$ROOT/status.tsv"

test -d "$MODEL"
test -f "$DATA"
test -f "$SRC/01_pairs/pairs.csv"

{
  echo -e "time\tname\thigh\tlow\tstatus"
  echo -e "$(date -Is)\tinit\t-\t-\tstarting"
} > "$STATUS"

run_vec() {
  local name="$1"
  local high="$2"
  local low="$3"
  local out="$ROOT/$name"
  mkdir -p "$out"
  rm -f "$out/run.log"

  echo "[$(date -Is)] start name=$name high=$high low=$low out=$out" | tee -a "$MASTER_LOG"
  echo -e "$(date -Is)\t$name\t$high\t$low\trunning" >> "$STATUS"

  CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.cecm_run_source_prompt_delta_control \
    --model "$MODEL" \
    --vector-open-rows "$DATA" \
    --eval-open-rows "$DATA" \
    --vector-pairs-csv "$SRC/01_pairs/pairs.csv" \
    --context-prompt-key "$high" \
    --prior-prompt-key "$low" \
    --prior-source dataset_orig \
    --components "$COMPONENTS" \
    --controls "$CONTROLS" \
    --random-baselines "$RANDOM_BASELINES" \
    --random-trials "$RANDOM_TRIALS" \
    --vector-split train \
    --vector-start 0 \
    --vector-max-rows "$VECTOR_ROWS" \
    --split val \
    --start 0 \
    --max-rows "$MAX_ROWS" \
    --alpha-sweep "$ALPHAS" \
    --generation-apply-mode prefill \
    --max-new-tokens 64 \
    --stop-strings "Q:" \
    --torch-dtype bfloat16 \
    --device cuda \
    --out-dir "$out" \
    > "$out/run.log" 2>&1

  echo "[$(date -Is)] done name=$name" | tee -a "$MASTER_LOG"
  echo -e "$(date -Is)\t$name\t$high\t$low\tdone" >> "$STATUS"
}

echo "[$(date -Is)] root=$ROOT max_rows=$MAX_ROWS vector_rows=$VECTOR_ROWS alphas=$ALPHAS controls=$CONTROLS random=$RANDOM_BASELINES" | tee -a "$MASTER_LOG"

run_vec strong_minus_base strong_rag base_rag
run_vec strong_minus_prior strong_rag prior_objective_rag
run_vec base_minus_prior base_rag prior_objective_rag

echo "[$(date -Is)] all done" | tee -a "$MASTER_LOG"
echo "summary files:"
find "$ROOT" -maxdepth 2 -name generation_summary.csv -print | sort
