#!/usr/bin/env bash
set -euo pipefail

# Refine already-selected attention components into head-level scaling controls.
# Default layers are the current CECM attention components: L31/L27/L9/L25.

cd "$(dirname "${BASH_SOURCE[0]}")/.."

GPU="${GPU:-2}"
MODEL="${MODEL:-LOCAL_HOME/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct}"
PAIRS="${PAIRS:-data_ckplug/cecm_source_context_over_prior_base_step1/pairs.csv}"
DATA="${DATA:-data_ckplug/confiqa_large_ck_default_test5000/02_prompts/confiqa_open_rows.jsonl}"
ROOT="${ROOT:-data_ckplug/cecm_attn_head_refine_smoke_v0}"

ATTN_LAYERS="${ATTN_LAYERS:-31,27,9,25}"
EVENT="${EVENT:-source_context_over_prior}"
GEN_PROMPT="${GEN_PROMPT:-base_rag}"
PRIOR_SOURCE="${PRIOR_SOURCE:-dataset_orig}"

SCAN_ROWS="${SCAN_ROWS:-24}"
MAX_ROWS="${MAX_ROWS:-20}"
SCAN_FACTORS="${SCAN_FACTORS:-0.5,0.0,1.5}"
TOPKS="${TOPKS:-2,4}"
GENERATION_KINDS="${GENERATION_KINDS:-suppress,boost,mixed}"
APPLY_MODES="${APPLY_MODES:-prefill,first_decode}"
SCORE_APPLY_MODE="${SCORE_APPLY_MODE:-decision_tokens}"
SCORE_MODE="${SCORE_MODE:-answer_rest_margin}"
MAX_ALIASES="${MAX_ALIASES:-1}"

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
  echo -e "time\tstage\tstatus"
  echo -e "$(date -Is)\tinit\tstarting"
} > "$STATUS"

echo "[$(date -Is)] root=$ROOT attn_layers=$ATTN_LAYERS scan_rows=$SCAN_ROWS eval_rows=$MAX_ROWS scan_factors=$SCAN_FACTORS topks=$TOPKS apply_modes=$APPLY_MODES" | tee -a "$MASTER_LOG"
echo -e "$(date -Is)\trun\trunning" >> "$STATUS"

CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.cecm_run_attention_head_refine \
  --model "$MODEL" \
  --pairs-csv "$PAIRS" \
  --event "$EVENT" \
  --eval-open-rows "$DATA" \
  --attn-layers "$ATTN_LAYERS" \
  --scan-factors "$SCAN_FACTORS" \
  --topks "$TOPKS" \
  --generation-kinds "$GENERATION_KINDS" \
  --generation-apply-modes "$APPLY_MODES" \
  --generation-prompt-key "$GEN_PROMPT" \
  --prior-source "$PRIOR_SOURCE" \
  --split val \
  --scan-start 0 \
  --scan-max-rows "$SCAN_ROWS" \
  --start 0 \
  --max-rows "$MAX_ROWS" \
  --score-mode "$SCORE_MODE" \
  --score-apply-mode "$SCORE_APPLY_MODE" \
  --max-aliases-per-side "$MAX_ALIASES" \
  --max-new-tokens 64 \
  --stop-strings "Q:" \
  --torch-dtype bfloat16 \
  --device cuda \
  --out-dir "$ROOT" \
  > "$ROOT/run.log" 2>&1

echo "[$(date -Is)] done" | tee -a "$MASTER_LOG"
echo -e "$(date -Is)\trun\tdone" >> "$STATUS"
find "$ROOT" -maxdepth 1 \( -name head_scan.csv -o -name head_group_plan.csv -o -name generation_summary.csv \) -print | sort
