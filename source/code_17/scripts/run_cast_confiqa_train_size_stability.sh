#!/usr/bin/env bash
set -euo pipefail

# Train-size stability sweep for CAST on ConFiQA.
#
# For each calibration size N this reruns the full low-data method:
#   prepare first N source rows -> build pairs -> discover components -> train
#   MLP/head actuators -> auto-tune alpha on that N-row calibration val split ->
#   evaluate the selected CAST control on the same heldout slice.
#
# It is intentionally serial and avoids Context-DPO/base prompt reruns by default.

cd "$(dirname "${BASH_SOURCE[0]}")/.."

GPU="${GPU:-3}"
TASK="${TASK:-qa}"
MODEL="${MODEL:-LOCAL_HOME/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct}"
BASE_OUT="${BASE_OUT:-data_ckplug/cast_confiqa_train_size_stability_${TASK}_v0}"

SIZES="${SIZES:-60 120 180 300}"
VAL_MOD="${VAL_MOD:-5}"
EVAL_SOURCE_START="${EVAL_SOURCE_START:-1000}"
EVAL_SOURCE_ROWS="${EVAL_SOURCE_ROWS:-}"
EVAL_ROWS="${EVAL_ROWS:-500}"
EVAL_SPLIT="${EVAL_SPLIT:-all}"
EPOCHS="${EPOCHS:-2}"
LR="${LR:-0.05}"
LAMBDA_NORM="${LAMBDA_NORM:-1e-4}"
EMPTY_CACHE_EVERY="${EMPTY_CACHE_EVERY:-25}"

HEAD_ALPHAS="${HEAD_ALPHAS:-0.5 0.65 0.8}"
MLP_ALPHAS="${MLP_ALPHAS:-0 0.025 0.05 0.075}"
TUNE_FAMILIES="${TUNE_FAMILIES:-attn full}"

RUN_BASE_PROMPTS="${RUN_BASE_PROMPTS:-0}"
RUN_CONTEXT_DPO="${RUN_CONTEXT_DPO:-0}"
DISCOVERY_TOPK_MLP="${DISCOVERY_TOPK_MLP:-4}"
DISCOVERY_TOPK_ATTN_LAYERS="${DISCOVERY_TOPK_ATTN_LAYERS:-4}"
HEAD_TOPK="${HEAD_TOPK:-4}"
HEAD_SCAN_FACTORS="${HEAD_SCAN_FACTORS:-0.0,1.5}"
HEAD_REFINE_EVAL_ROWS="${HEAD_REFINE_EVAL_ROWS:-1}"

mkdir -p "$BASE_OUT"
MASTER_LOG="$BASE_OUT/train_size_stability.log"
SUMMARY="$BASE_OUT/train_size_summary.tsv"
echo -e "train_source_rows\ttrain_rows\tval_rows\tdiscovery_rows\thead_scan_rows\tselected\tpc\tpo\tmr\tem\tcontext_only\tmean_chars\tselection_gap\tneeds_more_validation\tout_dir" > "$SUMMARY"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  deactivate || true
fi
source LOCAL_HOME/anaconda3/etc/profile.d/conda.sh
conda activate screscomp
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

for n in $SIZES; do
  if [[ "$n" -le 0 ]]; then
    echo "Invalid train source size: $n" >&2
    exit 2
  fi
  val_rows=$(((n + VAL_MOD - 1) / VAL_MOD))
  train_rows=$((n - val_rows))
  discovery_rows="$train_rows"
  if [[ "$discovery_rows" -gt 60 ]]; then discovery_rows=60; fi
  head_scan_rows="$train_rows"
  if [[ "$head_scan_rows" -gt 24 ]]; then head_scan_rows=24; fi
  if [[ "$head_scan_rows" -lt 1 ]]; then head_scan_rows=1; fi

  root="$BASE_OUT/n${n}"
  mkdir -p "$root"
  echo "[$(date -Is)] train-size n=$n train_rows=$train_rows val_rows=$val_rows root=$root" | tee -a "$MASTER_LOG"

  GPU="$GPU" \
    TASK="$TASK" \
    MODEL="$MODEL" \
    ROOT="$root" \
    TRAIN_SOURCE_ROWS="$n" \
    EVAL_SOURCE_START="$EVAL_SOURCE_START" \
    EVAL_SOURCE_ROWS="$EVAL_SOURCE_ROWS" \
    VAL_MOD="$VAL_MOD" \
    TRAIN_ROWS="$train_rows" \
    VAL_ROWS="$val_rows" \
    MLP_TRAIN_ROWS="$train_rows" \
    MLP_VAL_ROWS="$val_rows" \
    HEAD_TRAIN_ROWS="$train_rows" \
    HEAD_VAL_ROWS="$val_rows" \
    EPOCHS="$EPOCHS" \
    MLP_EPOCHS="$EPOCHS" \
    HEAD_EPOCHS="$EPOCHS" \
    LR="$LR" \
    LAMBDA_NORM="$LAMBDA_NORM" \
    EVAL_SPLIT="$EVAL_SPLIT" \
    EVAL_ROWS="$EVAL_ROWS" \
    RUN_BASE_PROMPTS="$RUN_BASE_PROMPTS" \
    RUN_CONTEXT_DPO="$RUN_CONTEXT_DPO" \
    DISCOVER_COMPONENTS=1 \
    REUSE_DISCOVERY=0 \
    DISCOVERY_ROWS="$discovery_rows" \
    DISCOVERY_TOPK_MLP="$DISCOVERY_TOPK_MLP" \
    DISCOVERY_TOPK_ATTN_LAYERS="$DISCOVERY_TOPK_ATTN_LAYERS" \
    HEAD_SCAN_ROWS="$head_scan_rows" \
    HEAD_SCAN_FACTORS="$HEAD_SCAN_FACTORS" \
    HEAD_TOPK="$HEAD_TOPK" \
    HEAD_REFINE_EVAL_ROWS="$HEAD_REFINE_EVAL_ROWS" \
    AUTO_TUNE=1 \
    HEAD_ALPHAS="$HEAD_ALPHAS" \
    MLP_ALPHAS="$MLP_ALPHAS" \
    TUNE_FAMILIES="$TUNE_FAMILIES" \
    EMPTY_CACHE_EVERY="$EMPTY_CACHE_EVERY" \
    CAST_CONTROLS="" \
    bash scripts/run_cast_confiqa_min_loop.sh \
    > "$root/nohup.log" 2>&1

  ROOT_DIR="$root" TRAIN_SOURCE="$n" TRAIN_ROWS_VALUE="$train_rows" VAL_ROWS_VALUE="$val_rows" DISCOVERY_ROWS_VALUE="$discovery_rows" HEAD_SCAN_ROWS_VALUE="$head_scan_rows" SUMMARY="$SUMMARY" python - <<'PY'
import csv
import json
import os
from pathlib import Path

root = Path(os.environ["ROOT_DIR"])
summary = Path(os.environ["SUMMARY"])
gen_summary = root / "eval_cast_base_rag" / "generation_summary.csv"
best_json = root / "auto_tune" / "selection" / "best_config.json"

rows = list(csv.DictReader(gen_summary.open("r", encoding="utf-8", newline="")))
if not rows:
    raise SystemExit(f"No generation summary rows: {gen_summary}")
row = rows[0]
if best_json.exists():
    payload = json.loads(best_json.read_text(encoding="utf-8"))
    selected = payload.get("best_control_name", row.get("control_name", ""))
    gap = payload.get("selection_gap", "")
    needs_more = int(bool(payload.get("needs_more_validation", False)))
else:
    selected = row.get("control_name", "")
    gap = ""
    needs_more = ""
values = [
    os.environ["TRAIN_SOURCE"],
    os.environ["TRAIN_ROWS_VALUE"],
    os.environ["VAL_ROWS_VALUE"],
    os.environ["DISCOVERY_ROWS_VALUE"],
    os.environ["HEAD_SCAN_ROWS_VALUE"],
    selected,
    row.get("pc", ""),
    row.get("po", ""),
    row.get("mr", ""),
    row.get("em", ""),
    row.get("context_only_rate", ""),
    row.get("mean_chars", ""),
    gap,
    needs_more,
    str(root),
]
with summary.open("a", encoding="utf-8") as fp:
    fp.write("\t".join(str(v) for v in values) + "\n")
PY
done

cat "$SUMMARY"
