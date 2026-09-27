#!/usr/bin/env bash
set -euo pipefail

MODEL=""
DATA_JSON="external/CK-PLUG-data/ConFiQA-QA.json"
AXIS="source_identity"
DIRN="forward"
SOURCE_CONTRAST_KIND="strong_no_rag"
GENERATION_PROMPT_KEY="strong_rag"
OUT_DIR=""
START=0
MAX_ROWS=200
ALPHA=1.0
APPLY_MODE="prefill"
DEVICE="cuda"
TORCH_DTYPE="bfloat16"
MAX_NEW_TOKENS=24

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --data-json) DATA_JSON="$2"; shift 2 ;;
    --axis) AXIS="$2"; shift 2 ;;
    --direction) DIRN="$2"; shift 2 ;;
    --source-contrast-kind) SOURCE_CONTRAST_KIND="$2"; shift 2 ;;
    --generation-prompt-key) GENERATION_PROMPT_KEY="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --start) START="$2"; shift 2 ;;
    --max-rows) MAX_ROWS="$2"; shift 2 ;;
    --alpha) ALPHA="$2"; shift 2 ;;
    --apply-mode) APPLY_MODE="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --torch-dtype) TORCH_DTYPE="$2"; shift 2 ;;
    --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "--model is required" >&2
  exit 2
fi
if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="data_ckplug/confiqa_${AXIS}_${DIRN}_scan_start${START}_rows${MAX_ROWS}"
fi

mkdir -p "$OUT_DIR/02_prompts" "$OUT_DIR/04_reports"

echo "[ckplug-factor-scan] prepare open rows axis=$AXIS"
python -m screscomp.cli.prepare_ckplug_open \
  --dataset confiqa \
  --data_json "$DATA_JSON" \
  --alias_policy raw \
  --start "$START" \
  --max_rows "$MAX_ROWS" \
  --out_jsonl "$OUT_DIR/02_prompts/confiqa_open_rows.jsonl"

echo "[ckplug-factor-scan] scan all attn/mlp components axis=$AXIS"
python -m screscomp.cli.score_ckplug_factor_component_scan \
  --eval_jsonl "$OUT_DIR/02_prompts/confiqa_open_rows.jsonl" \
  --axis "$AXIS" \
  --direction "$DIRN" \
  --source_contrast_kind "$SOURCE_CONTRAST_KIND" \
  --generation_prompt_key "$GENERATION_PROMPT_KEY" \
  --model "$MODEL" \
  --out_rows_jsonl "$OUT_DIR/04_reports/${AXIS}_component_rows.jsonl" \
  --out_summary_csv "$OUT_DIR/04_reports/${AXIS}_component_summary.csv" \
  --device "$DEVICE" \
  --torch_dtype "$TORCH_DTYPE" \
  --use_chat_template \
  --max_rows "$MAX_ROWS" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --alpha "$ALPHA" \
  --apply_mode "$APPLY_MODE"

echo "[ckplug-factor-scan] top components axis=$AXIS"
python - <<'PY' "$OUT_DIR/04_reports/${AXIS}_component_summary.csv"
import csv, sys
path = sys.argv[1]
rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
print(",".join(row["component_id"] for row in rows[:4]))
for row in rows[:12]:
    print(
        row["selection_rank"],
        row["component_id"],
        row["selection_score"],
        row.get("selection_metric", ""),
        row.get("mean_context_only", ""),
        row.get("mean_neither", ""),
        row.get("mean_output_chars", ""),
    )
PY

echo "[ckplug-factor-scan] done"
echo "Summary: $OUT_DIR/04_reports/${AXIS}_component_summary.csv"
