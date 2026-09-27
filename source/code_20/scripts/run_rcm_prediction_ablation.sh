#!/usr/bin/env bash
set -euo pipefail

MODEL=""
EVAL_JSONL="data_ckplug/confiqa_source_identity_taskdefined_val800/source_all_alpha_1p0/02_prompts/confiqa_open_rows.jsonl"
GENERATIONS_JSONL="data_ckplug/confiqa_source_identity_taskdefined_val800/source_all_alpha_1p0/04_reports/generations.jsonl"
GRAPH_JSON="data_ckplug/rcm_interface_smoke/clean_selected_graph.json"
OUT_DIR="data_ckplug/rcm_prediction_ablation_midpoint_400_400"
TARGET_METHOD="strong_rag"
PROMPT_KEY="strong_rag"
CALIB_START="0"
CALIB_ROWS="400"
TEST_START="400"
TEST_ROWS="400"
DEVICE="cuda"
TORCH_DTYPE="bfloat16"
SCORE_MODE="midpoint"
SOURCE_CONTRAST_KIND="strong_no_rag"
MIN_DIRECTIONAL_AUROC="0.58"
TOP_K_PER_TARGET="8"
WEIGHT_MODE="hybrid"
CONFLICT_GAMMA="1.0"
CROSS_CONFLICT_GAMMA="0.0"
USE_CHAT_TEMPLATE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --eval-jsonl) EVAL_JSONL="$2"; shift 2 ;;
    --generations-jsonl) GENERATIONS_JSONL="$2"; shift 2 ;;
    --graph-json) GRAPH_JSON="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --target-method) TARGET_METHOD="$2"; shift 2 ;;
    --prompt-key) PROMPT_KEY="$2"; shift 2 ;;
    --calib-start) CALIB_START="$2"; shift 2 ;;
    --calib-rows) CALIB_ROWS="$2"; shift 2 ;;
    --test-start) TEST_START="$2"; shift 2 ;;
    --test-rows) TEST_ROWS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --torch-dtype) TORCH_DTYPE="$2"; shift 2 ;;
    --score-mode) SCORE_MODE="$2"; shift 2 ;;
    --source-contrast-kind) SOURCE_CONTRAST_KIND="$2"; shift 2 ;;
    --min-directional-auroc) MIN_DIRECTIONAL_AUROC="$2"; shift 2 ;;
    --top-k-per-target) TOP_K_PER_TARGET="$2"; shift 2 ;;
    --weight-mode) WEIGHT_MODE="$2"; shift 2 ;;
    --conflict-gamma) CONFLICT_GAMMA="$2"; shift 2 ;;
    --cross-conflict-gamma) CROSS_CONFLICT_GAMMA="$2"; shift 2 ;;
    --use-chat-template) USE_CHAT_TEMPLATE=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "--model is required" >&2
  exit 2
fi

mkdir -p "$OUT_DIR/calibration" "$OUT_DIR/strict_test" "$OUT_DIR/calibrated_test"

chat_args=()
if [[ "$USE_CHAT_TEMPLATE" == "1" ]]; then
  chat_args+=(--use_chat_template)
fi

common_score_args=(
  -m screscomp.cli.rcm_score_events
  --eval_jsonl "$EVAL_JSONL"
  --generations_jsonl "$GENERATIONS_JSONL"
  --graph_json "$GRAPH_JSON"
  --model "$MODEL"
  --target_method "$TARGET_METHOD"
  --prompt_key "$PROMPT_KEY"
  --device "$DEVICE"
  --torch_dtype "$TORCH_DTYPE"
  --score_mode "$SCORE_MODE"
  --source_contrast_kind "$SOURCE_CONTRAST_KIND"
  --conflict_gamma "$CONFLICT_GAMMA"
  --cross_conflict_gamma "$CROSS_CONFLICT_GAMMA"
  "${chat_args[@]}"
)

python "${common_score_args[@]}" \
  --start "$CALIB_START" \
  --max_rows "$CALIB_ROWS" \
  --out_predictions_jsonl "$OUT_DIR/calibration/predictions.jsonl" \
  --out_summary_csv "$OUT_DIR/calibration/summary.csv"

python -m screscomp.cli.rcm_calibrate_readout \
  --predictions_jsonl "$OUT_DIR/calibration/predictions.jsonl" \
  --out_csv "$OUT_DIR/readout_calibration.csv" \
  --out_edge_stats_csv "$OUT_DIR/edge_stats.csv" \
  --edge_score_field raw_score \
  --min_directional_auroc "$MIN_DIRECTIONAL_AUROC" \
  --top_k_per_target "$TOP_K_PER_TARGET" \
  --weight_mode "$WEIGHT_MODE"

python "${common_score_args[@]}" \
  --start "$TEST_START" \
  --max_rows "$TEST_ROWS" \
  --edge_stats_csv "$OUT_DIR/edge_stats.csv" \
  --edge_normalization robust \
  --out_predictions_jsonl "$OUT_DIR/strict_test/predictions.jsonl" \
  --out_summary_csv "$OUT_DIR/strict_test/summary.csv"

python "${common_score_args[@]}" \
  --start "$TEST_START" \
  --max_rows "$TEST_ROWS" \
  --edge_stats_csv "$OUT_DIR/edge_stats.csv" \
  --edge_normalization robust \
  --readout_calibration_csv "$OUT_DIR/readout_calibration.csv" \
  --out_predictions_jsonl "$OUT_DIR/calibrated_test/predictions.jsonl" \
  --out_summary_csv "$OUT_DIR/calibrated_test/summary.csv"

python -m screscomp.cli.rcm_axis_audit \
  --strict_summary_csv "$OUT_DIR/strict_test/summary.csv" \
  --calibrated_summary_csv "$OUT_DIR/calibrated_test/summary.csv" \
  --graph_json "$GRAPH_JSON" \
  --out_csv "$OUT_DIR/axis_audit.csv"

echo "[run-rcm-prediction-ablation] calibration=$OUT_DIR/calibration/summary.csv"
echo "[run-rcm-prediction-ablation] edge_stats=$OUT_DIR/edge_stats.csv"
echo "[run-rcm-prediction-ablation] strict_test=$OUT_DIR/strict_test/summary.csv"
echo "[run-rcm-prediction-ablation] calibrated_test=$OUT_DIR/calibrated_test/summary.csv"
echo "[run-rcm-prediction-ablation] axis_audit=$OUT_DIR/axis_audit.csv"
