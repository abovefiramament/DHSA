#!/usr/bin/env bash
set -euo pipefail

MODEL=""
DATA_JSON="external/CK-PLUG-data/ConFiQA-QA.json"
OUT_DIR="data_ckplug/confiqa_from_zero_bidirectional_axis_acceptance_select1000_test1000"
START=1000
MAX_ROWS=1000
SCAN_START=0
SCAN_ROWS=1000
MAX_TRAIN_ROWS=96
TOP_K=8
ALPHA=1.0
SOURCE_APPLY_MODE="all"
FORMAT_ALPHA=1.0
FORMAT_APPLY_MODE="prefill"
COMMITMENT_ALPHA=1.0
COMMITMENT_APPLY_MODE="first_2_decode"
PRIOR_SUPPRESSION_ALPHA=1.0
PRIOR_SUPPRESSION_APPLY_MODE="first_2_decode"
DEVICE="cuda"
TORCH_DTYPE="bfloat16"
BOOTSTRAP_SAMPLES=500
SCORE_MODE="dot"
SKIP_SCANS=0
SKIP_PREDICTION=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --data-json) DATA_JSON="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --start) START="$2"; shift 2 ;;
    --max-rows) MAX_ROWS="$2"; shift 2 ;;
    --scan-start) SCAN_START="$2"; shift 2 ;;
    --scan-rows) SCAN_ROWS="$2"; shift 2 ;;
    --max-train-rows) MAX_TRAIN_ROWS="$2"; shift 2 ;;
    --top-k) TOP_K="$2"; shift 2 ;;
    --alpha) ALPHA="$2"; shift 2 ;;
    --source-apply-mode) SOURCE_APPLY_MODE="$2"; shift 2 ;;
    --format-alpha) FORMAT_ALPHA="$2"; shift 2 ;;
    --format-apply-mode) FORMAT_APPLY_MODE="$2"; shift 2 ;;
    --commitment-alpha) COMMITMENT_ALPHA="$2"; shift 2 ;;
    --commitment-apply-mode) COMMITMENT_APPLY_MODE="$2"; shift 2 ;;
    --prior-suppression-alpha) PRIOR_SUPPRESSION_ALPHA="$2"; shift 2 ;;
    --prior-suppression-apply-mode) PRIOR_SUPPRESSION_APPLY_MODE="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --torch-dtype) TORCH_DTYPE="$2"; shift 2 ;;
    --bootstrap-samples) BOOTSTRAP_SAMPLES="$2"; shift 2 ;;
    --score-mode) SCORE_MODE="$2"; shift 2 ;;
    --skip-scans) SKIP_SCANS=1; shift ;;
    --skip-prediction) SKIP_PREDICTION=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "--model is required" >&2
  exit 2
fi

DISCOVERY_DIR="$OUT_DIR/00_discovery"
REPORT_DIR="$OUT_DIR/04_reports"
mkdir -p "$DISCOVERY_DIR" "$REPORT_DIR"

scan_axis() {
  local axis="$1"
  local direction="$2"
  local out_dir="$3"
  local apply_mode="$4"
  local summary="$out_dir/04_reports/${axis}_component_summary.csv"
  if [[ "$SKIP_SCANS" == "1" && -s "$summary" ]]; then
    echo "[from-zero-bidir] reuse existing scan axis=$axis direction=$direction summary=$summary"
    return
  fi
  echo "[from-zero-bidir] scan axis=$axis direction=$direction"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" bash scripts/run_ckplug_factor_component_scan.sh \
    --model "$MODEL" \
    --data-json "$DATA_JSON" \
    --axis "$axis" \
    --direction "$direction" \
    --source-contrast-kind objective_prior \
    --generation-prompt-key strong_rag \
    --out-dir "$out_dir" \
    --start "$SCAN_START" \
    --max-rows "$SCAN_ROWS" \
    --alpha 1.0 \
    --apply-mode "$apply_mode" \
    --device "$DEVICE" \
    --torch-dtype "$TORCH_DTYPE" \
    --max-new-tokens 24
}

SOURCE_CONTEXT_DIR="$DISCOVERY_DIR/source_context"
SOURCE_PRIOR_DIR="$DISCOVERY_DIR/source_prior"
FORM_SHORT_DIR="$DISCOVERY_DIR/form_short"
FORM_VERBOSE_DIR="$DISCOVERY_DIR/form_verbose"
COMMITMENT_SINGLE_DIR="$DISCOVERY_DIR/commitment_single"
COMMITMENT_MIXED_DIR="$DISCOVERY_DIR/commitment_mixed"
PRIOR_NO_PRIOR_DIR="$DISCOVERY_DIR/prior_no_prior"
PRIOR_MEMORY_DIR="$DISCOVERY_DIR/prior_memory"

scan_axis "source_identity" "forward" "$SOURCE_CONTEXT_DIR" "$SOURCE_APPLY_MODE"
scan_axis "source_identity" "reverse" "$SOURCE_PRIOR_DIR" "$SOURCE_APPLY_MODE"
scan_axis "form" "forward" "$FORM_SHORT_DIR" "$FORMAT_APPLY_MODE"
scan_axis "form" "reverse" "$FORM_VERBOSE_DIR" "$FORMAT_APPLY_MODE"
scan_axis "commitment" "forward" "$COMMITMENT_SINGLE_DIR" "$COMMITMENT_APPLY_MODE"
scan_axis "commitment" "reverse" "$COMMITMENT_MIXED_DIR" "$COMMITMENT_APPLY_MODE"
scan_axis "prior_suppression" "forward" "$PRIOR_NO_PRIOR_DIR" "$PRIOR_SUPPRESSION_APPLY_MODE"
scan_axis "prior_suppression" "reverse" "$PRIOR_MEMORY_DIR" "$PRIOR_SUPPRESSION_APPLY_MODE"

SOURCE_CONTEXT_SUMMARY="$SOURCE_CONTEXT_DIR/04_reports/source_identity_component_summary.csv"
SOURCE_PRIOR_SUMMARY="$SOURCE_PRIOR_DIR/04_reports/source_identity_component_summary.csv"
FORM_SHORT_SUMMARY="$FORM_SHORT_DIR/04_reports/form_component_summary.csv"
FORM_VERBOSE_SUMMARY="$FORM_VERBOSE_DIR/04_reports/form_component_summary.csv"
COMMITMENT_SINGLE_SUMMARY="$COMMITMENT_SINGLE_DIR/04_reports/commitment_component_summary.csv"
COMMITMENT_MIXED_SUMMARY="$COMMITMENT_MIXED_DIR/04_reports/commitment_component_summary.csv"
PRIOR_NO_PRIOR_SUMMARY="$PRIOR_NO_PRIOR_DIR/04_reports/prior_suppression_component_summary.csv"
PRIOR_MEMORY_SUMMARY="$PRIOR_MEMORY_DIR/04_reports/prior_suppression_component_summary.csv"

METHODS="strong_rag"
METHODS+=",random_delta_prefill_context,ours_delta_prefill_context,random_delta_prefill_prior,ours_delta_prefill_prior"
METHODS+=",random_format_prefill_short,ours_format_prefill_short,random_format_prefill_verbose,ours_format_prefill_verbose"
METHODS+=",random_commitment_prefill_single,ours_commitment_prefill_single,random_commitment_prefill_mixed,ours_commitment_prefill_mixed"
METHODS+=",random_prior_suppression_prefill_no_prior,ours_prior_suppression_prefill_no_prior,random_prior_suppression_prefill_memory,ours_prior_suppression_prefill_memory"

COMPARE_PAIRS="random_delta_prefill_context>ours_delta_prefill_context,random_delta_prefill_prior>ours_delta_prefill_prior"
COMPARE_PAIRS+=",random_format_prefill_short>ours_format_prefill_short,random_format_prefill_verbose>ours_format_prefill_verbose"
COMPARE_PAIRS+=",random_commitment_prefill_single>ours_commitment_prefill_single,random_commitment_prefill_mixed>ours_commitment_prefill_mixed"
COMPARE_PAIRS+=",random_prior_suppression_prefill_no_prior>ours_prior_suppression_prefill_no_prior,random_prior_suppression_prefill_memory>ours_prior_suppression_prefill_memory"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" bash scripts/run_ckplug_generation_pilot.sh \
  --model "$MODEL" \
  --data-json "$DATA_JSON" \
  --out-dir "$OUT_DIR" \
  --start "$START" \
  --max-rows "$MAX_ROWS" \
  --max-train-rows "$MAX_TRAIN_ROWS" \
  --top-k "$TOP_K" \
  --component-summary-csv "$SOURCE_CONTEXT_SUMMARY" \
  --source-prior-component-summary-csv "$SOURCE_PRIOR_SUMMARY" \
  --format-component-summary-csv "$FORM_SHORT_SUMMARY" \
  --format-verbose-component-summary-csv "$FORM_VERBOSE_SUMMARY" \
  --format-top-k "$TOP_K" \
  --commitment-component-summary-csv "$COMMITMENT_SINGLE_SUMMARY" \
  --commitment-mixed-component-summary-csv "$COMMITMENT_MIXED_SUMMARY" \
  --commitment-top-k "$TOP_K" \
  --prior-suppression-component-summary-csv "$PRIOR_NO_PRIOR_SUMMARY" \
  --prior-memory-component-summary-csv "$PRIOR_MEMORY_SUMMARY" \
  --prior-suppression-top-k "$TOP_K" \
  --alpha "$ALPHA" \
  --source-delta-kind objective_context_vs_prior \
  --source-apply-mode "$SOURCE_APPLY_MODE" \
  --format-alpha "$FORMAT_ALPHA" \
  --format-apply-mode "$FORMAT_APPLY_MODE" \
  --commitment-alpha "$COMMITMENT_ALPHA" \
  --commitment-apply-mode "$COMMITMENT_APPLY_MODE" \
  --prior-suppression-alpha "$PRIOR_SUPPRESSION_ALPHA" \
  --prior-suppression-apply-mode "$PRIOR_SUPPRESSION_APPLY_MODE" \
  --use-chat-template \
  --device "$DEVICE" \
  --torch-dtype "$TORCH_DTYPE" \
  --methods "$METHODS" \
  --prompt-key strong_rag \
  --max-new-tokens 24 \
  --stop-strings Q: \
  --alias-policy raw \
  --compare-pairs "$COMPARE_PAIRS" \
  --bootstrap-samples "$BOOTSTRAP_SAMPLES"

SOURCE_BIDIR_SUMMARY="$REPORT_DIR/source_identity_bidirectional_summary.csv"
FORM_BIDIR_SUMMARY="$REPORT_DIR/form_bidirectional_summary.csv"
COMMITMENT_BIDIR_SUMMARY="$REPORT_DIR/commitment_bidirectional_summary.csv"
PRIOR_BIDIR_SUMMARY="$REPORT_DIR/prior_suppression_bidirectional_summary.csv"

python -m screscomp.cli.rcm_merge_bidirectional_axis_summaries \
  --axis source_identity \
  --forward_csv "$SOURCE_CONTEXT_SUMMARY" \
  --reverse_csv "$SOURCE_PRIOR_SUMMARY" \
  --top_forward_k "$TOP_K" \
  --top_reverse_k "$TOP_K" \
  --out_csv "$SOURCE_BIDIR_SUMMARY"
python -m screscomp.cli.rcm_merge_bidirectional_axis_summaries \
  --axis form \
  --forward_csv "$FORM_SHORT_SUMMARY" \
  --reverse_csv "$FORM_VERBOSE_SUMMARY" \
  --top_forward_k "$TOP_K" \
  --top_reverse_k "$TOP_K" \
  --out_csv "$FORM_BIDIR_SUMMARY"
python -m screscomp.cli.rcm_merge_bidirectional_axis_summaries \
  --axis commitment \
  --forward_csv "$COMMITMENT_SINGLE_SUMMARY" \
  --reverse_csv "$COMMITMENT_MIXED_SUMMARY" \
  --top_forward_k "$TOP_K" \
  --top_reverse_k "$TOP_K" \
  --out_csv "$COMMITMENT_BIDIR_SUMMARY"
python -m screscomp.cli.rcm_merge_bidirectional_axis_summaries \
  --axis prior_suppression \
  --forward_csv "$PRIOR_NO_PRIOR_SUMMARY" \
  --reverse_csv "$PRIOR_MEMORY_SUMMARY" \
  --top_forward_k "$TOP_K" \
  --top_reverse_k "$TOP_K" \
  --out_csv "$PRIOR_BIDIR_SUMMARY"

GRAPH_JSON="$REPORT_DIR/rcm_from_zero_bidirectional_graph.json"
python -m screscomp.cli.rcm_build_graph \
  --name rcm_from_zero_bidirectional \
  --source_summary_csv "$SOURCE_BIDIR_SUMMARY" \
  --form_summary_csv "$FORM_BIDIR_SUMMARY" \
  --commitment_summary_csv "$COMMITMENT_BIDIR_SUMMARY" \
  --prior_suppression_summary_csv "$PRIOR_BIDIR_SUMMARY" \
  --top_up_k "$TOP_K" \
  --top_down_k "$TOP_K" \
  --axis_timings "source_identity=$SOURCE_APPLY_MODE,form=$FORMAT_APPLY_MODE,commitment=$COMMITMENT_APPLY_MODE,prior_suppression=$PRIOR_SUPPRESSION_APPLY_MODE" \
  --axis_alphas "source_identity=$ALPHA,form=$FORMAT_ALPHA,commitment=$COMMITMENT_ALPHA,prior_suppression=$PRIOR_SUPPRESSION_ALPHA" \
  --out_graph_json "$GRAPH_JSON" \
  --out_components_csv "$REPORT_DIR/rcm_from_zero_bidirectional_components.csv" \
  --out_component_relations_csv "$REPORT_DIR/rcm_from_zero_bidirectional_component_relations.csv" \
  --out_axis_relations_csv "$REPORT_DIR/rcm_from_zero_bidirectional_axis_relations.csv"

if [[ "$SKIP_PREDICTION" != "1" ]]; then
  mkdir -p "$OUT_DIR/05_prediction/strict_test"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python -m screscomp.cli.rcm_score_events \
    --model "$MODEL" \
    --eval_jsonl "$OUT_DIR/02_prompts/confiqa_open_rows.jsonl" \
    --generations_jsonl "$OUT_DIR/04_reports/generations.jsonl" \
    --graph_json "$GRAPH_JSON" \
    --out_predictions_jsonl "$OUT_DIR/05_prediction/strict_test/predictions.jsonl" \
    --out_summary_csv "$OUT_DIR/05_prediction/strict_test/summary.csv" \
    --target_method strong_rag \
    --prompt_key strong_rag \
    --device "$DEVICE" \
    --torch_dtype "$TORCH_DTYPE" \
    --score_mode "$SCORE_MODE" \
    --source_contrast_kind objective_prior \
    --use_chat_template
  python -m screscomp.cli.rcm_axis_audit \
    --strict_summary_csv "$OUT_DIR/05_prediction/strict_test/summary.csv" \
    --graph_json "$GRAPH_JSON" \
    --out_csv "$OUT_DIR/05_prediction/axis_audit.csv"
fi

echo "[from-zero-bidir] generation_summary=$OUT_DIR/04_reports/generation_summary.csv"
echo "[from-zero-bidir] generation_comparison=$OUT_DIR/04_reports/generation_comparison.csv"
echo "[from-zero-bidir] graph=$GRAPH_JSON"
if [[ "$SKIP_PREDICTION" != "1" ]]; then
  echo "[from-zero-bidir] prediction_axis_audit=$OUT_DIR/05_prediction/axis_audit.csv"
fi
