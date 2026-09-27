#!/usr/bin/env bash
set -euo pipefail

# Clean Step-DPO/GSM8K-style CAST base loop:
#   1) adapt Step-DPO rows into generic y_plus/y_minus actuator pairs
#   2) discover residual-write components with the same competitive margin
#   3) select a small logic-support component pool
#   4) train component and optional attention-head actuators with gain-only Step-DPO-like loss
#
# This path intentionally does not reuse the older data_gsm8k loop.

cd "$(dirname "${BASH_SOURCE[0]}")/.."

GPU="${GPU:-0}"
MODEL="${MODEL:-LOCAL_HOME/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct}"
ROOT="${ROOT:-data_gsm8k/cast_step_dpo_base_llama3_v0}"

SOURCE_INPUT="${SOURCE_INPUT:-}"
HF_DATASET="${HF_DATASET:-xinlai/Math-Step-DPO-10K}"
HF_CONFIG="${HF_CONFIG:-}"
HF_SPLIT="${HF_SPLIT:-train}"
EVENT="${EVENT:-step_correct_over_error}"
PAIR_MODE="${PAIR_MODE:-step}"
ADMISSION_POLICY="${ADMISSION_POLICY:-basic}"
INCLUDE_INITIAL_IN_FULL_PROMPT="${INCLUDE_INITIAL_IN_FULL_PROMPT:-0}"
SOURCE_FILTER_FIELD="${SOURCE_FILTER_FIELD:-}"
SOURCE_FILTER_CONTAINS="${SOURCE_FILTER_CONTAINS:-}"

TRAIN_ROWS="${TRAIN_ROWS:-240}"
VAL_ROWS="${VAL_ROWS:-60}"
EVAL_ROWS="${EVAL_ROWS:-0}"
SPLIT_UNIT="${SPLIT_UNIT:-row}"
SHUFFLE_PAIRS="${SHUFFLE_PAIRS:-0}"
SEED="${SEED:-42}"

DISCOVERY_ROWS="${DISCOVERY_ROWS:-60}"
DISCOVERY_TOPK_MLP="${DISCOVERY_TOPK_MLP:-4}"
DISCOVERY_TOPK_ATTN_LAYERS="${DISCOVERY_TOPK_ATTN_LAYERS:-4}"
MIN_SIGN_CONSISTENCY="${MIN_SIGN_CONSISTENCY:-0.0}"
EXCLUDE_ATTN_LAYERS="${EXCLUDE_ATTN_LAYERS:-0}"

RUN_HEADS="${RUN_HEADS:-1}"
HEAD_SCAN_ROWS="${HEAD_SCAN_ROWS:-24}"
HEAD_SCAN_FACTORS="${HEAD_SCAN_FACTORS:-0.0,0.5,1.5}"
HEAD_TOPK="${HEAD_TOPK:-8}"
ALLOW_NONPOSITIVE_HEADS="${ALLOW_NONPOSITIVE_HEADS:-0}"

EPOCHS="${EPOCHS:-2}"
MLP_EPOCHS="${MLP_EPOCHS:-$EPOCHS}"
HEAD_EPOCHS="${HEAD_EPOCHS:-$EPOCHS}"
LR="${LR:-0.05}"
LAMBDA_NORM="${LAMBDA_NORM:-1e-4}"
STATE_MARGIN_WEIGHT="${STATE_MARGIN_WEIGHT:-0.0}"
GAIN_WEIGHT="${GAIN_WEIGHT:-1.0}"
TARGET_MARGIN="${TARGET_MARGIN:-0.0}"
TARGET_GAIN="${TARGET_GAIN:-0.0}"
SCORE_MODE="${SCORE_MODE:-avglogp}"
EMPTY_CACHE_EVERY="${EMPTY_CACHE_EVERY:-25}"
MAX_ALIASES_PER_SIDE="${MAX_ALIASES_PER_SIDE:-1}"
ALPHA_SWEEP="${ALPHA_SWEEP:-0,0.25,0.5,0.75,1.0,1.5,2.0}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "$ROOT"
MASTER_LOG="$ROOT/master.log"
STATUS="$ROOT/status.tsv"
PAIRS_DIR="$ROOT/pairs"
DISCOVERY_DIR="$ROOT/discovery"
SELECT_DIR="$DISCOVERY_DIR/selected"
MLP_OUT="$ROOT/train_logic_mlp"
HEAD_SCAN_OUT="$DISCOVERY_DIR/head_scan"
HEAD_OUT="$ROOT/train_logic_heads"

test -d "$MODEL"

if [[ -f LOCAL_HOME/anaconda3/etc/profile.d/conda.sh && -z "${SKIP_CONDA:-}" ]]; then
  # shellcheck disable=SC1091
  source LOCAL_HOME/anaconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV:-screscomp}"
fi
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

{
  echo -e "time\tstage\tstatus"
  echo -e "$(date -Is)\tinit\tstarting"
} > "$STATUS"
echo "[$(date -Is)] root=$ROOT model=$MODEL event=$EVENT pair_mode=$PAIR_MODE train=$TRAIN_ROWS val=$VAL_ROWS score=$SCORE_MODE" | tee -a "$MASTER_LOG"

echo -e "$(date -Is)\tprepare_pairs\trunning" >> "$STATUS"
prepare_args=(
  python -m screscomp.cli.prepare_step_dpo_pairs
  --out-dir "$PAIRS_DIR"
  --event "$EVENT"
  --train-rows "$TRAIN_ROWS"
  --val-rows "$VAL_ROWS"
  --eval-rows "$EVAL_ROWS"
  --split-unit "$SPLIT_UNIT"
  --pair-mode "$PAIR_MODE"
  --admission-policy "$ADMISSION_POLICY"
  --seed "$SEED"
)
if [[ -n "$SOURCE_FILTER_FIELD" && -n "$SOURCE_FILTER_CONTAINS" ]]; then
  prepare_args+=(--source-filter-field "$SOURCE_FILTER_FIELD" --source-filter-contains "$SOURCE_FILTER_CONTAINS")
fi
if [[ "$INCLUDE_INITIAL_IN_FULL_PROMPT" == "1" ]]; then
  prepare_args+=(--include-initial-in-full-prompt)
fi
if [[ "$SHUFFLE_PAIRS" == "1" ]]; then
  prepare_args+=(--shuffle)
fi
if [[ -n "$SOURCE_INPUT" ]]; then
  prepare_args+=(--input "$SOURCE_INPUT")
else
  prepare_args+=(--hf-dataset "$HF_DATASET" --hf-split "$HF_SPLIT")
  if [[ -n "$HF_CONFIG" ]]; then
    prepare_args+=(--hf-config "$HF_CONFIG")
  fi
fi
"${prepare_args[@]}" | tee -a "$MASTER_LOG"
echo -e "$(date -Is)\tprepare_pairs\tdone" >> "$STATUS"

echo -e "$(date -Is)\tscan_components\trunning" >> "$STATUS"
mkdir -p "$DISCOVERY_DIR/component_scan" "$SELECT_DIR"
CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.cecm_scan_component_contributions \
  --model "$MODEL" \
  --pairs-csv "$PAIRS_DIR/pairs.csv" \
  --event "$EVENT" \
  --split train \
  --start 0 \
  --max-rows "$DISCOVERY_ROWS" \
  --component-types attn,mlp \
  --apply-mode decision_tokens \
  --score-mode "$SCORE_MODE" \
  --max-aliases-per-side "$MAX_ALIASES_PER_SIDE" \
  --min-abs-delta 0.0 \
  --min-sign-consistency 0.50 \
  --allow-ci-cross-zero \
  --max-components-per-direction 16 \
  --torch-dtype "$TORCH_DTYPE" \
  --device "$DEVICE" \
  --out-dir "$DISCOVERY_DIR/component_scan" \
  > "$DISCOVERY_DIR/component_scan.log" 2>&1
echo -e "$(date -Is)\tscan_components\tdone" >> "$STATUS"

echo -e "$(date -Is)\tselect_components\trunning" >> "$STATUS"
python -m screscomp.cli.cecm_select_margin_components \
  --component-screen-csv "$DISCOVERY_DIR/component_scan/component_screen.csv" \
  --event "$EVENT" \
  --topk-mlp "$DISCOVERY_TOPK_MLP" \
  --topk-attn-layers "$DISCOVERY_TOPK_ATTN_LAYERS" \
  --mlp-policy positive \
  --min-sign-consistency "$MIN_SIGN_CONSISTENCY" \
  --exclude-attn-layers "$EXCLUDE_ATTN_LAYERS" \
  --out-dir "$SELECT_DIR" \
  | tee -a "$MASTER_LOG"
ATTN_LAYERS="$(tr -d '\r\n' < "$SELECT_DIR/attention_layers.txt")"
echo "[$(date -Is)] selected components=$SELECT_DIR/components.csv attn_layers=$ATTN_LAYERS" | tee -a "$MASTER_LOG"
echo -e "$(date -Is)\tselect_components\tdone" >> "$STATUS"

echo -e "$(date -Is)\ttrain_mlp\trunning" >> "$STATUS"
mkdir -p "$MLP_OUT"
CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.cecm_train_fixed_actuator \
  --model "$MODEL" \
  --pairs-csv "$PAIRS_DIR/pairs.csv" \
  --components-csv "$SELECT_DIR/components.csv" \
  --event "$EVENT" \
  --train-split train \
  --val-split val \
  --max-train-rows "$TRAIN_ROWS" \
  --max-val-rows "$VAL_ROWS" \
  --epochs "$MLP_EPOCHS" \
  --lr "$LR" \
  --lambda-norm "$LAMBDA_NORM" \
  --alpha-train 1.0 \
  --state-margin-weight "$STATE_MARGIN_WEIGHT" \
  --gain-weight "$GAIN_WEIGHT" \
  --target-margin "$TARGET_MARGIN" \
  --target-gain "$TARGET_GAIN" \
  --apply-mode decision_tokens \
  --score-mode "$SCORE_MODE" \
  --alpha-sweep "$ALPHA_SWEEP" \
  --max-aliases-per-side "$MAX_ALIASES_PER_SIDE" \
  --empty-cache-every "$EMPTY_CACHE_EVERY" \
  --torch-dtype "$TORCH_DTYPE" \
  --device "$DEVICE" \
  --out-dir "$MLP_OUT" \
  > "$MLP_OUT/train.log" 2>&1
echo -e "$(date -Is)\ttrain_mlp\tdone" >> "$STATUS"

if [[ "$RUN_HEADS" == "1" && -n "$ATTN_LAYERS" ]]; then
  echo -e "$(date -Is)\tscan_heads\trunning" >> "$STATUS"
  head_scan_args=(
    CUDA_VISIBLE_DEVICES="$GPU"
    python -m screscomp.cli.cecm_scan_attention_heads
    --model "$MODEL"
    --pairs-csv "$PAIRS_DIR/pairs.csv"
    --event "$EVENT"
    --attn-layers "$ATTN_LAYERS"
    --scan-factors "$HEAD_SCAN_FACTORS"
    --topk-heads "$HEAD_TOPK"
    --split train
    --scan-start 0
    --scan-max-rows "$HEAD_SCAN_ROWS"
    --score-mode "$SCORE_MODE"
    --score-apply-mode decision_tokens
    --max-aliases-per-side "$MAX_ALIASES_PER_SIDE"
    --torch-dtype "$TORCH_DTYPE"
    --device "$DEVICE"
    --out-dir "$HEAD_SCAN_OUT"
  )
  if [[ "$ALLOW_NONPOSITIVE_HEADS" == "1" ]]; then
    head_scan_args+=(--allow-nonpositive-selection)
  fi
  env "${head_scan_args[@]}" > "$HEAD_SCAN_OUT.log" 2>&1
  SELECTED_HEADS="$(tr -d '\r\n' < "$HEAD_SCAN_OUT/selected_heads.txt")"
  echo "[$(date -Is)] selected_heads=$SELECTED_HEADS" | tee -a "$MASTER_LOG"
  echo -e "$(date -Is)\tscan_heads\tdone" >> "$STATUS"

  if [[ -n "$SELECTED_HEADS" ]]; then
    echo -e "$(date -Is)\ttrain_heads\trunning" >> "$STATUS"
    mkdir -p "$HEAD_OUT"
    CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.cecm_train_attention_head_actuator \
      --model "$MODEL" \
      --pairs-csv "$PAIRS_DIR/pairs.csv" \
      --event "$EVENT" \
      --heads "$SELECTED_HEADS" \
      --train-split train \
      --val-split val \
      --max-train-rows "$TRAIN_ROWS" \
      --max-val-rows "$VAL_ROWS" \
      --epochs "$HEAD_EPOCHS" \
      --lr "$LR" \
      --lambda-norm "$LAMBDA_NORM" \
      --alpha-train 1.0 \
      --state-margin-weight "$STATE_MARGIN_WEIGHT" \
      --gain-weight "$GAIN_WEIGHT" \
      --target-margin "$TARGET_MARGIN" \
      --target-gain "$TARGET_GAIN" \
      --apply-mode decision_tokens \
      --score-mode "$SCORE_MODE" \
      --alpha-sweep "$ALPHA_SWEEP" \
      --max-aliases-per-side "$MAX_ALIASES_PER_SIDE" \
      --empty-cache-every "$EMPTY_CACHE_EVERY" \
      --torch-dtype "$TORCH_DTYPE" \
      --device "$DEVICE" \
      --out-dir "$HEAD_OUT" \
      > "$HEAD_OUT/train.log" 2>&1
    echo -e "$(date -Is)\ttrain_heads\tdone" >> "$STATUS"
  fi
else
  echo "[$(date -Is)] skip head path RUN_HEADS=$RUN_HEADS ATTN_LAYERS=$ATTN_LAYERS" | tee -a "$MASTER_LOG"
  echo -e "$(date -Is)\thead_path\tskipped" >> "$STATUS"
fi

echo "[$(date -Is)] done" | tee -a "$MASTER_LOG"
echo -e "$(date -Is)\tall\tdone" >> "$STATUS"
find "$ROOT" -maxdepth 3 \( -name pair_build_manifest.json -o -name component_selection_manifest.json -o -name head_selection_manifest.json -o -name alpha_summary.csv -o -name train_history.csv \) -print | sort
