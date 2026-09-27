#!/usr/bin/env bash
set -euo pipefail

# Directional top4 IMDb experiment matrix:
#   1) reuse existing directional top4 MLP/attention-layer selections
#   2) rescan attention heads inside the positive and negative attention layer groups
#   3) train four grouped actuators: mlp+/mlp-/head+/head-
#   4) evaluate mlp-only, att/head-only, and full combined groups under the same alpha sweep

cd LOCAL_HOME/RPEC/projects/screscomp

GPU="${GPU:-0}"
MODEL="${MODEL:-edbeeching/gpt2-large-imdb}"
EVENT="${EVENT:-imdb_positive_sentiment}"
CONDA_ENV="${CONDA_ENV:-screscomp}"

SCAN_ROOT="${SCAN_ROOT:-runs/imdb_dpo_prompt_positive_rollout_scan_20260605_115624}"
BASE_ROOT="${BASE_ROOT:-runs/imdb_sentiment_smoke_clean_20260604_155603}"
OUT_ROOT="${OUT_ROOT:-runs/imdb_directional_mlp_head_full_$(date +%Y%m%d_%H%M%S)}"

SELECT_ROOT="${SELECT_ROOT:-$SCAN_ROOT/discovery/directional_top4_selected}"
MLP_POS_COMPONENTS="${MLP_POS_COMPONENTS:-$SELECT_ROOT/mlp_positive_components.csv}"
MLP_NEG_COMPONENTS="${MLP_NEG_COMPONENTS:-$SELECT_ROOT/mlp_negative_components.csv}"
ATTN_POS_COMPONENTS="${ATTN_POS_COMPONENTS:-$SELECT_ROOT/attn_positive_components.csv}"
ATTN_NEG_COMPONENTS="${ATTN_NEG_COMPONENTS:-$SELECT_ROOT/attn_negative_components.csv}"

PAIRS_CSV="${PAIRS_CSV:-$BASE_ROOT/pairs/pairs.csv}"
EVAL_PROMPTS_JSONL="${EVAL_PROMPTS_JSONL:-$BASE_ROOT/eval_env/prompts.jsonl}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
VAL_SPLIT="${VAL_SPLIT:-val}"
EVAL_SPLIT="${EVAL_SPLIT:-eval}"

HEAD_TOPK="${HEAD_TOPK:-4}"
HEAD_SCAN_ROWS="${HEAD_SCAN_ROWS:-300}"
HEAD_SCAN_START="${HEAD_SCAN_START:-0}"
HEAD_SCAN_FACTORS="${HEAD_SCAN_FACTORS:-0.0,0.5,1.5}"
HEAD_SCAN_APPLY_MODE="${HEAD_SCAN_APPLY_MODE:-all}"
HEAD_REQUIRE_DIRECTIONAL_CI="${HEAD_REQUIRE_DIRECTIONAL_CI:-1}"
ALLOW_NONPOSITIVE_HEADS="${ALLOW_NONPOSITIVE_HEADS:-0}"

MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-512}"
MAX_VAL_ROWS="${MAX_VAL_ROWS:-300}"
EPOCHS="${EPOCHS:-2}"
LR="${LR:-0.05}"
LAMBDA_NORM="${LAMBDA_NORM:-1e-4}"
ALPHA_TRAIN="${ALPHA_TRAIN:-1.0}"
PREFERENCE_LOSS_MODE="${PREFERENCE_LOSS_MODE:-dpo}"
DPO_BETA="${DPO_BETA:-1.0}"
STATE_MARGIN_WEIGHT="${STATE_MARGIN_WEIGHT:-0.0}"
GAIN_WEIGHT="${GAIN_WEIGHT:-1.0}"
TARGET_MARGIN="${TARGET_MARGIN:-0.0}"
TARGET_GAIN="${TARGET_GAIN:-0.0}"
SCORE_MODE="${SCORE_MODE:-avglogp}"
TRAIN_APPLY_MODE_MLP="${TRAIN_APPLY_MODE_MLP:-all}"
TRAIN_APPLY_MODE_HEAD="${TRAIN_APPLY_MODE_HEAD:-all}"
TRAIN_CAUSAL_MASK_MLP="${TRAIN_CAUSAL_MASK_MLP:-1}"
TRAIN_CAUSAL_MASK_HEAD="${TRAIN_CAUSAL_MASK_HEAD:-1}"
EMPTY_CACHE_EVERY="${EMPTY_CACHE_EVERY:-25}"
MAX_ALIASES_PER_SIDE="${MAX_ALIASES_PER_SIDE:-1}"

ALPHA_SWEEP="${ALPHA_SWEEP:-0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}"
EVAL_GENERATION_APPLY_MODE="${EVAL_GENERATION_APPLY_MODE:-all}"
EVAL_COMPONENT_APPLY_MODE="${EVAL_COMPONENT_APPLY_MODE:-$TRAIN_APPLY_MODE_MLP}"
EVAL_HEAD_APPLY_MODE="${EVAL_HEAD_APPLY_MODE:-$TRAIN_APPLY_MODE_HEAD}"
EVAL_MAX_ROWS="${EVAL_MAX_ROWS:-128}"
EVAL_SAMPLES_PER_PROMPT="${EVAL_SAMPLES_PER_PROMPT:-2}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
DO_SAMPLE="${DO_SAMPLE:-1}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:-50}"
SEED="${SEED:-42}"
SAME_SEED_ACROSS_ALPHA="${SAME_SEED_ACROSS_ALPHA:-1}"

SCORER_MODEL="${SCORER_MODEL:-siebert/sentiment-roberta-large-english}"
SCORER_DEVICE="${SCORER_DEVICE:-0}"
SCORE_TEXT="${SCORE_TEXT:-completion}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-16}"
SCORER_MAX_LENGTH="${SCORER_MAX_LENGTH:-512}"
SCORE_STREAM_SUMMARY_EVERY_BATCHES="${SCORE_STREAM_SUMMARY_EVERY_BATCHES:-0}"

TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "$OUT_ROOT"
STATUS="$OUT_ROOT/status.tsv"
MASTER_LOG="$OUT_ROOT/master.log"

if [[ -f LOCAL_HOME/anaconda3/etc/profile.d/conda.sh && -z "${SKIP_CONDA:-}" ]]; then
  # shellcheck disable=SC1091
  source LOCAL_HOME/anaconda3/etc/profile.d/conda.sh
  conda activate "$CONDA_ENV"
fi
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

{
  echo -e "time\tstage\tstatus"
  echo -e "$(date -Is)\tinit\tstarting"
} > "$STATUS"

echo "[$(date -Is)] directional matrix scan_root=$SCAN_ROOT base_root=$BASE_ROOT out_root=$OUT_ROOT" | tee -a "$MASTER_LOG"
echo "[$(date -Is)] train_rows=$MAX_TRAIN_ROWS val_rows=$MAX_VAL_ROWS head_scan_rows=$HEAD_SCAN_ROWS alpha_sweep=$ALPHA_SWEEP" | tee -a "$MASTER_LOG"
echo "[$(date -Is)] apply mlp_train=$TRAIN_APPLY_MODE_MLP head_train=$TRAIN_APPLY_MODE_HEAD eval_comp=$EVAL_COMPONENT_APPLY_MODE eval_head=$EVAL_HEAD_APPLY_MODE" | tee -a "$MASTER_LOG"

test -f "$MLP_POS_COMPONENTS"
test -f "$MLP_NEG_COMPONENTS"
test -f "$ATTN_POS_COMPONENTS"
test -f "$ATTN_NEG_COMPONENTS"
test -f "$PAIRS_CSV"
test -f "$EVAL_PROMPTS_JSONL"

extract_layers() {
  local csv_path="$1"
  python - "$csv_path" <<'PY'
import csv, sys
from pathlib import Path
path = Path(sys.argv[1])
seen = []
seen_set = set()
with path.open("r", encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        value = str(row.get("layer_idx", "")).strip()
        if not value:
            continue
        if value not in seen_set:
            seen.append(value)
            seen_set.add(value)
print(",".join(seen))
PY
}

merge_layers() {
  local first="$1"
  local second="$2"
  python - "$first" "$second" <<'PY'
import sys

values = []
seen = set()
for raw in sys.argv[1:]:
    for item in str(raw).split(","):
        item = item.strip()
        if not item or item in seen:
            continue
        seen.add(item)
        values.append(item)
print(",".join(values))
PY
}

copy_head_group() {
  local src_csv="$1"
  local src_txt="$2"
  local dst_csv="$3"
  local dst_txt="$4"
  cp "$src_csv" "$dst_csv"
  cp "$src_txt" "$dst_txt"
}

run_head_scan() {
  local name="$1"
  local layers="$2"
  local out_dir="$3"
  mkdir -p "$out_dir"
  if [[ -z "$layers" ]]; then
    echo "Empty attn layer set for $name" >&2
    exit 2
  fi
  echo -e "$(date -Is)\thead_scan_${name}\trunning" >> "$STATUS"
  scan_args=(
    python -m screscomp.cli.cecm_scan_attention_heads
    --model "$MODEL"
    --pairs-csv "$PAIRS_CSV"
    --event "$EVENT"
    --attn-layers "$layers"
    --scan-factors "$HEAD_SCAN_FACTORS"
    --topk-heads "$HEAD_TOPK"
    --split "$TRAIN_SPLIT"
    --scan-start "$HEAD_SCAN_START"
    --scan-max-rows "$HEAD_SCAN_ROWS"
    --score-mode "$SCORE_MODE"
    --score-apply-mode "$HEAD_SCAN_APPLY_MODE"
    --max-aliases-per-side "$MAX_ALIASES_PER_SIDE"
    --torch-dtype "$TORCH_DTYPE"
    --device "$DEVICE"
    --out-dir "$out_dir"
  )
  if [[ "$HEAD_REQUIRE_DIRECTIONAL_CI" == "1" ]]; then
    scan_args+=(--require-directional-ci)
  fi
  if [[ "$ALLOW_NONPOSITIVE_HEADS" == "1" ]]; then
    scan_args+=(--allow-nonpositive-selection)
  fi
  CUDA_VISIBLE_DEVICES="$GPU" "${scan_args[@]}" > "$out_dir/head_scan.log" 2>&1
  echo -e "$(date -Is)\thead_scan_${name}\tdone" >> "$STATUS"
}

train_mlp_group() {
  local name="$1"
  local components_csv="$2"
  local out_dir="$3"
  mkdir -p "$out_dir"
  echo -e "$(date -Is)\ttrain_${name}\trunning" >> "$STATUS"
  extra_args=()
  if [[ "$TRAIN_CAUSAL_MASK_MLP" == "1" ]]; then
    extra_args+=(--causal-train-mask)
  fi
  CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.cecm_train_fixed_actuator \
    --model "$MODEL" \
    --pairs-csv "$PAIRS_CSV" \
    --components-csv "$components_csv" \
    --event "$EVENT" \
    --train-split "$TRAIN_SPLIT" \
    --val-split "$VAL_SPLIT" \
    --max-train-rows "$MAX_TRAIN_ROWS" \
    --max-val-rows "$MAX_VAL_ROWS" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --lambda-norm "$LAMBDA_NORM" \
    --alpha-train "$ALPHA_TRAIN" \
    --preference-loss-mode "$PREFERENCE_LOSS_MODE" \
    --dpo-beta "$DPO_BETA" \
    --state-margin-weight "$STATE_MARGIN_WEIGHT" \
    --gain-weight "$GAIN_WEIGHT" \
    --target-margin "$TARGET_MARGIN" \
    --target-gain "$TARGET_GAIN" \
    --apply-mode "$TRAIN_APPLY_MODE_MLP" \
    "${extra_args[@]}" \
    --score-mode "$SCORE_MODE" \
    --alpha-sweep "$ALPHA_SWEEP" \
    --max-aliases-per-side "$MAX_ALIASES_PER_SIDE" \
    --empty-cache-every "$EMPTY_CACHE_EVERY" \
    --torch-dtype "$TORCH_DTYPE" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --out-dir "$out_dir" \
    > "$out_dir/train.log" 2>&1
  echo -e "$(date -Is)\ttrain_${name}\tdone" >> "$STATUS"
}

train_head_group() {
  local name="$1"
  local heads_txt="$2"
  local out_dir="$3"
  mkdir -p "$out_dir"
  local heads
  heads="$(tr -d '\r\n' < "$heads_txt")"
  if [[ -z "$heads" ]]; then
    echo "No heads selected for $name: $heads_txt" >&2
    exit 2
  fi
  echo -e "$(date -Is)\ttrain_${name}\trunning" >> "$STATUS"
  extra_args=()
  if [[ "$TRAIN_CAUSAL_MASK_HEAD" == "1" ]]; then
    extra_args+=(--causal-train-mask)
  fi
  CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.cecm_train_attention_head_actuator \
    --model "$MODEL" \
    --pairs-csv "$PAIRS_CSV" \
    --event "$EVENT" \
    --heads "$heads" \
    --train-split "$TRAIN_SPLIT" \
    --val-split "$VAL_SPLIT" \
    --max-train-rows "$MAX_TRAIN_ROWS" \
    --max-val-rows "$MAX_VAL_ROWS" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --lambda-norm "$LAMBDA_NORM" \
    --alpha-train "$ALPHA_TRAIN" \
    --preference-loss-mode "$PREFERENCE_LOSS_MODE" \
    --dpo-beta "$DPO_BETA" \
    --state-margin-weight "$STATE_MARGIN_WEIGHT" \
    --gain-weight "$GAIN_WEIGHT" \
    --target-margin "$TARGET_MARGIN" \
    --target-gain "$TARGET_GAIN" \
    --apply-mode "$TRAIN_APPLY_MODE_HEAD" \
    "${extra_args[@]}" \
    --score-mode "$SCORE_MODE" \
    --alpha-sweep "$ALPHA_SWEEP" \
    --max-aliases-per-side "$MAX_ALIASES_PER_SIDE" \
    --empty-cache-every "$EMPTY_CACHE_EVERY" \
    --torch-dtype "$TORCH_DTYPE" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --out-dir "$out_dir" \
    > "$out_dir/train.log" 2>&1
  echo -e "$(date -Is)\ttrain_${name}\tdone" >> "$STATUS"
}

run_eval_group() {
  local control_name="$1"
  local out_dir="$2"
  local component_actuator="$3"
  local head_actuator="$4"
  mkdir -p "$out_dir"
  local gen_jsonl="$out_dir/generations.jsonl"
  local scored_jsonl="$out_dir/scored_generations.jsonl"
  local scored_csv="$out_dir/scored_generations.csv"
  local summary_csv="$out_dir/score_summary.csv"

  echo -e "$(date -Is)\teval_${control_name}_generate\trunning" >> "$STATUS"
  gen_args=(
    python -m screscomp.cli.run_imdb_sentiment_actuator_generation
    --model "$MODEL"
    --prompts-jsonl "$EVAL_PROMPTS_JSONL"
    --out-jsonl "$gen_jsonl"
    --control-name "$control_name"
    --alpha-sweep "$ALPHA_SWEEP"
    --generation-apply-mode "$EVAL_GENERATION_APPLY_MODE"
    --component-apply-mode "$EVAL_COMPONENT_APPLY_MODE"
    --head-apply-mode "$EVAL_HEAD_APPLY_MODE"
    --split "$EVAL_SPLIT"
    --samples-per-prompt "$EVAL_SAMPLES_PER_PROMPT"
    --generation-batch-size "$GENERATION_BATCH_SIZE"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --temperature "$TEMPERATURE"
    --top-p "$TOP_P"
    --top-k "$TOP_K"
    --seed "$SEED"
    --torch-dtype "$TORCH_DTYPE"
    --device "$DEVICE"
  )
  if [[ -n "$component_actuator" ]]; then
    gen_args+=(--actuator "$component_actuator")
  fi
  if [[ -n "$head_actuator" ]]; then
    gen_args+=(--head-actuator "$head_actuator")
  fi
  if [[ "${EVAL_MAX_ROWS:-0}" -gt 0 ]]; then
    gen_args+=(--max-rows "$EVAL_MAX_ROWS")
  fi
  if [[ "$DO_SAMPLE" == "0" ]]; then
    gen_args+=(--no-do-sample)
  fi
  if [[ "$SAME_SEED_ACROSS_ALPHA" == "1" ]]; then
    gen_args+=(--same-seed-across-alpha)
  fi
  CUDA_VISIBLE_DEVICES="$GPU" "${gen_args[@]}" > "$out_dir/generate.log" 2>&1
  echo -e "$(date -Is)\teval_${control_name}_generate\tdone" >> "$STATUS"

  echo -e "$(date -Is)\teval_${control_name}_score\trunning" >> "$STATUS"
  CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.score_imdb_sentiment_generations \
    --input "$gen_jsonl" \
    --out-jsonl "$scored_jsonl" \
    --out-csv "$scored_csv" \
    --summary-csv "$summary_csv" \
    --score-text "$SCORE_TEXT" \
    --batch-size "$SCORE_BATCH_SIZE" \
    --scorer-model "$SCORER_MODEL" \
    --scorer-max-length "$SCORER_MAX_LENGTH" \
    --stream-summary-every-batches "$SCORE_STREAM_SUMMARY_EVERY_BATCHES" \
    --progress-json "$out_dir/score_progress.json" \
    --overwrite \
    --device "$SCORER_DEVICE" \
    > "$out_dir/score.log" 2>&1
  echo -e "$(date -Is)\teval_${control_name}_score\tdone" >> "$STATUS"
}

ATTN_POS_LAYERS="$(extract_layers "$ATTN_POS_COMPONENTS")"
ATTN_NEG_LAYERS="$(extract_layers "$ATTN_NEG_COMPONENTS")"
ATTN_HEAD_POOL_LAYERS="$(merge_layers "$ATTN_POS_LAYERS" "$ATTN_NEG_LAYERS")"
echo "[$(date -Is)] attn_positive_layers=$ATTN_POS_LAYERS attn_negative_layers=$ATTN_NEG_LAYERS head_pool_layers=$ATTN_HEAD_POOL_LAYERS" | tee -a "$MASTER_LOG"

HEAD_SCAN_DIR="$OUT_ROOT/head_scan_pool"
HEAD_GROUP_DIR="$OUT_ROOT/head_groups"
mkdir -p "$HEAD_GROUP_DIR"

run_head_scan pool "$ATTN_HEAD_POOL_LAYERS" "$HEAD_SCAN_DIR"

copy_head_group \
  "$HEAD_SCAN_DIR/selected_boost_heads.csv" \
  "$HEAD_SCAN_DIR/selected_boost_heads.txt" \
  "$HEAD_GROUP_DIR/head_positive_heads.csv" \
  "$HEAD_GROUP_DIR/head_positive_heads.txt"
copy_head_group \
  "$HEAD_SCAN_DIR/selected_suppress_heads.csv" \
  "$HEAD_SCAN_DIR/selected_suppress_heads.txt" \
  "$HEAD_GROUP_DIR/head_negative_heads.csv" \
  "$HEAD_GROUP_DIR/head_negative_heads.txt"

train_mlp_group mlp_positive "$MLP_POS_COMPONENTS" "$OUT_ROOT/train_mlp_positive"
train_mlp_group mlp_negative "$MLP_NEG_COMPONENTS" "$OUT_ROOT/train_mlp_negative"
train_head_group head_positive "$HEAD_GROUP_DIR/head_positive_heads.txt" "$OUT_ROOT/train_head_positive"
train_head_group head_negative "$HEAD_GROUP_DIR/head_negative_heads.txt" "$OUT_ROOT/train_head_negative"

MLP_POS_ACT="$OUT_ROOT/train_mlp_positive/fixed_actuator.pt"
MLP_NEG_ACT="$OUT_ROOT/train_mlp_negative/fixed_actuator.pt"
HEAD_POS_ACT="$OUT_ROOT/train_head_positive/head_actuator.pt"
HEAD_NEG_ACT="$OUT_ROOT/train_head_negative/head_actuator.pt"

run_eval_group mlp_positive "$OUT_ROOT/eval_mlp_positive" "$MLP_POS_ACT" ""
run_eval_group mlp_negative "$OUT_ROOT/eval_mlp_negative" "$MLP_NEG_ACT" ""
run_eval_group att_positive "$OUT_ROOT/eval_att_positive" "" "$HEAD_POS_ACT"
run_eval_group att_negative "$OUT_ROOT/eval_att_negative" "" "$HEAD_NEG_ACT"
run_eval_group full_positive "$OUT_ROOT/eval_full_positive" "$MLP_POS_ACT" "$HEAD_POS_ACT"
run_eval_group full_negative "$OUT_ROOT/eval_full_negative" "$MLP_NEG_ACT" "$HEAD_NEG_ACT"

echo -e "$(date -Is)\tsummarize\trunning" >> "$STATUS"
python -m screscomp.cli.summarize_imdb_sentiment_matrix \
  --root "$OUT_ROOT" \
  --out-csv "$OUT_ROOT/matrix_score_summary.csv" \
  | tee -a "$MASTER_LOG"
echo -e "$(date -Is)\tsummarize\tdone" >> "$STATUS"

echo -e "$(date -Is)\tall\tdone" >> "$STATUS"
echo "[$(date -Is)] directional mlp/head/full matrix done out=$OUT_ROOT" | tee -a "$MASTER_LOG"
