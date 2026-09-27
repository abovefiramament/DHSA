#!/usr/bin/env bash
set -euo pipefail

# IMDb MLP timing matrix:
#   1) retrain mlp_positive / mlp_negative under multiple train apply modes
#   2) evaluate each trained actuator under multiple generation/component timings
#   3) score reward + KL with batched generation for faster turnaround

cd LOCAL_HOME/RPEC/projects/screscomp

GPU="${GPU:-0}"
MODEL="${MODEL:-edbeeching/gpt2-large-imdb}"
EVENT="${EVENT:-imdb_positive_sentiment}"
CONDA_ENV="${CONDA_ENV:-screscomp}"

SCAN_ROOT="${SCAN_ROOT:-runs/imdb_dpo_prompt_positive_rollout_scan_20260605_115624}"
BASE_ROOT="${BASE_ROOT:-runs/imdb_sentiment_smoke_clean_20260604_155603}"
OUT_ROOT="${OUT_ROOT:-runs/imdb_directional_mlp_timing_matrix_$(date +%Y%m%d_%H%M%S)}"

SELECT_ROOT="${SELECT_ROOT:-$SCAN_ROOT/discovery/directional_top4_selected}"
MLP_POS_COMPONENTS="${MLP_POS_COMPONENTS:-$SELECT_ROOT/mlp_positive_components.csv}"
MLP_NEG_COMPONENTS="${MLP_NEG_COMPONENTS:-$SELECT_ROOT/mlp_negative_components.csv}"

PAIRS_CSV="${PAIRS_CSV:-$BASE_ROOT/pairs/pairs.csv}"
EVAL_PROMPTS_JSONL="${EVAL_PROMPTS_JSONL:-$BASE_ROOT/eval_env/prompts.jsonl}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
VAL_SPLIT="${VAL_SPLIT:-val}"
EVAL_SPLIT="${EVAL_SPLIT:-eval}"

MLP_GROUPS="${MLP_GROUPS:-mlp_positive,mlp_negative}"
TRAIN_APPLY_MODES="${TRAIN_APPLY_MODES:-prefill,first_decode,first_8_decode,all}"
EVAL_APPLY_MODES="${EVAL_APPLY_MODES:-prefill,first_decode,first_8_decode,all}"

MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-1200}"
MAX_VAL_ROWS="${MAX_VAL_ROWS:-300}"
EVAL_MAX_ROWS="${EVAL_MAX_ROWS:-256}"
EVAL_SAMPLES_PER_PROMPT="${EVAL_SAMPLES_PER_PROMPT:-1}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-32}"

EPOCHS="${EPOCHS:-3}"
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
TRAIN_CAUSAL_MASK="${TRAIN_CAUSAL_MASK:-1}"
EMPTY_CACHE_EVERY="${EMPTY_CACHE_EVERY:-25}"
MAX_ALIASES_PER_SIDE="${MAX_ALIASES_PER_SIDE:-1}"

ALPHA_SWEEP="${ALPHA_SWEEP:-0,0.25,0.5,1.0,2.0}"
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
SCORE_STREAM_SUMMARY_EVERY_BATCHES="${SCORE_STREAM_SUMMARY_EVERY_BATCHES:-16}"
KL_STREAM_SUMMARY_EVERY_ROWS="${KL_STREAM_SUMMARY_EVERY_ROWS:-32}"
KL_BATCH_SIZE="${KL_BATCH_SIZE:-32}"
RESUME="${RESUME:-1}"

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

echo "[$(date -Is)] mlp timing matrix base_root=$BASE_ROOT out_root=$OUT_ROOT" | tee -a "$MASTER_LOG"
echo "[$(date -Is)] groups=$MLP_GROUPS train_modes=$TRAIN_APPLY_MODES eval_modes=$EVAL_APPLY_MODES alpha_sweep=$ALPHA_SWEEP" | tee -a "$MASTER_LOG"
echo "[$(date -Is)] train_rows=$MAX_TRAIN_ROWS val_rows=$MAX_VAL_ROWS eval_rows=$EVAL_MAX_ROWS gen_batch=$GENERATION_BATCH_SIZE" | tee -a "$MASTER_LOG"

run_kl_with_backoff() {
  local out_dir="$1"
  shift
  local batch_size="$KL_BATCH_SIZE"
  local first_attempt="$1"
  shift
  local rc=0
  local attempt=1
  while true; do
    local kl_args=("$@")
    kl_args+=(--batch-size "$batch_size")
    if [[ "$first_attempt" == "1" && "$attempt" == "1" ]]; then
      kl_args+=(--overwrite)
    else
      kl_args+=(--resume)
    fi
    echo "[$(date -Is)] kl-run out=$out_dir attempt=$attempt batch_size=$batch_size mode=$(if [[ "$first_attempt" == "1" && "$attempt" == "1" ]]; then echo overwrite; else echo resume; fi)" | tee -a "$MASTER_LOG"
    if CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.compute_imdb_generation_kl "${kl_args[@]}" > "$out_dir/kl.log" 2>&1; then
      return 0
    fi
    rc=$?
    if [[ "$batch_size" -le 1 ]]; then
      echo "[$(date -Is)] kl-run failed out=$out_dir attempt=$attempt batch_size=$batch_size rc=$rc" | tee -a "$MASTER_LOG"
      return "$rc"
    fi
    local next_batch_size=$((batch_size / 2))
    if [[ "$next_batch_size" -lt 1 ]]; then
      next_batch_size=1
    fi
    echo "[$(date -Is)] kl-run retry out=$out_dir rc=$rc batch_size=$batch_size next_batch_size=$next_batch_size" | tee -a "$MASTER_LOG"
    batch_size="$next_batch_size"
    attempt=$((attempt + 1))
  done
}

test -f "$MLP_POS_COMPONENTS"
test -f "$MLP_NEG_COMPONENTS"
test -f "$PAIRS_CSV"
test -f "$EVAL_PROMPTS_JSONL"

component_csv_for_group() {
  local group="$1"
  case "$group" in
    mlp_positive)
      echo "$MLP_POS_COMPONENTS"
      ;;
    mlp_negative)
      echo "$MLP_NEG_COMPONENTS"
      ;;
    *)
      echo "Unknown MLP group: $group" >&2
      exit 2
      ;;
  esac
}

train_group_mode() {
  local group="$1"
  local train_mode="$2"
  local components_csv="$3"
  local out_dir="$4"
  mkdir -p "$out_dir"
  if [[ "$RESUME" == "1" && -f "$out_dir/fixed_actuator.pt" ]]; then
    echo "[$(date -Is)] resume-skip train group=$group train_mode=$train_mode" | tee -a "$MASTER_LOG"
    return
  fi
  echo -e "$(date -Is)\ttrain_${group}_${train_mode}\trunning" >> "$STATUS"
  extra_args=()
  if [[ "$TRAIN_CAUSAL_MASK" == "1" ]]; then
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
    --apply-mode "$train_mode" \
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
  echo -e "$(date -Is)\ttrain_${group}_${train_mode}\tdone" >> "$STATUS"
}

eval_group_mode() {
  local group="$1"
  local train_mode="$2"
  local eval_mode="$3"
  local actuator_path="$4"
  local out_dir="$5"
  mkdir -p "$out_dir"
  local gen_jsonl="$out_dir/generations.jsonl"
  local scored_jsonl="$out_dir/scored_generations.jsonl"
  local scored_csv="$out_dir/scored_generations.csv"
  local summary_csv="$out_dir/score_summary.csv"
  local score_manifest="$out_dir/score_manifest.json"
  local kl_jsonl="$out_dir/kl_scored_generations.jsonl"
  local kl_csv="$out_dir/kl_scored_generations.csv"
  local kl_manifest="$out_dir/kl_manifest.json"

  if [[ "$RESUME" != "1" || ! -f "$score_manifest" ]]; then
    echo -e "$(date -Is)\teval_${group}_${train_mode}_${eval_mode}_generate\trunning" >> "$STATUS"
    gen_args=(
      python -m screscomp.cli.run_imdb_sentiment_actuator_generation
      --model "$MODEL"
      --prompts-jsonl "$EVAL_PROMPTS_JSONL"
      --out-jsonl "$gen_jsonl"
      --actuator "$actuator_path"
      --control-name "${group}__train_${train_mode}__eval_${eval_mode}"
      --alpha-sweep "$ALPHA_SWEEP"
      --generation-apply-mode "$eval_mode"
      --component-apply-mode "$eval_mode"
      --split "$EVAL_SPLIT"
      --max-rows "$EVAL_MAX_ROWS"
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
    if [[ "$DO_SAMPLE" == "0" ]]; then
      gen_args+=(--no-do-sample)
    fi
    if [[ "$SAME_SEED_ACROSS_ALPHA" == "1" ]]; then
      gen_args+=(--same-seed-across-alpha)
    fi
    if [[ "$RESUME" != "1" ]]; then
      gen_args+=(--overwrite)
    fi
    CUDA_VISIBLE_DEVICES="$GPU" "${gen_args[@]}" > "$out_dir/generate.log" 2>&1
    echo -e "$(date -Is)\teval_${group}_${train_mode}_${eval_mode}_generate\tdone" >> "$STATUS"

    echo -e "$(date -Is)\teval_${group}_${train_mode}_${eval_mode}_score\trunning" >> "$STATUS"
    CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.score_imdb_sentiment_generations \
      --input "$gen_jsonl" \
      --out-jsonl "$scored_jsonl" \
      --out-csv "$scored_csv" \
      --summary-csv "$summary_csv" \
      --progress-json "$out_dir/score_progress.json" \
      --score-text "$SCORE_TEXT" \
      --batch-size "$SCORE_BATCH_SIZE" \
      --scorer-model "$SCORER_MODEL" \
      --scorer-max-length "$SCORER_MAX_LENGTH" \
      --stream-summary-every-batches "$SCORE_STREAM_SUMMARY_EVERY_BATCHES" \
      --overwrite \
      --device "$SCORER_DEVICE" \
      > "$out_dir/score.log" 2>&1
    echo -e "$(date -Is)\teval_${group}_${train_mode}_${eval_mode}_score\tdone" >> "$STATUS"
  else
    echo "[$(date -Is)] resume-skip score group=$group train_mode=$train_mode eval_mode=$eval_mode" | tee -a "$MASTER_LOG"
  fi

  echo -e "$(date -Is)\teval_${group}_${train_mode}_${eval_mode}_kl\trunning" >> "$STATUS"
  kl_args=(
    --input-jsonl "$scored_jsonl"
    --out-jsonl "$kl_jsonl"
    --out-csv "$kl_csv"
    --summary-csv "$summary_csv"
    --progress-json "$out_dir/kl_progress.json"
    --model "$MODEL"
    --torch-dtype "$TORCH_DTYPE"
    --device "$DEVICE"
    --stream-summary-every-rows "$KL_STREAM_SUMMARY_EVERY_ROWS"
  )
  if [[ "$RESUME" == "1" ]]; then
    run_kl_with_backoff "$out_dir" 0 "${kl_args[@]}"
  else
    run_kl_with_backoff "$out_dir" 1 "${kl_args[@]}"
  fi
  echo -e "$(date -Is)\teval_${group}_${train_mode}_${eval_mode}_kl\tdone" >> "$STATUS"
}

IFS=',' read -r -a GROUP_ARRAY <<< "$MLP_GROUPS"
IFS=',' read -r -a TRAIN_MODE_ARRAY <<< "$TRAIN_APPLY_MODES"
IFS=',' read -r -a EVAL_MODE_ARRAY <<< "$EVAL_APPLY_MODES"

for raw_group in "${GROUP_ARRAY[@]}"; do
  group="$(echo "$raw_group" | xargs)"
  [[ -n "$group" ]] || continue
  components_csv="$(component_csv_for_group "$group")"
  for raw_train_mode in "${TRAIN_MODE_ARRAY[@]}"; do
    train_mode="$(echo "$raw_train_mode" | xargs)"
    [[ -n "$train_mode" ]] || continue
    train_dir="$OUT_ROOT/train__${group}__${train_mode}"
    train_group_mode "$group" "$train_mode" "$components_csv" "$train_dir"
    actuator_path="$train_dir/fixed_actuator.pt"
    test -f "$actuator_path"
    for raw_eval_mode in "${EVAL_MODE_ARRAY[@]}"; do
      eval_mode="$(echo "$raw_eval_mode" | xargs)"
      [[ -n "$eval_mode" ]] || continue
      eval_dir="$OUT_ROOT/eval__${group}__train_${train_mode}__eval_${eval_mode}"
      eval_group_mode "$group" "$train_mode" "$eval_mode" "$actuator_path" "$eval_dir"
    done
  done
done

echo -e "$(date -Is)\tsummarize\trunning" >> "$STATUS"
python -m screscomp.cli.summarize_imdb_sentiment_matrix \
  --root "$OUT_ROOT" \
  --out-csv "$OUT_ROOT/matrix_score_summary.csv" \
  > "$OUT_ROOT/summarize.log" 2>&1
echo -e "$(date -Is)\tsummarize\tdone" >> "$STATUS"
echo -e "$(date -Is)\tall\tdone" >> "$STATUS"
echo "[$(date -Is)] mlp timing matrix done out=$OUT_ROOT" | tee -a "$MASTER_LOG"
