#!/usr/bin/env bash
set -euo pipefail

# Full-test IMDb evaluation for an existing directional mlp/head/full run.
# This script:
#   1) prepares or reuses a full IMDb test prefix set
#   2) evaluates existing trained mlp/head/full actuators on the full test set
#   3) streams partial score summaries to disk during scoring

cd LOCAL_HOME/RPEC/projects/screscomp

GPU="${GPU:-0}"
MODEL="${MODEL:-edbeeching/gpt2-large-imdb}"
EVENT="${EVENT:-imdb_positive_sentiment}"
CONDA_ENV="${CONDA_ENV:-screscomp}"

TRAIN_ROOT="${TRAIN_ROOT:-runs/imdb_directional_mlp_head_full_20260605_190538}"
OUT_ROOT="${OUT_ROOT:-runs/imdb_directional_full_test_eval_$(date +%Y%m%d_%H%M%S)}"
EVAL_ENV_DIR="${EVAL_ENV_DIR:-$OUT_ROOT/eval_env_full_test}"

REBUILD_EVAL_ENV="${REBUILD_EVAL_ENV:-0}"
HF_DATASET="${HF_DATASET:-stanfordnlp/imdb}"
HF_SPLIT="${HF_SPLIT:-test}"
SOURCE_DATASET="${SOURCE_DATASET:-stanfordnlp/imdb}"
MAX_SOURCE_ROWS="${MAX_SOURCE_ROWS:-25000}"
EVAL_ROWS="${EVAL_ROWS:-25000}"
PREFIX_TOKEN_MIN="${PREFIX_TOKEN_MIN:-2}"
PREFIX_TOKEN_MAX="${PREFIX_TOKEN_MAX:-8}"
PREFIX_MODE="${PREFIX_MODE:-tokenizer}"
TOKENIZER="${TOKENIZER:-$MODEL}"
PROMPT_TEMPLATE="${PROMPT_TEMPLATE:-{prefix}}"
TARGET_SENTIMENT="${TARGET_SENTIMENT:-positive}"
COMPLETIONS_PER_PREFIX="${COMPLETIONS_PER_PREFIX:-4}"
SHUFFLE_PROMPTS="${SHUFFLE_PROMPTS:-1}"
SEED="${SEED:-42}"

EVAL_GROUPS="${EVAL_GROUPS:-mlp_positive,mlp_negative,att_positive,att_negative,full_positive,full_negative}"
ALPHA_SWEEP="${ALPHA_SWEEP:-0,0.25,0.5,1.0,2.0,4.0}"
EVAL_SPLIT="${EVAL_SPLIT:-eval}"
EVAL_SAMPLES_PER_PROMPT="${EVAL_SAMPLES_PER_PROMPT:-1}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
DO_SAMPLE="${DO_SAMPLE:-1}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:-50}"
SAME_SEED_ACROSS_ALPHA="${SAME_SEED_ACROSS_ALPHA:-1}"

EVAL_GENERATION_APPLY_MODE="${EVAL_GENERATION_APPLY_MODE:-all}"
EVAL_COMPONENT_APPLY_MODE="${EVAL_COMPONENT_APPLY_MODE:-all}"
EVAL_HEAD_APPLY_MODE="${EVAL_HEAD_APPLY_MODE:-all}"

SCORER_MODEL="${SCORER_MODEL:-siebert/sentiment-roberta-large-english}"
SCORER_DEVICE="${SCORER_DEVICE:-0}"
SCORE_TEXT="${SCORE_TEXT:-completion}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-16}"
SCORER_MAX_LENGTH="${SCORER_MAX_LENGTH:-512}"
SCORE_STREAM_SUMMARY_EVERY_BATCHES="${SCORE_STREAM_SUMMARY_EVERY_BATCHES:-32}"

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

echo "[$(date -Is)] full-test eval train_root=$TRAIN_ROOT out_root=$OUT_ROOT" | tee -a "$MASTER_LOG"
echo "[$(date -Is)] groups=$EVAL_GROUPS alphas=$ALPHA_SWEEP eval_rows=$EVAL_ROWS samples_per_prompt=$EVAL_SAMPLES_PER_PROMPT" | tee -a "$MASTER_LOG"

prepare_eval_env() {
  if [[ "$REBUILD_EVAL_ENV" != "1" && -f "$EVAL_ENV_DIR/prompts.jsonl" ]]; then
    echo "[$(date -Is)] reuse full-test eval env prompts=$EVAL_ENV_DIR/prompts.jsonl" | tee -a "$MASTER_LOG"
    return
  fi
  mkdir -p "$EVAL_ENV_DIR"
  echo -e "$(date -Is)\tprepare_eval_env\trunning" >> "$STATUS"
  prep_args=(
    python -m screscomp.cli.prepare_imdb_sentiment_env
    --hf-dataset "$HF_DATASET"
    --hf-split "$HF_SPLIT"
    --out-dir "$EVAL_ENV_DIR"
    --event "$EVENT"
    --source-dataset "$SOURCE_DATASET"
    --max-source-rows "$MAX_SOURCE_ROWS"
    --train-rows 0
    --val-rows 0
    --eval-rows "$EVAL_ROWS"
    --seed "$SEED"
    --prefix-token-min "$PREFIX_TOKEN_MIN"
    --prefix-token-max "$PREFIX_TOKEN_MAX"
    --prefix-mode "$PREFIX_MODE"
    --tokenizer "$TOKENIZER"
    --prompt-template "$PROMPT_TEMPLATE"
    --target-sentiment "$TARGET_SENTIMENT"
    --completions-per-prefix "$COMPLETIONS_PER_PREFIX"
    --reference-model "$MODEL"
  )
  if [[ "$SHUFFLE_PROMPTS" == "1" ]]; then
    prep_args+=(--shuffle)
  fi
  CUDA_VISIBLE_DEVICES="$GPU" "${prep_args[@]}" > "$EVAL_ENV_DIR/prepare.log" 2>&1
  echo -e "$(date -Is)\tprepare_eval_env\tdone" >> "$STATUS"
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
    --prompts-jsonl "$EVAL_ENV_DIR/prompts.jsonl"
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
    --overwrite
  )
  if [[ -n "$component_actuator" ]]; then
    gen_args+=(--actuator "$component_actuator")
  fi
  if [[ -n "$head_actuator" ]]; then
    gen_args+=(--head-actuator "$head_actuator")
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
    --progress-json "$out_dir/score_progress.json" \
    --score-text "$SCORE_TEXT" \
    --batch-size "$SCORE_BATCH_SIZE" \
    --scorer-model "$SCORER_MODEL" \
    --scorer-max-length "$SCORER_MAX_LENGTH" \
    --stream-summary-every-batches "$SCORE_STREAM_SUMMARY_EVERY_BATCHES" \
    --overwrite \
    --device "$SCORER_DEVICE" \
    > "$out_dir/score.log" 2>&1
  echo -e "$(date -Is)\teval_${control_name}_score\tdone" >> "$STATUS"
}

prepare_eval_env

MLP_POS_ACT="$TRAIN_ROOT/train_mlp_positive/fixed_actuator.pt"
MLP_NEG_ACT="$TRAIN_ROOT/train_mlp_negative/fixed_actuator.pt"
HEAD_POS_ACT="$TRAIN_ROOT/train_head_positive/head_actuator.pt"
HEAD_NEG_ACT="$TRAIN_ROOT/train_head_negative/head_actuator.pt"

test -f "$MLP_POS_ACT"
test -f "$MLP_NEG_ACT"
test -f "$HEAD_POS_ACT"
test -f "$HEAD_NEG_ACT"
test -f "$EVAL_ENV_DIR/prompts.jsonl"

IFS=',' read -r -a GROUP_LIST <<< "$EVAL_GROUPS"
for raw_group in "${GROUP_LIST[@]}"; do
  group="$(echo "$raw_group" | xargs)"
  case "$group" in
    mlp_positive)
      run_eval_group mlp_positive "$OUT_ROOT/eval_mlp_positive" "$MLP_POS_ACT" ""
      ;;
    mlp_negative)
      run_eval_group mlp_negative "$OUT_ROOT/eval_mlp_negative" "$MLP_NEG_ACT" ""
      ;;
    att_positive)
      run_eval_group att_positive "$OUT_ROOT/eval_att_positive" "" "$HEAD_POS_ACT"
      ;;
    att_negative)
      run_eval_group att_negative "$OUT_ROOT/eval_att_negative" "" "$HEAD_NEG_ACT"
      ;;
    full_positive)
      run_eval_group full_positive "$OUT_ROOT/eval_full_positive" "$MLP_POS_ACT" "$HEAD_POS_ACT"
      ;;
    full_negative)
      run_eval_group full_negative "$OUT_ROOT/eval_full_negative" "$MLP_NEG_ACT" "$HEAD_NEG_ACT"
      ;;
    "")
      ;;
    *)
      echo "Unknown group: $group" >&2
      exit 2
      ;;
  esac
done

echo -e "$(date -Is)\tsummarize\trunning" >> "$STATUS"
python -m screscomp.cli.summarize_imdb_sentiment_matrix \
  --root "$OUT_ROOT" \
  --out-csv "$OUT_ROOT/matrix_score_summary.csv" \
  | tee -a "$MASTER_LOG"
echo -e "$(date -Is)\tsummarize\tdone" >> "$STATUS"

echo -e "$(date -Is)\tall\tdone" >> "$STATUS"
echo "[$(date -Is)] full-test eval done out=$OUT_ROOT" | tee -a "$MASTER_LOG"
