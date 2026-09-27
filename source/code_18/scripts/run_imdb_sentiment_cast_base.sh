#!/usr/bin/env bash
set -euo pipefail

# IMDb sentiment CAST/CECM base path:
#   1) build DPO-style IMDb prefix environment
#   2) optionally generate start-policy completions from a GPT-2-large IMDb SFT model
#   3) optionally score completions with the SiEBERT sentiment classifier
#   4) convert scored completions to generic y_plus/y_minus actuator pairs
#   5) reuse the existing component scan/select/train core with a DPO-shaped gain loss

cd LOCAL_HOME/RPEC/projects/screscomp

GPU="${GPU:-0}"
MODEL="${MODEL:-edbeeching/gpt2-large-imdb}"
ROOT="${ROOT:-data_preference/cast_imdb_sentiment_gpt2large_v0}"

SOURCE_INPUT="${SOURCE_INPUT:-}"
HF_DATASET="${HF_DATASET:-stanfordnlp/imdb}"
HF_CONFIG="${HF_CONFIG:-}"
HF_SPLIT="${HF_SPLIT:-train}"
USE_SEPARATE_EVAL_ENV="${USE_SEPARATE_EVAL_ENV:-0}"
EVAL_HF_DATASET="${EVAL_HF_DATASET:-$HF_DATASET}"
EVAL_HF_CONFIG="${EVAL_HF_CONFIG:-$HF_CONFIG}"
EVAL_HF_SPLIT="${EVAL_HF_SPLIT:-test}"
EVENT="${EVENT:-imdb_positive_sentiment}"

TRAIN_PROMPTS="${TRAIN_PROMPTS:-25000}"
VAL_PROMPTS="${VAL_PROMPTS:-0}"
EVAL_PROMPTS="${EVAL_PROMPTS:-0}"
PREFIX_MODE="${PREFIX_MODE:-tokenizer}"
TOKENIZER="${TOKENIZER:-$MODEL}"
PREFIX_TOKEN_MIN="${PREFIX_TOKEN_MIN:-2}"
PREFIX_TOKEN_MAX="${PREFIX_TOKEN_MAX:-8}"
SEED="${SEED:-42}"
SHUFFLE_PROMPTS="${SHUFFLE_PROMPTS:-1}"
PROMPT_TEMPLATE="${PROMPT_TEMPLATE-}"
if [[ -z "$PROMPT_TEMPLATE" ]]; then
  PROMPT_TEMPLATE="{prefix}"
fi
TARGET_SENTIMENT="${TARGET_SENTIMENT:-positive}"
REWARD_SCORE_FIELD="${REWARD_SCORE_FIELD:-positive_sentiment_score}"

RUN_GENERATE="${RUN_GENERATE:-0}"
RUN_SCORE="${RUN_SCORE:-0}"
RUN_EVAL="${RUN_EVAL:-0}"
PREPARE_EVAL_ENV="${PREPARE_EVAL_ENV:-$RUN_EVAL}"
EVAL_ONLY="${EVAL_ONLY:-0}"
STOP_AFTER_ENV="${STOP_AFTER_ENV:-0}"
STOP_AFTER_PAIRS="${STOP_AFTER_PAIRS:-0}"
STOP_AFTER_SCAN="${STOP_AFTER_SCAN:-0}"
STOP_AFTER_SELECT="${STOP_AFTER_SELECT:-0}"
REBUILD_ENV="${REBUILD_ENV:-0}"

COMPLETIONS_PER_PREFIX="${COMPLETIONS_PER_PREFIX:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-504}"
GENERATION_SPLIT="${GENERATION_SPLIT:-all}"
GENERATION_START="${GENERATION_START:-0}"
GENERATION_MAX_ROWS="${GENERATION_MAX_ROWS:-}"
GENERATION_DO_SAMPLE="${GENERATION_DO_SAMPLE:-1}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:-50}"
STOP_STRINGS="${STOP_STRINGS:-}"

SCORER_MODEL="${SCORER_MODEL:-siebert/sentiment-roberta-large-english}"
SCORER_DEVICE="${SCORER_DEVICE:-0}"
SCORE_TEXT="${SCORE_TEXT:-completion}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-16}"
SCORED_GENERATIONS="${SCORED_GENERATIONS:-}"

DISCOVERY_SCAN_MODE="${DISCOVERY_SCAN_MODE:-rollout}"
DISCOVERY_SPLIT="${DISCOVERY_SPLIT:-train}"
if [[ -z "${DISCOVERY_APPLY_MODE+x}" ]]; then
  if [[ "$DISCOVERY_SCAN_MODE" == "rollout" ]]; then
    DISCOVERY_APPLY_MODE="prefill"
  else
    DISCOVERY_APPLY_MODE="decision_tokens"
  fi
fi
DISCOVERY_ROWS="${DISCOVERY_ROWS:-60}"
DISCOVERY_SAMPLES_PER_PROMPT="${DISCOVERY_SAMPLES_PER_PROMPT:-1}"
DISCOVERY_GENERATION_BATCH_SIZE="${DISCOVERY_GENERATION_BATCH_SIZE:-1}"
DISCOVERY_MAX_NEW_TOKENS="${DISCOVERY_MAX_NEW_TOKENS:-64}"
DISCOVERY_DO_SAMPLE="${DISCOVERY_DO_SAMPLE:-$GENERATION_DO_SAMPLE}"
DISCOVERY_TEMPERATURE="${DISCOVERY_TEMPERATURE:-$TEMPERATURE}"
DISCOVERY_TOP_P="${DISCOVERY_TOP_P:-$TOP_P}"
DISCOVERY_TOP_K="${DISCOVERY_TOP_K:-$TOP_K}"
DISCOVERY_SCORE_TEXT="${DISCOVERY_SCORE_TEXT:-$SCORE_TEXT}"
DISCOVERY_SCORER_BATCH_SIZE="${DISCOVERY_SCORER_BATCH_SIZE:-$SCORE_BATCH_SIZE}"
DISCOVERY_SCORER_MAX_LENGTH="${DISCOVERY_SCORER_MAX_LENGTH:-512}"
case "${TARGET_SENTIMENT,,}" in
  negative)
    DEFAULT_DISCOVERY_TARGET_LABEL="NEGATIVE"
    DEFAULT_DISCOVERY_SOURCE_LABEL="POSITIVE"
    ;;
  *)
    DEFAULT_DISCOVERY_TARGET_LABEL="POSITIVE"
    DEFAULT_DISCOVERY_SOURCE_LABEL="NEGATIVE"
    ;;
esac
DISCOVERY_TARGET_LABEL="${DISCOVERY_TARGET_LABEL:-$DEFAULT_DISCOVERY_TARGET_LABEL}"
DISCOVERY_SOURCE_LABEL="${DISCOVERY_SOURCE_LABEL:-$DEFAULT_DISCOVERY_SOURCE_LABEL}"
DISCOVERY_TOPK_MLP="${DISCOVERY_TOPK_MLP:-4}"
DISCOVERY_TOPK_NEGATIVE_MLP="${DISCOVERY_TOPK_NEGATIVE_MLP:-4}"
DISCOVERY_TOPK_ATTN_LAYERS="${DISCOVERY_TOPK_ATTN_LAYERS:-4}"
MLP_POLICY="${MLP_POLICY:-positive}"
if [[ "$DISCOVERY_SCAN_MODE" == "rollout" ]]; then
  DEFAULT_MIN_SIGN_CONSISTENCY="0.50"
  DEFAULT_MIN_MLP_ABS_MEAN_DELTA="0.04"
  DEFAULT_MIN_ATTN_ABS_MEAN_DELTA="0.04"
  DEFAULT_MIN_DIRECTIONAL_TRANSITION_RATE="0.50"
  DEFAULT_MIN_ABLATED_FORMAT_OK="0.90"
  DEFAULT_MAX_FORMAT_COLLAPSE_RATE="0.10"
  DEFAULT_MIN_ABS_MEAN_TARGET_DROP="0.04"
  DEFAULT_MIN_ABS_MEAN_SOURCE_RISE="0.04"
else
  DEFAULT_MIN_SIGN_CONSISTENCY="0.60"
  DEFAULT_MIN_MLP_ABS_MEAN_DELTA="0.30"
  DEFAULT_MIN_ATTN_ABS_MEAN_DELTA="0.30"
  DEFAULT_MIN_DIRECTIONAL_TRANSITION_RATE="0.0"
  DEFAULT_MIN_ABLATED_FORMAT_OK="0.0"
  DEFAULT_MAX_FORMAT_COLLAPSE_RATE="1.0"
  DEFAULT_MIN_ABS_MEAN_TARGET_DROP="0.0"
  DEFAULT_MIN_ABS_MEAN_SOURCE_RISE="0.0"
fi
MIN_SIGN_CONSISTENCY="${MIN_SIGN_CONSISTENCY:-$DEFAULT_MIN_SIGN_CONSISTENCY}"
MLP_REQUIRE_DIRECTIONAL_CI="${MLP_REQUIRE_DIRECTIONAL_CI:-1}"
MIN_MLP_ABS_MEAN_DELTA="${MIN_MLP_ABS_MEAN_DELTA:-$DEFAULT_MIN_MLP_ABS_MEAN_DELTA}"
MIN_ATTN_ABS_MEAN_DELTA="${MIN_ATTN_ABS_MEAN_DELTA:-$DEFAULT_MIN_ATTN_ABS_MEAN_DELTA}"
MIN_DIRECTIONAL_TRANSITION_RATE="${MIN_DIRECTIONAL_TRANSITION_RATE:-$DEFAULT_MIN_DIRECTIONAL_TRANSITION_RATE}"
MIN_ABLATED_FORMAT_OK="${MIN_ABLATED_FORMAT_OK:-$DEFAULT_MIN_ABLATED_FORMAT_OK}"
MAX_FORMAT_COLLAPSE_RATE="${MAX_FORMAT_COLLAPSE_RATE:-$DEFAULT_MAX_FORMAT_COLLAPSE_RATE}"
MIN_ABS_MEAN_TARGET_DROP="${MIN_ABS_MEAN_TARGET_DROP:-$DEFAULT_MIN_ABS_MEAN_TARGET_DROP}"
MIN_ABS_MEAN_SOURCE_RISE="${MIN_ABS_MEAN_SOURCE_RISE:-$DEFAULT_MIN_ABS_MEAN_SOURCE_RISE}"
EXCLUDE_MLP_LAYERS="${EXCLUDE_MLP_LAYERS:-0}"
EXCLUDE_ATTN_LAYERS="${EXCLUDE_ATTN_LAYERS:-}"
ALLOW_EMPTY_ATTN="${ALLOW_EMPTY_ATTN:-1}"

RUN_HEADS="${RUN_HEADS:-0}"
HEAD_SCAN_ROWS="${HEAD_SCAN_ROWS:-24}"
HEAD_SCAN_FACTORS="${HEAD_SCAN_FACTORS:-0.0,0.5,1.5}"
HEAD_TOPK="${HEAD_TOPK:-8}"
ALLOW_NONPOSITIVE_HEADS="${ALLOW_NONPOSITIVE_HEADS:-0}"

EPOCHS="${EPOCHS:-2}"
MLP_EPOCHS="${MLP_EPOCHS:-$EPOCHS}"
HEAD_EPOCHS="${HEAD_EPOCHS:-$EPOCHS}"
LR="${LR:-0.05}"
LAMBDA_NORM="${LAMBDA_NORM:-1e-4}"
ALPHA_TRAIN="${ALPHA_TRAIN:-1.0}"
ALPHA_SWEEP="${ALPHA_SWEEP:-0,0.25,0.5,0.75,1.0,1.5,2.0}"
EVAL_ALPHA_SWEEP="${EVAL_ALPHA_SWEEP:-$ALPHA_SWEEP}"
TRAIN_APPLY_MODE="${TRAIN_APPLY_MODE:-$DISCOVERY_APPLY_MODE}"
HEAD_TRAIN_APPLY_MODE="${HEAD_TRAIN_APPLY_MODE:-$TRAIN_APPLY_MODE}"
HEAD_SCAN_APPLY_MODE="${HEAD_SCAN_APPLY_MODE:-$TRAIN_APPLY_MODE}"
if [[ -z "${TRAIN_CAUSAL_MASK+x}" ]]; then
  if [[ "$DISCOVERY_SCAN_MODE" == "rollout" ]]; then
    TRAIN_CAUSAL_MASK="1"
  else
    TRAIN_CAUSAL_MASK="0"
  fi
fi
HEAD_TRAIN_CAUSAL_MASK="${HEAD_TRAIN_CAUSAL_MASK:-$TRAIN_CAUSAL_MASK}"
PREFERENCE_LOSS_MODE="${PREFERENCE_LOSS_MODE:-dpo}"
DPO_BETA="${DPO_BETA:-1.0}"
STATE_MARGIN_WEIGHT="${STATE_MARGIN_WEIGHT:-0.0}"
GAIN_WEIGHT="${GAIN_WEIGHT:-1.0}"
TARGET_MARGIN="${TARGET_MARGIN:-0.0}"
TARGET_GAIN="${TARGET_GAIN:-0.0}"
SCORE_MODE="${SCORE_MODE:-avglogp}"
EMPTY_CACHE_EVERY="${EMPTY_CACHE_EVERY:-25}"
MAX_ALIASES_PER_SIDE="${MAX_ALIASES_PER_SIDE:-1}"
MAX_TRAIN_PAIRS="${MAX_TRAIN_PAIRS:-240}"
MAX_VAL_PAIRS="${MAX_VAL_PAIRS:-60}"
PAIR_MAX_PROMPTS_PER_SPLIT="${PAIR_MAX_PROMPTS_PER_SPLIT:-0}"
MIN_SCORE_MARGIN="${MIN_SCORE_MARGIN:-0.1}"
MAX_PAIRS_PER_PROMPT="${MAX_PAIRS_PER_PROMPT:-0}"
PAIR_SCORE_FIELD="${PAIR_SCORE_FIELD:-$REWARD_SCORE_FIELD}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"

EVAL_SPLIT="${EVAL_SPLIT:-eval}"
EVAL_START="${EVAL_START:-0}"
EVAL_MAX_ROWS="${EVAL_MAX_ROWS:-200}"
FINAL_EVAL_PROMPTS="${FINAL_EVAL_PROMPTS:-$EVAL_PROMPTS}"
if [[ "$USE_SEPARATE_EVAL_ENV" == "1" && "$FINAL_EVAL_PROMPTS" == "0" ]]; then
  FINAL_EVAL_PROMPTS="$EVAL_MAX_ROWS"
fi
EVAL_SAMPLES_PER_PROMPT="${EVAL_SAMPLES_PER_PROMPT:-4}"
EVAL_CONTROL_NAME="${EVAL_CONTROL_NAME:-mlp_positive}"
if [[ -z "${EVAL_APPLY_MODE+x}" ]]; then
  case "$DISCOVERY_APPLY_MODE" in
    prefill|decode|all|first_decode|first_*_decode)
      EVAL_APPLY_MODE="$DISCOVERY_APPLY_MODE"
      ;;
    *)
      EVAL_APPLY_MODE="prefill"
      ;;
  esac
fi
EVAL_ACTUATOR="${EVAL_ACTUATOR-$ROOT/train_mlp_positive/fixed_actuator.pt}"
EVAL_HEAD_ACTUATOR="${EVAL_HEAD_ACTUATOR:-}"
EVAL_COMPONENT_APPLY_MODE="${EVAL_COMPONENT_APPLY_MODE:-}"
EVAL_HEAD_APPLY_MODE="${EVAL_HEAD_APPLY_MODE:-}"
EVAL_DO_SAMPLE="${EVAL_DO_SAMPLE:-$GENERATION_DO_SAMPLE}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-$TEMPERATURE}"
EVAL_TOP_P="${EVAL_TOP_P:-$TOP_P}"
EVAL_TOP_K="${EVAL_TOP_K:-$TOP_K}"
EVAL_SAME_SEED_ACROSS_ALPHA="${EVAL_SAME_SEED_ACROSS_ALPHA:-0}"
EVAL_SCORE_TEXT="${EVAL_SCORE_TEXT:-completion}"

ENV_DIR="$ROOT/env"
PROMPTS_JSONL="$ENV_DIR/prompts.jsonl"
EVAL_ENV_DIR="${EVAL_ENV_DIR:-$ROOT/eval_env}"
START_DIR="$ROOT/start_policy"
GENERATIONS_JSONL="${GENERATIONS_JSONL:-$START_DIR/generations.jsonl}"
if [[ -z "$SCORED_GENERATIONS" ]]; then
  SCORED_GENERATIONS="$START_DIR/scored_generations.jsonl"
fi
PAIRS_DIR="$ROOT/pairs"
DISCOVERY_DIR="$ROOT/discovery"
SELECT_DIR="$DISCOVERY_DIR/selected"
MLP_OUT="${MLP_OUT:-$ROOT/train_mlp_positive}"
TRAIN_COMPONENTS_CSV="${TRAIN_COMPONENTS_CSV:-$SELECT_DIR/components.csv}"
HEAD_SCAN_OUT="$DISCOVERY_DIR/head_scan"
HEAD_OUT="${HEAD_OUT:-$ROOT/train_heads}"
if [[ -z "${EVAL_PROMPTS_JSONL:-}" && ( "$USE_SEPARATE_EVAL_ENV" == "1" || ( "$EVAL_ONLY" == "1" && -f "$EVAL_ENV_DIR/prompts.jsonl" ) ) ]]; then
  EVAL_PROMPTS_JSONL="$EVAL_ENV_DIR/prompts.jsonl"
else
  EVAL_PROMPTS_JSONL="${EVAL_PROMPTS_JSONL:-$PROMPTS_JSONL}"
fi
EVAL_OUT="${EVAL_OUT:-$ROOT/eval_mlp_positive}"
EVAL_GENERATIONS_JSONL="${EVAL_GENERATIONS_JSONL:-$EVAL_OUT/generations.jsonl}"
EVAL_SCORED_GENERATIONS="${EVAL_SCORED_GENERATIONS:-$EVAL_OUT/scored_generations.jsonl}"
MASTER_LOG="$ROOT/master.log"
STATUS="$ROOT/status.tsv"

mkdir -p "$ROOT"
if [[ -f LOCAL_HOME/anaconda3/etc/profile.d/conda.sh && -z "${SKIP_CONDA:-}" ]]; then
  # shellcheck disable=SC1091
  source LOCAL_HOME/anaconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV:-screscomp}"
fi
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

if [[ "$EVAL_ONLY" == "1" && -f "$STATUS" ]]; then
  echo -e "$(date -Is)\tresume_eval_only\tstarting" >> "$STATUS"
else
  {
    echo -e "time\tstage\tstatus"
    echo -e "$(date -Is)\tinit\tstarting"
  } > "$STATUS"
fi
echo "[$(date -Is)] root=$ROOT model=$MODEL event=$EVENT train_prompts=$TRAIN_PROMPTS alpha=$ALPHA_SWEEP fixed_loss=$PREFERENCE_LOSS_MODE discovery=$DISCOVERY_SCAN_MODE discovery_apply=$DISCOVERY_APPLY_MODE train_apply=$TRAIN_APPLY_MODE train_causal=$TRAIN_CAUSAL_MASK eval_apply=$EVAL_APPLY_MODE" | tee -a "$MASTER_LOG"

run_eval_stage() {
  if [[ -z "$EVAL_ACTUATOR" && -z "$EVAL_HEAD_ACTUATOR" ]]; then
    echo "Missing eval actuator: set EVAL_ACTUATOR and/or EVAL_HEAD_ACTUATOR" >&2
    exit 2
  fi
  if [[ -n "$EVAL_ACTUATOR" && ! -f "$EVAL_ACTUATOR" ]]; then
    echo "Missing fixed actuator: $EVAL_ACTUATOR" >&2
    exit 2
  fi
  if [[ -n "$EVAL_HEAD_ACTUATOR" && ! -f "$EVAL_HEAD_ACTUATOR" ]]; then
    echo "Missing head actuator: $EVAL_HEAD_ACTUATOR" >&2
    exit 2
  fi
  if [[ ! -f "$EVAL_PROMPTS_JSONL" ]]; then
    echo "Missing eval prompts: $EVAL_PROMPTS_JSONL" >&2
    exit 2
  fi
  echo -e "$(date -Is)\teval_mlp_positive_generate\trunning" >> "$STATUS"
  mkdir -p "$EVAL_OUT"
  eval_gen_args=(
    python -m screscomp.cli.run_imdb_sentiment_actuator_generation
    --model "$MODEL"
    --prompts-jsonl "$EVAL_PROMPTS_JSONL"
    --out-jsonl "$EVAL_GENERATIONS_JSONL"
    --control-name "$EVAL_CONTROL_NAME"
    --alpha-sweep "$EVAL_ALPHA_SWEEP"
    --generation-apply-mode "$EVAL_APPLY_MODE"
    --split "$EVAL_SPLIT"
    --start "$EVAL_START"
    --samples-per-prompt "$EVAL_SAMPLES_PER_PROMPT"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --stop-strings "$STOP_STRINGS"
    --temperature "$EVAL_TEMPERATURE"
    --top-p "$EVAL_TOP_P"
    --top-k "$EVAL_TOP_K"
    --seed "$SEED"
    --torch-dtype "$TORCH_DTYPE"
    --device "$DEVICE"
  )
  if [[ -n "$EVAL_ACTUATOR" ]]; then
    eval_gen_args+=(--actuator "$EVAL_ACTUATOR")
  fi
  if [[ -n "$EVAL_HEAD_ACTUATOR" ]]; then
    eval_gen_args+=(--head-actuator "$EVAL_HEAD_ACTUATOR")
  fi
  if [[ -n "$EVAL_COMPONENT_APPLY_MODE" ]]; then
    eval_gen_args+=(--component-apply-mode "$EVAL_COMPONENT_APPLY_MODE")
  fi
  if [[ -n "$EVAL_HEAD_APPLY_MODE" ]]; then
    eval_gen_args+=(--head-apply-mode "$EVAL_HEAD_APPLY_MODE")
  fi
  if [[ -n "$EVAL_MAX_ROWS" ]]; then
    eval_gen_args+=(--max-rows "$EVAL_MAX_ROWS")
  fi
  if [[ "$EVAL_DO_SAMPLE" == "0" ]]; then
    eval_gen_args+=(--no-do-sample)
  fi
  if [[ "$EVAL_SAME_SEED_ACROSS_ALPHA" == "1" ]]; then
    eval_gen_args+=(--same-seed-across-alpha)
  fi
  CUDA_VISIBLE_DEVICES="$GPU" "${eval_gen_args[@]}" | tee -a "$MASTER_LOG"
  echo -e "$(date -Is)\teval_mlp_positive_generate\tdone" >> "$STATUS"

  echo -e "$(date -Is)\teval_mlp_positive_score\trunning" >> "$STATUS"
  CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.score_imdb_sentiment_generations \
    --input "$EVAL_GENERATIONS_JSONL" \
    --out-jsonl "$EVAL_SCORED_GENERATIONS" \
    --out-csv "$EVAL_OUT/scored_generations.csv" \
    --summary-csv "$EVAL_OUT/score_summary.csv" \
    --scorer-model "$SCORER_MODEL" \
    --score-text "$EVAL_SCORE_TEXT" \
    --batch-size "$SCORE_BATCH_SIZE" \
    --device "$SCORER_DEVICE" \
    | tee -a "$MASTER_LOG"
  echo -e "$(date -Is)\teval_mlp_positive_score\tdone" >> "$STATUS"
}

PAIRS_PREPARED=0
prepare_pairs_stage() {
  if [[ "$PAIRS_PREPARED" == "1" ]]; then
    return
  fi
  if [[ ! -f "$SCORED_GENERATIONS" ]]; then
    if [[ -f "$PAIRS_DIR/pairs.csv" ]]; then
      echo "[$(date -Is)] reuse pairs=$PAIRS_DIR/pairs.csv" | tee -a "$MASTER_LOG"
      PAIRS_PREPARED=1
      return
    fi
    echo "No scored generations found. Set SCORED_GENERATIONS=/path/to/scored.jsonl or use RUN_GENERATE=1 RUN_SCORE=1." >&2
    exit 2
  fi

  echo -e "$(date -Is)\tprepare_pairs\trunning" >> "$STATUS"
  pair_args=(
    python -m screscomp.cli.prepare_imdb_sentiment_pairs
    --input "$SCORED_GENERATIONS"
    --out-dir "$PAIRS_DIR"
    --event "$EVENT"
    --score-field "$PAIR_SCORE_FIELD"
    --min-score-margin "$MIN_SCORE_MARGIN"
  )
  if [[ "$MAX_PAIRS_PER_PROMPT" != "0" ]]; then
    pair_args+=(--max-pairs-per-prompt "$MAX_PAIRS_PER_PROMPT")
  fi
  if [[ "$PAIR_MAX_PROMPTS_PER_SPLIT" != "0" ]]; then
    pair_args+=(--max-prompts-per-split "$PAIR_MAX_PROMPTS_PER_SPLIT")
  fi
  "${pair_args[@]}" | tee -a "$MASTER_LOG"
  PAIRS_PREPARED=1
  echo -e "$(date -Is)\tprepare_pairs\tdone" >> "$STATUS"
}

if [[ "$EVAL_ONLY" == "1" ]]; then
  run_eval_stage
  echo "[$(date -Is)] eval_only done" | tee -a "$MASTER_LOG"
  echo -e "$(date -Is)\tall\tdone" >> "$STATUS"
  exit 0
fi

if [[ "$REBUILD_ENV" == "1" || ! -f "$PROMPTS_JSONL" ]]; then
  echo -e "$(date -Is)\tprepare_env\trunning" >> "$STATUS"
  env_args=(
    python -m screscomp.cli.prepare_imdb_sentiment_env
    --out-dir "$ENV_DIR"
    --event "$EVENT"
    --train-rows "$TRAIN_PROMPTS"
    --val-rows "$VAL_PROMPTS"
    --eval-rows "$EVAL_PROMPTS"
    --prefix-mode "$PREFIX_MODE"
    --prefix-token-min "$PREFIX_TOKEN_MIN"
    --prefix-token-max "$PREFIX_TOKEN_MAX"
    --completions-per-prefix "$COMPLETIONS_PER_PREFIX"
    --scorer-model "$SCORER_MODEL"
    --prompt-template "$PROMPT_TEMPLATE"
    --target-sentiment "$TARGET_SENTIMENT"
    --reward-score-field "$REWARD_SCORE_FIELD"
    --reference-model "$MODEL"
    --seed "$SEED"
  )
  if [[ "$SHUFFLE_PROMPTS" == "1" ]]; then
    env_args+=(--shuffle)
  fi
  if [[ -n "$TOKENIZER" ]]; then
    env_args+=(--tokenizer "$TOKENIZER")
  fi
  if [[ -n "$SOURCE_INPUT" ]]; then
    env_args+=(--input "$SOURCE_INPUT")
  else
    env_args+=(--hf-dataset "$HF_DATASET" --hf-split "$HF_SPLIT")
    if [[ -n "$HF_CONFIG" ]]; then
      env_args+=(--hf-config "$HF_CONFIG")
    fi
  fi
  "${env_args[@]}" | tee -a "$MASTER_LOG"
  echo -e "$(date -Is)\tprepare_env\tdone" >> "$STATUS"
else
  echo "[$(date -Is)] reuse env prompts=$PROMPTS_JSONL" | tee -a "$MASTER_LOG"
fi

if [[ "$USE_SEPARATE_EVAL_ENV" == "1" && ( "$PREPARE_EVAL_ENV" == "1" || "$RUN_EVAL" == "1" || "$STOP_AFTER_ENV" == "1" ) ]]; then
  if [[ "$REBUILD_ENV" == "1" || ! -f "$EVAL_PROMPTS_JSONL" ]]; then
    echo -e "$(date -Is)\tprepare_eval_env\trunning" >> "$STATUS"
    eval_env_args=(
      python -m screscomp.cli.prepare_imdb_sentiment_env
      --out-dir "$EVAL_ENV_DIR"
      --event "$EVENT"
      --train-rows 0
      --val-rows 0
      --eval-rows "$FINAL_EVAL_PROMPTS"
      --prefix-mode "$PREFIX_MODE"
      --prefix-token-min "$PREFIX_TOKEN_MIN"
      --prefix-token-max "$PREFIX_TOKEN_MAX"
      --completions-per-prefix "$COMPLETIONS_PER_PREFIX"
      --scorer-model "$SCORER_MODEL"
      --prompt-template "$PROMPT_TEMPLATE"
      --target-sentiment "$TARGET_SENTIMENT"
      --reward-score-field "$REWARD_SCORE_FIELD"
      --reference-model "$MODEL"
      --seed "$SEED"
      --hf-dataset "$EVAL_HF_DATASET"
      --hf-split "$EVAL_HF_SPLIT"
    )
    if [[ "$SHUFFLE_PROMPTS" == "1" ]]; then
      eval_env_args+=(--shuffle)
    fi
    if [[ -n "$TOKENIZER" ]]; then
      eval_env_args+=(--tokenizer "$TOKENIZER")
    fi
    if [[ -n "$EVAL_HF_CONFIG" ]]; then
      eval_env_args+=(--hf-config "$EVAL_HF_CONFIG")
    fi
    "${eval_env_args[@]}" | tee -a "$MASTER_LOG"
    echo -e "$(date -Is)\tprepare_eval_env\tdone" >> "$STATUS"
  else
    echo "[$(date -Is)] reuse eval env prompts=$EVAL_PROMPTS_JSONL" | tee -a "$MASTER_LOG"
  fi
fi

if [[ "$STOP_AFTER_ENV" == "1" ]]; then
  echo -e "$(date -Is)\tall\tstopped_after_env" >> "$STATUS"
  exit 0
fi

if [[ "$RUN_GENERATE" == "1" ]]; then
  echo -e "$(date -Is)\tgenerate_start_policy\trunning" >> "$STATUS"
  gen_args=(
    python -m screscomp.cli.generate_imdb_sentiment_completions
    --model "$MODEL"
    --prompts-jsonl "$PROMPTS_JSONL"
    --out-jsonl "$GENERATIONS_JSONL"
    --split "$GENERATION_SPLIT"
    --start "$GENERATION_START"
    --completions-per-prefix "$COMPLETIONS_PER_PREFIX"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --stop-strings "$STOP_STRINGS"
    --temperature "$TEMPERATURE"
    --top-p "$TOP_P"
    --top-k "$TOP_K"
    --seed "$SEED"
    --torch-dtype "$TORCH_DTYPE"
    --device "$DEVICE"
  )
  if [[ -n "$GENERATION_MAX_ROWS" ]]; then
    gen_args+=(--max-rows "$GENERATION_MAX_ROWS")
  fi
  if [[ "$GENERATION_DO_SAMPLE" == "0" ]]; then
    gen_args+=(--no-do-sample)
  fi
  CUDA_VISIBLE_DEVICES="$GPU" "${gen_args[@]}" | tee -a "$MASTER_LOG"
  echo -e "$(date -Is)\tgenerate_start_policy\tdone" >> "$STATUS"
fi

if [[ "$RUN_SCORE" == "1" ]]; then
  if [[ ! -f "$GENERATIONS_JSONL" ]]; then
    echo "Missing generations: set GENERATIONS_JSONL or RUN_GENERATE=1" >&2
    exit 2
  fi
  echo -e "$(date -Is)\tscore_generations\trunning" >> "$STATUS"
  CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.score_imdb_sentiment_generations \
    --input "$GENERATIONS_JSONL" \
    --out-jsonl "$SCORED_GENERATIONS" \
    --out-csv "$START_DIR/scored_generations.csv" \
    --scorer-model "$SCORER_MODEL" \
    --score-text "$SCORE_TEXT" \
    --batch-size "$SCORE_BATCH_SIZE" \
    --device "$SCORER_DEVICE" \
    | tee -a "$MASTER_LOG"
  echo -e "$(date -Is)\tscore_generations\tdone" >> "$STATUS"
fi

if [[ "$DISCOVERY_SCAN_MODE" == "margin" || "$STOP_AFTER_PAIRS" == "1" ]]; then
  prepare_pairs_stage
fi

if [[ "$STOP_AFTER_PAIRS" == "1" ]]; then
  echo -e "$(date -Is)\tall\tstopped_after_pairs" >> "$STATUS"
  exit 0
fi

echo -e "$(date -Is)\tscan_components\trunning" >> "$STATUS"
mkdir -p "$DISCOVERY_DIR/component_scan" "$SELECT_DIR"
scan_args=(
  python -m screscomp.cli.cecm_scan_component_contributions
  --model "$MODEL"
  --event "$EVENT"
  --scan-mode "$DISCOVERY_SCAN_MODE"
  --split "$DISCOVERY_SPLIT"
  --start 0
  --max-rows "$DISCOVERY_ROWS"
  --component-types attn,mlp
  --apply-mode "$DISCOVERY_APPLY_MODE"
  --min-abs-delta 0.0
  --min-sign-consistency 0.50
  --allow-ci-cross-zero
  --max-components-per-direction 16
  --torch-dtype "$TORCH_DTYPE"
  --device "$DEVICE"
  --out-dir "$DISCOVERY_DIR/component_scan"
)
if [[ "$DISCOVERY_SCAN_MODE" == "margin" ]]; then
  scan_args+=(
    --pairs-csv "$PAIRS_DIR/pairs.csv"
    --score-mode "$SCORE_MODE"
    --max-aliases-per-side "$MAX_ALIASES_PER_SIDE"
  )
elif [[ "$DISCOVERY_SCAN_MODE" == "rollout" ]]; then
  scan_args+=(
    --prompts-jsonl "$PROMPTS_JSONL"
    --samples-per-prompt "$DISCOVERY_SAMPLES_PER_PROMPT"
    --generation-batch-size "$DISCOVERY_GENERATION_BATCH_SIZE"
    --max-new-tokens "$DISCOVERY_MAX_NEW_TOKENS"
    --stop-strings "$STOP_STRINGS"
    --temperature "$DISCOVERY_TEMPERATURE"
    --top-p "$DISCOVERY_TOP_P"
    --top-k "$DISCOVERY_TOP_K"
    --seed "$SEED"
    --scorer-model "$SCORER_MODEL"
    --target-label "$DISCOVERY_TARGET_LABEL"
    --source-label "$DISCOVERY_SOURCE_LABEL"
    --score-text "$DISCOVERY_SCORE_TEXT"
    --scorer-batch-size "$DISCOVERY_SCORER_BATCH_SIZE"
    --scorer-max-length "$DISCOVERY_SCORER_MAX_LENGTH"
    --scorer-device "$SCORER_DEVICE"
  )
  if [[ "$DISCOVERY_DO_SAMPLE" == "0" ]]; then
    scan_args+=(--no-do-sample)
  fi
else
  echo "Unsupported DISCOVERY_SCAN_MODE=$DISCOVERY_SCAN_MODE; expected margin or rollout." >&2
  exit 2
fi
CUDA_VISIBLE_DEVICES="$GPU" "${scan_args[@]}" > "$DISCOVERY_DIR/component_scan.log" 2>&1
echo -e "$(date -Is)\tscan_components\tdone" >> "$STATUS"

if [[ "$STOP_AFTER_SCAN" == "1" ]]; then
  echo -e "$(date -Is)\tall\tstopped_after_scan" >> "$STATUS"
  exit 0
fi

echo -e "$(date -Is)\tselect_components\trunning" >> "$STATUS"
select_args=(
  python -m screscomp.cli.cecm_select_margin_components
  --component-screen-csv "$DISCOVERY_DIR/component_scan/component_screen.csv"
  --event "$EVENT"
  --topk-mlp "$DISCOVERY_TOPK_MLP"
  --topk-negative-mlp "$DISCOVERY_TOPK_NEGATIVE_MLP"
  --topk-attn-layers "$DISCOVERY_TOPK_ATTN_LAYERS"
  --mlp-policy "$MLP_POLICY"
  --min-sign-consistency "$MIN_SIGN_CONSISTENCY"
  --min-mlp-abs-mean-delta "$MIN_MLP_ABS_MEAN_DELTA"
  --min-attn-abs-mean-delta "$MIN_ATTN_ABS_MEAN_DELTA"
  --min-directional-transition-rate "$MIN_DIRECTIONAL_TRANSITION_RATE"
  --min-ablated-format-ok "$MIN_ABLATED_FORMAT_OK"
  --max-format-collapse-rate "$MAX_FORMAT_COLLAPSE_RATE"
  --min-abs-mean-target-drop "$MIN_ABS_MEAN_TARGET_DROP"
  --min-abs-mean-source-rise "$MIN_ABS_MEAN_SOURCE_RISE"
  --exclude-mlp-layers "$EXCLUDE_MLP_LAYERS"
  --out-dir "$SELECT_DIR"
)
if [[ "$MLP_REQUIRE_DIRECTIONAL_CI" == "1" ]]; then
  select_args+=(--mlp-require-directional-ci)
fi
if [[ -n "$EXCLUDE_ATTN_LAYERS" ]]; then
  select_args+=(--exclude-attn-layers "$EXCLUDE_ATTN_LAYERS")
fi
if [[ "$ALLOW_EMPTY_ATTN" == "1" ]]; then
  select_args+=(--allow-empty-attn)
fi
"${select_args[@]}" | tee -a "$MASTER_LOG"
ATTN_LAYERS="$(tr -d '\r\n' < "$SELECT_DIR/attention_layers.txt")"
echo "[$(date -Is)] selected mlp_policy=$MLP_POLICY train_components=$TRAIN_COMPONENTS_CSV positive_mlp=$SELECT_DIR/mlp_positive_components.csv negative_mlp=$SELECT_DIR/mlp_negative_components.csv attn_layers=$ATTN_LAYERS" | tee -a "$MASTER_LOG"
echo -e "$(date -Is)\tselect_components\tdone" >> "$STATUS"

if [[ "$STOP_AFTER_SELECT" == "1" ]]; then
  echo -e "$(date -Is)\tall\tstopped_after_select" >> "$STATUS"
  exit 0
fi

prepare_pairs_stage

echo -e "$(date -Is)\ttrain_mlp_positive\trunning" >> "$STATUS"
mkdir -p "$MLP_OUT"
train_fixed_extra_args=()
if [[ "$TRAIN_CAUSAL_MASK" == "1" ]]; then
  train_fixed_extra_args+=(--causal-train-mask)
fi
CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.cecm_train_fixed_actuator \
  --model "$MODEL" \
  --pairs-csv "$PAIRS_DIR/pairs.csv" \
  --components-csv "$TRAIN_COMPONENTS_CSV" \
  --event "$EVENT" \
  --train-split train \
  --val-split val \
  --max-train-rows "$MAX_TRAIN_PAIRS" \
  --max-val-rows "$MAX_VAL_PAIRS" \
  --epochs "$MLP_EPOCHS" \
  --lr "$LR" \
  --lambda-norm "$LAMBDA_NORM" \
  --alpha-train "$ALPHA_TRAIN" \
  --preference-loss-mode "$PREFERENCE_LOSS_MODE" \
  --dpo-beta "$DPO_BETA" \
  --state-margin-weight "$STATE_MARGIN_WEIGHT" \
  --gain-weight "$GAIN_WEIGHT" \
  --target-margin "$TARGET_MARGIN" \
  --target-gain "$TARGET_GAIN" \
  --apply-mode "$TRAIN_APPLY_MODE" \
  "${train_fixed_extra_args[@]}" \
  --score-mode "$SCORE_MODE" \
  --alpha-sweep "$ALPHA_SWEEP" \
  --max-aliases-per-side "$MAX_ALIASES_PER_SIDE" \
  --empty-cache-every "$EMPTY_CACHE_EVERY" \
  --torch-dtype "$TORCH_DTYPE" \
  --device "$DEVICE" \
  --out-dir "$MLP_OUT" \
  > "$MLP_OUT/train.log" 2>&1
echo -e "$(date -Is)\ttrain_mlp_positive\tdone" >> "$STATUS"

if [[ "$RUN_EVAL" == "1" ]]; then
  run_eval_stage
fi

if [[ "$RUN_HEADS" == "1" && -n "$ATTN_LAYERS" ]]; then
  echo -e "$(date -Is)\tscan_heads\trunning" >> "$STATUS"
  mkdir -p "$HEAD_SCAN_OUT"
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
    --score-apply-mode "$HEAD_SCAN_APPLY_MODE"
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
    train_head_extra_args=()
    if [[ "$HEAD_TRAIN_CAUSAL_MASK" == "1" ]]; then
      train_head_extra_args+=(--causal-train-mask)
    fi
    CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.cecm_train_attention_head_actuator \
      --model "$MODEL" \
      --pairs-csv "$PAIRS_DIR/pairs.csv" \
      --event "$EVENT" \
      --heads "$SELECTED_HEADS" \
      --train-split train \
      --val-split val \
      --max-train-rows "$MAX_TRAIN_PAIRS" \
      --max-val-rows "$MAX_VAL_PAIRS" \
      --epochs "$HEAD_EPOCHS" \
      --lr "$LR" \
      --lambda-norm "$LAMBDA_NORM" \
      --alpha-train "$ALPHA_TRAIN" \
      --state-margin-weight "$STATE_MARGIN_WEIGHT" \
      --gain-weight "$GAIN_WEIGHT" \
      --target-margin "$TARGET_MARGIN" \
      --target-gain "$TARGET_GAIN" \
      --apply-mode "$HEAD_TRAIN_APPLY_MODE" \
      "${train_head_extra_args[@]}" \
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
find "$ROOT" -maxdepth 3 \( -name environment_manifest.json -o -name pair_build_manifest.json -o -name component_selection_manifest.json -o -name alpha_summary.csv -o -name train_history.csv -o -name generation_manifest.json -o -name score_manifest.json -o -name score_summary.csv \) -print | sort
