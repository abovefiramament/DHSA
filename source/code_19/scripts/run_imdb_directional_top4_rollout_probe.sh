#!/usr/bin/env bash
set -euo pipefail

# Directional-gross-mass IMDb probe:
#   1) read component_directionality.csv from a rollout scan
#   2) select mlp/attn x positive/negative top-k components under health audit
#   3) train a fixed actuator for each group on the existing IMDb pairs
#   4) run held-out rollout eval and score sentiment

cd LOCAL_HOME/RPEC/projects/screscomp

GPU="${GPU:-0}"
MODEL="${MODEL:-edbeeching/gpt2-large-imdb}"
EVENT="${EVENT:-imdb_positive_sentiment}"

SCAN_ROOT="${SCAN_ROOT:-runs/imdb_dpo_prompt_positive_rollout_scan_20260605_115624}"
BASE_ROOT="${BASE_ROOT:-runs/imdb_sentiment_smoke_clean_20260604_155603}"
OUT_ROOT="${OUT_ROOT:-runs/imdb_directional_top4_probe_$(date +%Y%m%d_%H%M%S)}"

TOPK_MLP="${TOPK_MLP:-4}"
TOPK_ATTN="${TOPK_ATTN:-4}"
MIN_ABLATED_FORMAT_OK="${MIN_ABLATED_FORMAT_OK:-0.90}"
MAX_FORMAT_COLLAPSE_RATE="${MAX_FORMAT_COLLAPSE_RATE:-0.10}"
MIN_HEALTHY_PAIR_RATE="${MIN_HEALTHY_PAIR_RATE:-0.90}"
MIN_HEALTHY_N="${MIN_HEALTHY_N:-1}"
REQUIRE_DIRECTION_MATCH="${REQUIRE_DIRECTION_MATCH:-0}"
DISALLOW_CROSS_GROUP_OVERLAP="${DISALLOW_CROSS_GROUP_OVERLAP:-0}"

EVAL_GROUPS="${EVAL_GROUPS:-mlp_positive,mlp_negative,attn_positive,attn_negative}"

PAIRS_CSV="${PAIRS_CSV:-$BASE_ROOT/pairs/pairs.csv}"
EVAL_PROMPTS_JSONL="${EVAL_PROMPTS_JSONL:-$BASE_ROOT/eval_env/prompts.jsonl}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
VAL_SPLIT="${VAL_SPLIT:-val}"
EVAL_SPLIT="${EVAL_SPLIT:-eval}"

MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-240}"
MAX_VAL_ROWS="${MAX_VAL_ROWS:-60}"
EPOCHS="${EPOCHS:-2}"
LR="${LR:-0.05}"
LAMBDA_NORM="${LAMBDA_NORM:-1e-4}"
ALPHA_TRAIN="${ALPHA_TRAIN:-1.0}"
TRAIN_APPLY_MODE="${TRAIN_APPLY_MODE:-all}"
TRAIN_CAUSAL_MASK="${TRAIN_CAUSAL_MASK:-1}"
PREFERENCE_LOSS_MODE="${PREFERENCE_LOSS_MODE:-dpo}"
DPO_BETA="${DPO_BETA:-1.0}"
STATE_MARGIN_WEIGHT="${STATE_MARGIN_WEIGHT:-0.0}"
GAIN_WEIGHT="${GAIN_WEIGHT:-1.0}"
TARGET_MARGIN="${TARGET_MARGIN:-0.0}"
TARGET_GAIN="${TARGET_GAIN:-0.0}"
SCORE_MODE="${SCORE_MODE:-avglogp}"
ALPHA_SWEEP="${ALPHA_SWEEP:-0,0.5,1.0}"

EVAL_MAX_ROWS="${EVAL_MAX_ROWS:-64}"
EVAL_SAMPLES_PER_PROMPT="${EVAL_SAMPLES_PER_PROMPT:-1}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-32}"
EVAL_APPLY_MODE="${EVAL_APPLY_MODE:-all}"
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

TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "$OUT_ROOT"

if [[ -f LOCAL_HOME/anaconda3/etc/profile.d/conda.sh && -z "${SKIP_CONDA:-}" ]]; then
  # shellcheck disable=SC1091
  source LOCAL_HOME/anaconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV:-screscomp}"
fi
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

SELECT_DIR="$OUT_ROOT/selected"
STATUS="$OUT_ROOT/status.tsv"
MASTER_LOG="$OUT_ROOT/master.log"

{
  echo -e "time\tstage\tstatus"
  echo -e "$(date -Is)\tinit\tstarting"
} > "$STATUS"

echo "[$(date -Is)] directional probe scan_root=$SCAN_ROOT base_root=$BASE_ROOT out_root=$OUT_ROOT groups=$EVAL_GROUPS" | tee -a "$MASTER_LOG"

select_args=(
  python -m screscomp.cli.cecm_select_rollout_directional_components
  --component-directionality-csv "$SCAN_ROOT/discovery/directionality_scan/component_directionality.csv"
  --event "$EVENT"
  --topk-mlp "$TOPK_MLP"
  --topk-attn "$TOPK_ATTN"
  --min-ablated-format-ok "$MIN_ABLATED_FORMAT_OK"
  --max-format-collapse-rate "$MAX_FORMAT_COLLAPSE_RATE"
  --min-healthy-pair-rate "$MIN_HEALTHY_PAIR_RATE"
  --min-healthy-n "$MIN_HEALTHY_N"
  --out-dir "$SELECT_DIR"
)
if [[ "$REQUIRE_DIRECTION_MATCH" == "1" ]]; then
  select_args+=(--require-direction-match)
fi
if [[ "$DISALLOW_CROSS_GROUP_OVERLAP" == "1" ]]; then
  select_args+=(--disallow-cross-group-overlap)
fi
echo -e "$(date -Is)\tselect_directional_components\trunning" >> "$STATUS"
"${select_args[@]}" | tee -a "$MASTER_LOG"
echo -e "$(date -Is)\tselect_directional_components\tdone" >> "$STATUS"

IFS=',' read -r -a GROUP_LIST <<< "$EVAL_GROUPS"
for raw_group in "${GROUP_LIST[@]}"; do
  group="$(echo "$raw_group" | xargs)"
  [[ -z "$group" ]] && continue
  COMPONENTS_CSV="$SELECT_DIR/${group}_components.csv"
  if [[ ! -s "$COMPONENTS_CSV" ]]; then
    echo "[$(date -Is)] skip group=$group reason=empty_components_csv path=$COMPONENTS_CSV" | tee -a "$MASTER_LOG"
    echo -e "$(date -Is)\t${group}\tskipped_empty" >> "$STATUS"
    continue
  fi

  TRAIN_OUT="$OUT_ROOT/train_${group}"
  EVAL_OUT="$OUT_ROOT/eval_${group}"
  EVAL_GENERATIONS="$EVAL_OUT/generations.jsonl"
  EVAL_SCORED="$EVAL_OUT/scored_generations.jsonl"
  mkdir -p "$TRAIN_OUT" "$EVAL_OUT"

  echo -e "$(date -Is)\ttrain_${group}\trunning" >> "$STATUS"
  train_args=(
    python -m screscomp.cli.cecm_train_fixed_actuator
    --model "$MODEL"
    --pairs-csv "$PAIRS_CSV"
    --components-csv "$COMPONENTS_CSV"
    --event "$EVENT"
    --train-split "$TRAIN_SPLIT"
    --val-split "$VAL_SPLIT"
    --max-train-rows "$MAX_TRAIN_ROWS"
    --max-val-rows "$MAX_VAL_ROWS"
    --epochs "$EPOCHS"
    --lr "$LR"
    --lambda-norm "$LAMBDA_NORM"
    --alpha-train "$ALPHA_TRAIN"
    --preference-loss-mode "$PREFERENCE_LOSS_MODE"
    --dpo-beta "$DPO_BETA"
    --state-margin-weight "$STATE_MARGIN_WEIGHT"
    --gain-weight "$GAIN_WEIGHT"
    --target-margin "$TARGET_MARGIN"
    --target-gain "$TARGET_GAIN"
    --apply-mode "$TRAIN_APPLY_MODE"
    --score-mode "$SCORE_MODE"
    --alpha-sweep "$ALPHA_SWEEP"
    --torch-dtype "$TORCH_DTYPE"
    --device "$DEVICE"
    --seed "$SEED"
    --out-dir "$TRAIN_OUT"
  )
  if [[ "$TRAIN_CAUSAL_MASK" == "1" ]]; then
    train_args+=(--causal-train-mask)
  fi
  CUDA_VISIBLE_DEVICES="$GPU" "${train_args[@]}" > "$TRAIN_OUT/train.log" 2>&1
  echo -e "$(date -Is)\ttrain_${group}\tdone" >> "$STATUS"

  echo -e "$(date -Is)\teval_${group}_generate\trunning" >> "$STATUS"
  eval_args=(
    python -m screscomp.cli.run_imdb_sentiment_actuator_generation
    --model "$MODEL"
    --prompts-jsonl "$EVAL_PROMPTS_JSONL"
    --out-jsonl "$EVAL_GENERATIONS"
    --actuator "$TRAIN_OUT/fixed_actuator.pt"
    --control-name "$group"
    --alpha-sweep "$ALPHA_SWEEP"
    --generation-apply-mode "$EVAL_APPLY_MODE"
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
    eval_args+=(--no-do-sample)
  fi
  if [[ "$SAME_SEED_ACROSS_ALPHA" == "1" ]]; then
    eval_args+=(--same-seed-across-alpha)
  fi
  CUDA_VISIBLE_DEVICES="$GPU" "${eval_args[@]}" > "$EVAL_OUT/generate.log" 2>&1
  echo -e "$(date -Is)\teval_${group}_generate\tdone" >> "$STATUS"

  echo -e "$(date -Is)\teval_${group}_score\trunning" >> "$STATUS"
  CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.score_imdb_sentiment_generations \
    --input "$EVAL_GENERATIONS" \
    --out-jsonl "$EVAL_SCORED" \
    --score-text "$SCORE_TEXT" \
    --batch-size "$SCORE_BATCH_SIZE" \
    --scorer-model "$SCORER_MODEL" \
    --scorer-max-length "$SCORER_MAX_LENGTH" \
    --device "$SCORER_DEVICE" \
    > "$EVAL_OUT/score.log" 2>&1
  echo -e "$(date -Is)\teval_${group}_score\tdone" >> "$STATUS"
done

echo -e "$(date -Is)\tsummarize\trunning" >> "$STATUS"
python -m screscomp.cli.summarize_imdb_sentiment_matrix \
  --root "$OUT_ROOT" \
  --out-csv "$OUT_ROOT/matrix_score_summary.csv" \
  | tee -a "$MASTER_LOG"
echo -e "$(date -Is)\tsummarize\tdone" >> "$STATUS"
echo -e "$(date -Is)\tall\tdone" >> "$STATUS"

echo "[$(date -Is)] directional rollout probe done out=$OUT_ROOT" | tee -a "$MASTER_LOG"
