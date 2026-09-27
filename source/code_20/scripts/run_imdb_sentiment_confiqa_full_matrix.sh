#!/usr/bin/env bash
set -euo pipefail

# ConFiQA-style full IMDb sentiment-control with staged search:
#   Phase 1: full-size sampling + component discovery + training (MLP + heads)
#   Phase 2: MLP internal timing selection (only previously validated timings)
#   Phase 3: Head internal timing selection
#   Phase 4: Full combined with best MLP timing + best head timing, small alpha sweep
#   Phase 5: Final summary

cd LOCAL_HOME/RPEC/projects/screscomp

GPU="${GPU:-0}"
ROOT="${ROOT:-runs/imdb_sentiment_confiqa_full_$(date +%Y%m%d_%H%M%S)}"
CONDA_ENV="${CONDA_ENV:-screscomp}"

# --- Phase control ---
RUN_BASE="${RUN_BASE:-1}"
RUN_MLP_SELECT="${RUN_MLP_SELECT:-1}"
RUN_HEAD_SELECT="${RUN_HEAD_SELECT:-1}"
RUN_FULL_COMBINED="${RUN_FULL_COMBINED:-1}"
BASE_RUN_GENERATE="${BASE_RUN_GENERATE:-1}"
BASE_RUN_SCORE="${BASE_RUN_SCORE:-1}"
BASE_PREPARE_EVAL_ENV="${BASE_PREPARE_EVAL_ENV:-1}"

# --- Model & data ---
MODEL="${MODEL:-edbeeching/gpt2-large-imdb}"
SCORER_MODEL="${SCORER_MODEL:-siebert/sentiment-roberta-large-english}"
HF_DATASET="${HF_DATASET:-stanfordnlp/imdb}"
HF_SPLIT="${HF_SPLIT:-train}"
EVAL_HF_SPLIT="${EVAL_HF_SPLIT:-test}"

TRAIN_PROMPTS="${TRAIN_PROMPTS:-24000}"
VAL_PROMPTS="${VAL_PROMPTS:-1000}"
FINAL_EVAL_PROMPTS="${FINAL_EVAL_PROMPTS:-1000}"
COMPLETIONS_PER_PREFIX="${COMPLETIONS_PER_PREFIX:-4}"
EVAL_SAMPLES_PER_PROMPT="${EVAL_SAMPLES_PER_PROMPT:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-504}"

# --- Discovery: larger pool, strict thresholds ---
DISCOVERY_ROWS="${DISCOVERY_ROWS:-400}"
DISCOVERY_TOPK_MLP="${DISCOVERY_TOPK_MLP:-4}"
DISCOVERY_TOPK_NEGATIVE_MLP="${DISCOVERY_TOPK_NEGATIVE_MLP:-4}"
DISCOVERY_TOPK_ATTN_LAYERS="${DISCOVERY_TOPK_ATTN_LAYERS:-4}"
MIN_SIGN_CONSISTENCY="${MIN_SIGN_CONSISTENCY:-0.55}"
MLP_REQUIRE_DIRECTIONAL_CI="${MLP_REQUIRE_DIRECTIONAL_CI:-1}"
MIN_MLP_ABS_MEAN_DELTA="${MIN_MLP_ABS_MEAN_DELTA:-0.3}"
MIN_ATTN_ABS_MEAN_DELTA="${MIN_ATTN_ABS_MEAN_DELTA:-0.3}"
HEAD_SCAN_ROWS="${HEAD_SCAN_ROWS:-400}"
HEAD_TOPK="${HEAD_TOPK:-8}"
RUN_HEADS="${RUN_HEADS:-1}"

# --- Training ---
MAX_TRAIN_PAIRS="${MAX_TRAIN_PAIRS:-4000}"
MAX_VAL_PAIRS="${MAX_VAL_PAIRS:-500}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-0.05}"
LAMBDA_NORM="${LAMBDA_NORM:-1e-4}"
ALPHA_SWEEP="${ALPHA_SWEEP:-0,0.01,0.02,0.05,0.1,0.2,0.3,0.5,0.75,1.0}"
EVAL_ALPHA_SWEEP="${EVAL_ALPHA_SWEEP:-$ALPHA_SWEEP}"

TRAIN_APPLY_MODE="${TRAIN_APPLY_MODE:-decision_tokens}"
HEAD_TRAIN_APPLY_MODE="${HEAD_TRAIN_APPLY_MODE:-$TRAIN_APPLY_MODE}"

# --- Staged timing search: only previously validated timings ---
# MLP: prefill (ConFiQA+GSM8K validated), decision_tokens (train mode)
MLP_EVAL_APPLY_MODES="${MLP_EVAL_APPLY_MODES:-prefill,decision_tokens}"
# Head: all (ConFiQA+GSM8K validated), first_decode (GSM8K nocot validated)
HEAD_EVAL_APPLY_MODES="${HEAD_EVAL_APPLY_MODES:-all,first_decode}"
# Full: only the canonical combo from previous datasets
FULL_MLP_TIMING="${FULL_MLP_TIMING:-prefill}"
FULL_HEAD_TIMING="${FULL_HEAD_TIMING:-all}"

EVAL_SAME_SEED_ACROSS_ALPHA="${EVAL_SAME_SEED_ACROSS_ALPHA:-1}"

# --- Loss: pure state margin (offline, no online policy) ---
# Rationale: we sample trajectories from base model, can't update online.
# gain (margin_steered - margin_base) compares steered behavior against base
# trajectories, which is semantically mismatched. Instead, optimize for the
# steered model itself having clear behavioral distinction (margin > 0).
# score_mode=avglogp: signal diluted by neutral tokens, but no answer position
# for IMDB. Relies on large training set to compensate.
SCORE_MODE="${SCORE_MODE:-avglogp}"
STATE_MARGIN_WEIGHT="${STATE_MARGIN_WEIGHT:-1.0}"
GAIN_WEIGHT="${GAIN_WEIGHT:-0.0}"
TARGET_MARGIN="${TARGET_MARGIN:-0.0}"

export HF_HOME="${HF_HOME:-LOCAL_HOME/RPEC/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export DATASETS_OFFLINE="${DATASETS_OFFLINE:-1}"

if [[ -f LOCAL_HOME/anaconda3/etc/profile.d/conda.sh && -z "${SKIP_CONDA:-}" ]]; then
  # shellcheck disable=SC1091
  source LOCAL_HOME/anaconda3/etc/profile.d/conda.sh
  conda activate "$CONDA_ENV"
fi
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

mkdir -p "$ROOT"
echo "[$(date -Is)] imdb_confiqa_staged root=$ROOT gpu=$GPU model=$MODEL" | tee -a "$ROOT/matrix.log"
echo "[$(date -Is)] train=$TRAIN_PROMPTS val=$VAL_PROMPTS eval=$FINAL_EVAL_PROMPTS" | tee -a "$ROOT/matrix.log"
echo "[$(date -Is)] loss score=$SCORE_MODE state_margin=$STATE_MARGIN_WEIGHT gain=$GAIN_WEIGHT" | tee -a "$ROOT/matrix.log"
echo "[$(date -Is)] discovery rows=$DISCOVERY_ROWS mlp_topk=$DISCOVERY_TOPK_MLP neg_topk=$DISCOVERY_TOPK_NEGATIVE_MLP attn_topk=$DISCOVERY_TOPK_ATTN_LAYERS" | tee -a "$ROOT/matrix.log"
echo "[$(date -Is)] thresholds sign=$MIN_SIGN_CONSISTENCY mlp_delta=$MIN_MLP_ABS_MEAN_DELTA attn_delta=$MIN_ATTN_ABS_MEAN_DELTA ci_no_cross=$MLP_REQUIRE_DIRECTIONAL_CI" | tee -a "$ROOT/matrix.log"
echo "[$(date -Is)] alpha_sweep=$ALPHA_SWEEP" | tee -a "$ROOT/matrix.log"
echo "[$(date -Is)] staged_timing mlp=[$MLP_EVAL_APPLY_MODES] head=[$HEAD_EVAL_APPLY_MODES] full=mlp:$FULL_MLP_TIMING+head:$FULL_HEAD_TIMING" | tee -a "$ROOT/matrix.log"

# ============================================================
# Phase 1: Base training (env → generate → score → pairs → scan → select → train)
# ============================================================
if [[ "$RUN_BASE" == "1" ]]; then
  echo "[$(date -Is)] === Phase 1: Base training ===" | tee -a "$ROOT/matrix.log"
  ROOT="$ROOT" \
  GPU="$GPU" \
  MODEL="$MODEL" \
  SCORER_MODEL="$SCORER_MODEL" \
  HF_DATASET="$HF_DATASET" \
  HF_SPLIT="$HF_SPLIT" \
  USE_SEPARATE_EVAL_ENV=1 \
  EVAL_HF_SPLIT="$EVAL_HF_SPLIT" \
  PREFIX_MODE=tokenizer \
  TOKENIZER="$MODEL" \
  SHUFFLE_PROMPTS=1 \
  TRAIN_PROMPTS="$TRAIN_PROMPTS" \
  VAL_PROMPTS="$VAL_PROMPTS" \
  EVAL_PROMPTS=0 \
  FINAL_EVAL_PROMPTS="$FINAL_EVAL_PROMPTS" \
  RUN_GENERATE="$BASE_RUN_GENERATE" \
  RUN_SCORE="$BASE_RUN_SCORE" \
  RUN_EVAL=0 \
  PREPARE_EVAL_ENV="$BASE_PREPARE_EVAL_ENV" \
  RUN_HEADS="$RUN_HEADS" \
  MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
  GENERATION_SPLIT=all \
  COMPLETIONS_PER_PREFIX="$COMPLETIONS_PER_PREFIX" \
  MIN_SCORE_MARGIN="${MIN_SCORE_MARGIN:-0.1}" \
  DISCOVERY_ROWS="$DISCOVERY_ROWS" \
  DISCOVERY_TOPK_MLP="$DISCOVERY_TOPK_MLP" \
  DISCOVERY_TOPK_NEGATIVE_MLP="$DISCOVERY_TOPK_NEGATIVE_MLP" \
  DISCOVERY_TOPK_ATTN_LAYERS="$DISCOVERY_TOPK_ATTN_LAYERS" \
  MIN_SIGN_CONSISTENCY="$MIN_SIGN_CONSISTENCY" \
  MLP_REQUIRE_DIRECTIONAL_CI="$MLP_REQUIRE_DIRECTIONAL_CI" \
  MIN_MLP_ABS_MEAN_DELTA="$MIN_MLP_ABS_MEAN_DELTA" \
  MIN_ATTN_ABS_MEAN_DELTA="$MIN_ATTN_ABS_MEAN_DELTA" \
  EXCLUDE_MLP_LAYERS="${EXCLUDE_MLP_LAYERS:-0}" \
  EXCLUDE_ATTN_LAYERS="${EXCLUDE_ATTN_LAYERS:-0}" \
  ALLOW_EMPTY_ATTN=1 \
  HEAD_SCAN_ROWS="$HEAD_SCAN_ROWS" \
  HEAD_TOPK="$HEAD_TOPK" \
  HEAD_SCAN_FACTORS="${HEAD_SCAN_FACTORS:-0.0,0.5,1.5}" \
  ALLOW_NONPOSITIVE_HEADS="${ALLOW_NONPOSITIVE_HEADS:-0}" \
  MAX_TRAIN_PAIRS="$MAX_TRAIN_PAIRS" \
  MAX_VAL_PAIRS="$MAX_VAL_PAIRS" \
  EPOCHS="$EPOCHS" \
  LR="$LR" \
  LAMBDA_NORM="$LAMBDA_NORM" \
  ALPHA_SWEEP="$ALPHA_SWEEP" \
  TRAIN_APPLY_MODE="$TRAIN_APPLY_MODE" \
  HEAD_TRAIN_APPLY_MODE="$HEAD_TRAIN_APPLY_MODE" \
  SCORE_MODE="$SCORE_MODE" \
  STATE_MARGIN_WEIGHT="$STATE_MARGIN_WEIGHT" \
  GAIN_WEIGHT="$GAIN_WEIGHT" \
  TARGET_MARGIN="$TARGET_MARGIN" \
  EVAL_MAX_ROWS="$FINAL_EVAL_PROMPTS" \
  EVAL_SAMPLES_PER_PROMPT="$EVAL_SAMPLES_PER_PROMPT" \
  TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}" \
  DEVICE="${DEVICE:-cuda}" \
  bash scripts/run_imdb_sentiment_cast_base.sh | tee -a "$ROOT/matrix.log"
fi

# ============================================================
# Helper: run eval for a single control/timing combo
# ============================================================
run_eval() {
  local control_name="$1"
  local out_name="$2"
  local eval_apply_mode="$3"
  local component_apply_mode="$4"
  local head_apply_mode="$5"
  local component_actuator="$6"
  local head_actuator="$7"

  echo "[$(date -Is)] eval control=$control_name out=$out_name apply=$eval_apply_mode comp=$component_apply_mode head=$head_apply_mode" | tee -a "$ROOT/matrix.log"
  ROOT="$ROOT" \
  GPU="$GPU" \
  MODEL="$MODEL" \
  SCORER_MODEL="$SCORER_MODEL" \
  USE_SEPARATE_EVAL_ENV=1 \
  EVAL_ONLY=1 \
  EVAL_OUT="$ROOT/$out_name" \
  EVAL_CONTROL_NAME="$control_name" \
  EVAL_APPLY_MODE="$eval_apply_mode" \
  EVAL_COMPONENT_APPLY_MODE="$component_apply_mode" \
  EVAL_HEAD_APPLY_MODE="$head_apply_mode" \
  EVAL_ACTUATOR="$component_actuator" \
  EVAL_HEAD_ACTUATOR="$head_actuator" \
  EVAL_ALPHA_SWEEP="$EVAL_ALPHA_SWEEP" \
  EVAL_MAX_ROWS="$FINAL_EVAL_PROMPTS" \
  EVAL_SAMPLES_PER_PROMPT="$EVAL_SAMPLES_PER_PROMPT" \
  EVAL_SAME_SEED_ACROSS_ALPHA="$EVAL_SAME_SEED_ACROSS_ALPHA" \
  MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
  TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}" \
  DEVICE="${DEVICE:-cuda}" \
  bash scripts/run_imdb_sentiment_cast_base.sh | tee -a "$ROOT/matrix.log"
}

# Helper: pick best timing from score summaries
pick_best_timing() {
  local prefix="$1"
  local metric="${2:-positive_rate}"
  local best_dir=""
  local best_val="-1"
  for dir in "$ROOT"/eval_${prefix}_*; do
    [[ -d "$dir" ]] || continue
    summary="$dir/score_summary.csv"
    [[ -f "$summary" ]] || continue
    # pick max positive_rate across non-zero alphas
    val="$(python - "$summary" "$metric" <<'PY'
import csv, sys
path, metric = sys.argv[1], sys.argv[2]
best = -1.0
with open(path) as f:
    for row in csv.DictReader(f):
        a = float(row.get("alpha", 0))
        if a == 0:
            continue
        v = float(row.get(metric, 0))
        if v > best:
            best = v
print(f"{best:.6f}")
PY
)"
    if python -c "print(1 if float('$val') > float('$best_val') else 0)" | grep -q 1; then
      best_val="$val"
      best_dir="$dir"
    fi
  done
  basename "$best_dir" 2>/dev/null || echo ""
}

MLP_ACT="$ROOT/train_mlp_positive/fixed_actuator.pt"
HEAD_ACT="$ROOT/train_heads/head_actuator.pt"

# ============================================================
# Phase 2: MLP internal timing selection
# ============================================================
if [[ "$RUN_MLP_SELECT" == "1" && -f "$MLP_ACT" ]]; then
  echo "[$(date -Is)] === Phase 2: MLP timing selection ===" | tee -a "$ROOT/matrix.log"
  IFS=',' read -r -a MLP_MODES <<< "$MLP_EVAL_APPLY_MODES"
  for mode in "${MLP_MODES[@]}"; do
    mode="${mode//[[:space:]]/}"
    [[ -z "$mode" ]] && continue
    run_eval "mlp_${mode}" "eval_mlp_${mode}" "$mode" "$mode" "" "$MLP_ACT" ""
  done
  BEST_MLP_TIMING="$(pick_best_timing mlp)"
  echo "[$(date -Is)] best MLP timing: $BEST_MLP_TIMING" | tee -a "$ROOT/matrix.log"
else
  BEST_MLP_TIMING="${FULL_MLP_TIMING}"
  echo "[$(date -Is)] skip MLP timing selection, using default: $BEST_MLP_TIMING" | tee -a "$ROOT/matrix.log"
fi

# ============================================================
# Phase 3: Head internal timing selection
# ============================================================
if [[ "$RUN_HEAD_SELECT" == "1" && -f "$HEAD_ACT" ]]; then
  echo "[$(date -Is)] === Phase 3: Head timing selection ===" | tee -a "$ROOT/matrix.log"
  IFS=',' read -r -a HEAD_MODES <<< "$HEAD_EVAL_APPLY_MODES"
  for mode in "${HEAD_MODES[@]}"; do
    mode="${mode//[[:space:]]/}"
    [[ -z "$mode" ]] && continue
    run_eval "head_${mode}" "eval_head_${mode}" "$mode" "" "$mode" "" "$HEAD_ACT"
  done
  BEST_HEAD_TIMING="$(pick_best_timing head)"
  echo "[$(date -Is)] best Head timing: $BEST_HEAD_TIMING" | tee -a "$ROOT/matrix.log"
else
  BEST_HEAD_TIMING="${FULL_HEAD_TIMING}"
  echo "[$(date -Is)] skip Head timing selection, using default: $BEST_HEAD_TIMING" | tee -a "$ROOT/matrix.log"
fi

# ============================================================
# Phase 4: Full combined (best MLP timing + best head timing)
# ============================================================
if [[ "$RUN_FULL_COMBINED" == "1" && -f "$MLP_ACT" && -f "$HEAD_ACT" ]]; then
  echo "[$(date -Is)] === Phase 4: Full combined (mlp=$BEST_MLP_TIMING head=$BEST_HEAD_TIMING) ===" | tee -a "$ROOT/matrix.log"
  # Extract the timing string from the eval dir name
  MLP_T="${BEST_MLP_TIMING#eval_mlp_}"
  HEAD_T="${BEST_HEAD_TIMING#eval_head_}"
  [[ -z "$MLP_T" ]] && MLP_T="${FULL_MLP_TIMING}"
  [[ -z "$HEAD_T" ]] && HEAD_T="${FULL_HEAD_TIMING}"
  run_eval "full_mlp${MLP_T}_head${HEAD_T}" "eval_full_mlp${MLP_T}_head${HEAD_T}" "all" "$MLP_T" "$HEAD_T" "$MLP_ACT" "$HEAD_ACT"
fi

# ============================================================
# Phase 5: Summary
# ============================================================
python -m screscomp.cli.summarize_imdb_sentiment_matrix \
  --root "$ROOT" \
  --out-csv "$ROOT/matrix_score_summary.csv" | tee -a "$ROOT/matrix.log"

find "$ROOT" -maxdepth 3 \( \
  -name matrix_score_summary.csv -o \
  -name alpha_summary.csv -o \
  -name vector_summary.csv -o \
  -name component_screen.csv -o \
  -name components.csv -o \
  -name mlp_positive_components.csv -o \
  -name mlp_negative_components.csv -o \
  -name attention_layers.csv -o \
  -name selected_heads.csv -o \
  -name score_summary.csv \
\) -print | sort | tee -a "$ROOT/matrix.log"

echo "[$(date -Is)] imdb_confiqa_staged done root=$ROOT" | tee -a "$ROOT/matrix.log"
