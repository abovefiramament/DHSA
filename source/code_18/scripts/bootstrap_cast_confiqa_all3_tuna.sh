#!/usr/bin/env bash
set -euo pipefail

# One-command bootstrap for the ConFiQA CAST minimal loop.
# - Uses Tsinghua/TUNA for Python packages.
# - Downloads the Llama-3 base model and Context-DPO PEFT adapter.
# - Runs QA / MR / MC serially with small-sample CAST training and heldout evaluation.
#
# Gated base model note:
#   meta-llama/Meta-Llama-3-8B-Instruct requires accepted access on Hugging Face.
#   Set HF_TOKEN or run `hf auth login` before this script.

cd "$(dirname "${BASH_SOURCE[0]}")/.."

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" >&2
}

die() {
  log "ERROR: $*"
  exit 1
}

repo_cache_dir() {
  local repo_id="$1"
  local escaped="${repo_id//\//--}"
  printf '%s/models--%s' "$HF_HUB_CACHE" "$escaped"
}

CONDA_ENV="${CONDA_ENV:-screscomp}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"

# TUNA is for Python/Conda packages. Hugging Face model files are not mirrored by
# TUNA, so use a configurable Hub endpoint. hf-mirror is the common stable choice
# from mainland China; set HF_ENDPOINT=https://huggingface.co to force official HF.
HF_HOME="${HF_HOME:-LOCAL_HOME/.cache/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HF_MAX_WORKERS="${HF_MAX_WORKERS:-1}"
DOWNLOAD_RETRIES="${DOWNLOAD_RETRIES:-8}"
DOWNLOAD_RETRY_SLEEP="${DOWNLOAD_RETRY_SLEEP:-30}"
export HF_HOME HF_HUB_CACHE HF_ENDPOINT

MODEL_ID="${MODEL_ID:-meta-llama/Meta-Llama-3-8B-Instruct}"
CONTEXT_DPO_ID="${CONTEXT_DPO_ID:-Bibaolong/Context-Faithful-LLaMA-3-8b-instruct}"
BASE_MODEL_CACHE_DIR="${BASE_MODEL_CACHE_DIR:-$(repo_cache_dir "$MODEL_ID")}"
CONTEXT_DPO_CACHE_DIR="${CONTEXT_DPO_CACHE_DIR:-$(repo_cache_dir "$CONTEXT_DPO_ID")}"
MODEL="${MODEL:-$BASE_MODEL_CACHE_DIR}"
CONTEXT_DPO_MODEL="${CONTEXT_DPO_MODEL:-$CONTEXT_DPO_CACHE_DIR}"

GPU="${GPU:-3}"
TASKS="${TASKS:-qa mr mc}"
OUT="${OUT:-data_ckplug/cast_confiqa_all3_smalltrain_heldout5000_v0}"

TRAIN_SOURCE_ROWS="${TRAIN_SOURCE_ROWS:-300}"
EVAL_SOURCE_START="${EVAL_SOURCE_START:-1000}"
EVAL_SOURCE_ROWS="${EVAL_SOURCE_ROWS:-}"
VAL_MOD="${VAL_MOD:-5}"
TRAIN_ROWS="${TRAIN_ROWS:-240}"
VAL_ROWS="${VAL_ROWS:-60}"
EPOCHS="${EPOCHS:-2}"
MLP_EPOCHS="${MLP_EPOCHS:-$EPOCHS}"
HEAD_EPOCHS="${HEAD_EPOCHS:-$EPOCHS}"
MLP_TRAIN_ROWS="${MLP_TRAIN_ROWS:-$TRAIN_ROWS}"
MLP_VAL_ROWS="${MLP_VAL_ROWS:-$VAL_ROWS}"
HEAD_TRAIN_ROWS="${HEAD_TRAIN_ROWS:-$TRAIN_ROWS}"
HEAD_VAL_ROWS="${HEAD_VAL_ROWS:-$VAL_ROWS}"
EVAL_SPLIT="${EVAL_SPLIT:-all}"
EVAL_ROWS="${EVAL_ROWS:-0}"
RUN_CONTEXT_DPO="${RUN_CONTEXT_DPO:-1}"
MLP_ALPHA="${MLP_ALPHA:-0.05}"
HEAD_ALPHA="${HEAD_ALPHA:-0.5}"
EMPTY_CACHE_EVERY="${EMPTY_CACHE_EVERY:-25}"
DISCOVER_COMPONENTS="${DISCOVER_COMPONENTS:-1}"
REUSE_DISCOVERY="${REUSE_DISCOVERY:-0}"
DISCOVERY_ROWS="${DISCOVERY_ROWS:-60}"
DISCOVERY_TOPK_MLP="${DISCOVERY_TOPK_MLP:-4}"
DISCOVERY_TOPK_ATTN_LAYERS="${DISCOVERY_TOPK_ATTN_LAYERS:-4}"
HEAD_SCAN_ROWS="${HEAD_SCAN_ROWS:-24}"
HEAD_SCAN_FACTORS="${HEAD_SCAN_FACTORS:-0.0,1.5}"
HEAD_TOPK="${HEAD_TOPK:-4}"
HEAD_REFINE_EVAL_ROWS="${HEAD_REFINE_EVAL_ROWS:-1}"
AUTO_TUNE="${AUTO_TUNE:-0}"
FOREGROUND="${FOREGROUND:-0}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
DOWNLOAD_MODELS="${DOWNLOAD_MODELS:-1}"

USER_CAST_CONTROLS="${CAST_CONTROLS:-}"
CAST_CONTROLS_DEFAULT="cast_attn=head_act:suppress:${HEAD_ALPHA}:all+head_act:boost:${HEAD_ALPHA}:all;cast_full=head_act:suppress:${HEAD_ALPHA}:all+head_act:boost:${HEAD_ALPHA}:all+comp:prior_mlp:${MLP_ALPHA}:prefill"
CAST_CONTROLS="${CAST_CONTROLS:-$CAST_CONTROLS_DEFAULT}"

activate_env() {
  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    deactivate || true
  fi
  if [[ -f LOCAL_HOME/anaconda3/etc/profile.d/conda.sh ]]; then
    # shellcheck disable=SC1091
    source LOCAL_HOME/anaconda3/etc/profile.d/conda.sh
  elif command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1090
    source "$(conda info --base)/etc/profile.d/conda.sh"
  else
    die "conda not found; expected environment '$CONDA_ENV'"
  fi
  conda activate "$CONDA_ENV"
  export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
}

configure_tuna_pip() {
  export PIP_INDEX_URL PIP_TRUSTED_HOST
  python -m pip config set global.index-url "$PIP_INDEX_URL" >/dev/null || true
  python -m pip config set global.trusted-host "$PIP_TRUSTED_HOST" >/dev/null || true
}

install_deps() {
  [[ "$INSTALL_DEPS" == "1" ]] || return 0
  log "Installing/updating Python dependencies from TUNA: $PIP_INDEX_URL"
  python -m pip install -U \
    -i "$PIP_INDEX_URL" --trusted-host "$PIP_TRUSTED_HOST" \
    pip setuptools wheel
  python -m pip install -U \
    -i "$PIP_INDEX_URL" --trusted-host "$PIP_TRUSTED_HOST" \
    "huggingface_hub[cli]>=0.24" \
    "peft>=0.13" \
    "accelerate>=0.30" \
    "safetensors>=0.4" \
    "tqdm>=4.66"
  python -m pip install -e . \
    -i "$PIP_INDEX_URL" --trusted-host "$PIP_TRUSTED_HOST"
}

has_payload() {
  local dir="$1"
  compgen -G "$dir/*.safetensors" >/dev/null \
    || compgen -G "$dir/pytorch_model*.bin" >/dev/null \
    || compgen -G "$dir/model*.bin" >/dev/null
}

has_tokenizer_payload() {
  local dir="$1"
  [[ -f "$dir/tokenizer.json" ]] \
    || [[ -f "$dir/tokenizer.model" ]] \
    || [[ -f "$dir/vocab.json" ]] \
    || [[ -f "$dir/tokenizer_config.json" ]]
}

cached_repo_ready() {
  local repo_id="$1"
  local kind="$2"
  local cache_dir snapshot
  cache_dir="$(repo_cache_dir "$repo_id")"
  [[ -d "$cache_dir/snapshots" ]] || return 1
  for snapshot in "$cache_dir"/snapshots/*; do
    [[ -d "$snapshot" ]] || continue
    if [[ "$kind" == "base" ]]; then
      [[ -f "$snapshot/config.json" ]] && has_payload "$snapshot" && has_tokenizer_payload "$snapshot" && return 0
    else
      [[ -f "$snapshot/adapter_config.json" ]] && has_payload "$snapshot" && has_tokenizer_payload "$snapshot" && return 0
    fi
  done
  return 1
}

check_hf_auth_for_gated_base() {
  if [[ -n "${HF_TOKEN:-}" ]]; then
    return 0
  fi
  if hf auth whoami >/dev/null 2>&1; then
    return 0
  fi
  cat >&2 <<'EOF'
Missing Hugging Face auth for the gated Llama base model.

Do one of these first:
  export HF_TOKEN=hf_xxx
or:
  hf auth login

Also make sure your HF account has accepted the Meta-Llama-3-8B-Instruct license.
EOF
  exit 2
}

download_hf_repo() {
  local repo_id="$1"
  local kind="$2"
  local cache_dir
  local attempt
  cache_dir="$(repo_cache_dir "$repo_id")"

  if cached_repo_ready "$repo_id" "$kind"; then
    log "Skip $kind download; HF cache already has payload: $cache_dir"
    return 0
  fi

  for attempt in $(seq 1 "$DOWNLOAD_RETRIES"); do
    log "Prewarming $kind repo $repo_id into HF cache: $HF_HUB_CACHE attempt=$attempt/$DOWNLOAD_RETRIES"
    if hf download "$repo_id" --max-workers "$HF_MAX_WORKERS"; then
      break
    fi
    if [[ "$attempt" == "$DOWNLOAD_RETRIES" ]]; then
      die "Download failed after $DOWNLOAD_RETRIES attempts: $repo_id"
    fi
    log "Download interrupted; retrying in ${DOWNLOAD_RETRY_SLEEP}s. Existing HF cache will be reused."
    sleep "$DOWNLOAD_RETRY_SLEEP"
  done
  [[ -d "$cache_dir" ]] || die "Expected HF cache directory not found after download: $cache_dir"
  cached_repo_ready "$repo_id" "$kind" || die "Downloaded cache lacks expected $kind payload: $cache_dir"
}

download_models() {
  [[ "$DOWNLOAD_MODELS" == "1" ]] || return 0
  export HF_HOME HF_HUB_CACHE HF_ENDPOINT
  cached_repo_ready "$MODEL_ID" base || check_hf_auth_for_gated_base
  download_hf_repo "$MODEL_ID" base
  if [[ "$RUN_CONTEXT_DPO" == "1" ]]; then
    download_hf_repo "$CONTEXT_DPO_ID" adapter
  fi
}

run_all_tasks() {
  activate_env
  export HF_HOME HF_HUB_CACHE HF_ENDPOINT
  mkdir -p "$OUT"
  log "Running tasks='$TASKS' on GPU=$GPU; output=$OUT"
  log "train_source_rows=$TRAIN_SOURCE_ROWS train_rows=$TRAIN_ROWS val_rows=$VAL_ROWS eval_start=$EVAL_SOURCE_START eval_rows=$EVAL_ROWS"

  for task in $TASKS; do
    log "Starting ConFiQA subset: $task"
    GPU="$GPU" TASK="$task" \
    ROOT="$OUT/$task" \
    MODEL="$MODEL" \
    CONTEXT_DPO_MODEL="$CONTEXT_DPO_MODEL" \
    TRAIN_SOURCE_ROWS="$TRAIN_SOURCE_ROWS" \
    EVAL_SOURCE_START="$EVAL_SOURCE_START" \
    EVAL_SOURCE_ROWS="$EVAL_SOURCE_ROWS" \
    VAL_MOD="$VAL_MOD" \
    TRAIN_ROWS="$TRAIN_ROWS" \
    VAL_ROWS="$VAL_ROWS" \
    EPOCHS="$EPOCHS" \
    MLP_EPOCHS="$MLP_EPOCHS" \
    HEAD_EPOCHS="$HEAD_EPOCHS" \
    MLP_TRAIN_ROWS="$MLP_TRAIN_ROWS" \
    MLP_VAL_ROWS="$MLP_VAL_ROWS" \
    HEAD_TRAIN_ROWS="$HEAD_TRAIN_ROWS" \
    HEAD_VAL_ROWS="$HEAD_VAL_ROWS" \
    STATE_MARGIN_WEIGHT="${STATE_MARGIN_WEIGHT:-0.0}" \
    GAIN_WEIGHT="${GAIN_WEIGHT:-1.0}" \
    TARGET_MARGIN="${TARGET_MARGIN:-0.0}" \
    TARGET_GAIN="${TARGET_GAIN:-0.0}" \
    EVAL_SPLIT="$EVAL_SPLIT" \
    EVAL_ROWS="$EVAL_ROWS" \
    RUN_CONTEXT_DPO="$RUN_CONTEXT_DPO" \
    MLP_ALPHA="$MLP_ALPHA" \
    HEAD_ALPHA="$HEAD_ALPHA" \
    EMPTY_CACHE_EVERY="$EMPTY_CACHE_EVERY" \
    DISCOVER_COMPONENTS="$DISCOVER_COMPONENTS" \
    REUSE_DISCOVERY="$REUSE_DISCOVERY" \
    DISCOVERY_ROWS="$DISCOVERY_ROWS" \
    DISCOVERY_TOPK_MLP="$DISCOVERY_TOPK_MLP" \
    DISCOVERY_TOPK_ATTN_LAYERS="$DISCOVERY_TOPK_ATTN_LAYERS" \
    HEAD_SCAN_ROWS="$HEAD_SCAN_ROWS" \
    HEAD_SCAN_FACTORS="$HEAD_SCAN_FACTORS" \
    HEAD_TOPK="$HEAD_TOPK" \
    HEAD_REFINE_EVAL_ROWS="$HEAD_REFINE_EVAL_ROWS" \
    AUTO_TUNE="$AUTO_TUNE" \
    HEAD_ALPHAS="${HEAD_ALPHAS:-}" \
    MLP_ALPHAS="${MLP_ALPHAS:-}" \
    TUNE_FAMILIES="${TUNE_FAMILIES:-}" \
    TUNE_ROWS="${TUNE_ROWS:-}" \
    SELECT_MAX_PC_DROP="${SELECT_MAX_PC_DROP:-}" \
    SELECT_MAX_CONTEXT_ONLY_DROP="${SELECT_MAX_CONTEXT_ONLY_DROP:-}" \
    CAST_CONTROLS="$USER_CAST_CONTROLS" \
    bash scripts/run_cast_confiqa_min_loop.sh
    log "Finished ConFiQA subset: $task"
  done
  log "All tasks finished. Summaries:"
  find "$OUT" -maxdepth 3 \( -name generation_summary.csv -o -name run_summary.csv -o -name status.tsv \) -print | sort
}

if [[ "${CAST_BOOTSTRAP_CHILD:-0}" == "1" ]]; then
  run_all_tasks
  exit 0
fi

activate_env
configure_tuna_pip
install_deps
download_models

mkdir -p "$OUT"
cat > "$OUT/bootstrap_config.env" <<EOF
CONDA_ENV=$CONDA_ENV
PIP_INDEX_URL=$PIP_INDEX_URL
HF_HOME=$HF_HOME
HF_HUB_CACHE=$HF_HUB_CACHE
HF_ENDPOINT=$HF_ENDPOINT
HF_MAX_WORKERS=$HF_MAX_WORKERS
DOWNLOAD_RETRIES=$DOWNLOAD_RETRIES
DOWNLOAD_RETRY_SLEEP=$DOWNLOAD_RETRY_SLEEP
MODEL_ID=$MODEL_ID
CONTEXT_DPO_ID=$CONTEXT_DPO_ID
BASE_MODEL_CACHE_DIR=$BASE_MODEL_CACHE_DIR
CONTEXT_DPO_CACHE_DIR=$CONTEXT_DPO_CACHE_DIR
MODEL=$MODEL
CONTEXT_DPO_MODEL=$CONTEXT_DPO_MODEL
GPU=$GPU
TASKS=$TASKS
TRAIN_SOURCE_ROWS=$TRAIN_SOURCE_ROWS
EVAL_SOURCE_START=$EVAL_SOURCE_START
EVAL_SOURCE_ROWS=$EVAL_SOURCE_ROWS
VAL_MOD=$VAL_MOD
TRAIN_ROWS=$TRAIN_ROWS
VAL_ROWS=$VAL_ROWS
EPOCHS=$EPOCHS
MLP_EPOCHS=$MLP_EPOCHS
HEAD_EPOCHS=$HEAD_EPOCHS
MLP_TRAIN_ROWS=$MLP_TRAIN_ROWS
MLP_VAL_ROWS=$MLP_VAL_ROWS
HEAD_TRAIN_ROWS=$HEAD_TRAIN_ROWS
HEAD_VAL_ROWS=$HEAD_VAL_ROWS
EVAL_SPLIT=$EVAL_SPLIT
EVAL_ROWS=$EVAL_ROWS
RUN_CONTEXT_DPO=$RUN_CONTEXT_DPO
MLP_ALPHA=$MLP_ALPHA
HEAD_ALPHA=$HEAD_ALPHA
EMPTY_CACHE_EVERY=$EMPTY_CACHE_EVERY
DISCOVER_COMPONENTS=$DISCOVER_COMPONENTS
REUSE_DISCOVERY=$REUSE_DISCOVERY
DISCOVERY_ROWS=$DISCOVERY_ROWS
DISCOVERY_TOPK_MLP=$DISCOVERY_TOPK_MLP
DISCOVERY_TOPK_ATTN_LAYERS=$DISCOVERY_TOPK_ATTN_LAYERS
HEAD_SCAN_ROWS=$HEAD_SCAN_ROWS
HEAD_SCAN_FACTORS=$HEAD_SCAN_FACTORS
HEAD_TOPK=$HEAD_TOPK
HEAD_REFINE_EVAL_ROWS=$HEAD_REFINE_EVAL_ROWS
AUTO_TUNE=$AUTO_TUNE
EOF

if [[ "$FOREGROUND" == "1" ]]; then
  log "Starting foreground run; log will also be written to $OUT/nohup.log"
  CAST_BOOTSTRAP_CHILD=1 bash "$0" 2>&1 | tee "$OUT/nohup.log"
else
  log "Starting detached run; log: $OUT/nohup.log"
  CAST_BOOTSTRAP_CHILD=1 nohup bash "$0" > "$OUT/nohup.log" 2>&1 &
  pid=$!
  log "Started PID=$pid"
  log "Watch with: tail -f $OUT/nohup.log"
fi
