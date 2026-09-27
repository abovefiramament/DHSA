#!/usr/bin/env bash
set -euo pipefail

# Parallel sharded start-policy generation for IMDb sentiment experiments.
# Use this when full 25k-prefix generation is too slow for one GPU.

cd LOCAL_HOME/RPEC/projects/screscomp

ROOT="${ROOT:?Set ROOT to an existing IMDb sentiment run root}"
CONDA_ENV="${CONDA_ENV:-screscomp}"
MODEL="${MODEL:-edbeeching/gpt2-large-imdb}"
PROMPTS_JSONL="${PROMPTS_JSONL:-$ROOT/env/prompts.jsonl}"
OUT_DIR="${OUT_DIR:-$ROOT/start_policy_shards}"
COMBINED_JSONL="${COMBINED_JSONL:-$ROOT/start_policy/generations.jsonl}"

GPU_IDS="${GPU_IDS:-0}"
SHARD_ROWS="${SHARD_ROWS:-1000}"
GENERATION_SPLIT="${GENERATION_SPLIT:-all}"
COMPLETIONS_PER_PREFIX="${COMPLETIONS_PER_PREFIX:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
GENERATION_DO_SAMPLE="${GENERATION_DO_SAMPLE:-1}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:-50}"
STOP_STRINGS="${STOP_STRINGS:-}"
SEED="${SEED:-42}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"

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

if [[ ! -f "$PROMPTS_JSONL" ]]; then
  echo "Missing prompts: $PROMPTS_JSONL" >&2
  exit 2
fi

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
GPU_COUNT="${#GPUS[@]}"
if [[ "$GPU_COUNT" -le 0 ]]; then
  echo "GPU_IDS is empty" >&2
  exit 2
fi

TOTAL_ROWS="${TOTAL_ROWS:-$(python - "$PROMPTS_JSONL" "$GENERATION_SPLIT" <<'PY'
import json, sys
path, split = sys.argv[1], sys.argv[2]
n = 0
with open(path, encoding="utf-8-sig") as f:
    for line in f:
        if not line.strip():
            continue
        row = json.loads(line)
        admitted = str(row.get("admitted", "1")) not in {"0", "false", "False"}
        if admitted and (split == "all" or str(row.get("split", "")) == split):
            n += 1
print(n)
PY
)}"

mkdir -p "$OUT_DIR" "$(dirname "$COMBINED_JSONL")"
echo "[$(date -Is)] sharded_start_policy root=$ROOT total_rows=$TOTAL_ROWS shard_rows=$SHARD_ROWS gpus=$GPU_IDS max_new_tokens=$MAX_NEW_TOKENS"

pids=()
shard_index=0
for ((start=0; start<TOTAL_ROWS; start+=SHARD_ROWS)); do
  rows="$SHARD_ROWS"
  if (( start + rows > TOTAL_ROWS )); then
    rows=$(( TOTAL_ROWS - start ))
  fi
  gpu="${GPUS[$(( shard_index % GPU_COUNT ))]}"
  shard_jsonl="$OUT_DIR/generations_start$(printf "%06d" "$start")_n$(printf "%06d" "$rows").jsonl"
  shard_log="$OUT_DIR/generations_start$(printf "%06d" "$start")_n$(printf "%06d" "$rows").log"
  echo "[$(date -Is)] launch shard start=$start rows=$rows gpu=$gpu out=$shard_jsonl"
  gen_args=(
    python -m screscomp.cli.generate_imdb_sentiment_completions
    --model "$MODEL"
    --prompts-jsonl "$PROMPTS_JSONL"
    --out-jsonl "$shard_jsonl"
    --split "$GENERATION_SPLIT"
    --start "$start"
    --max-rows "$rows"
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
  if [[ "$GENERATION_DO_SAMPLE" == "0" ]]; then
    gen_args+=(--no-do-sample)
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "${gen_args[@]}" > "$shard_log" 2>&1 &
  pids+=("$!")
  shard_index=$(( shard_index + 1 ))
  if (( ${#pids[@]} >= GPU_COUNT )); then
    wait -n
    still_running=()
    for pid in "${pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        still_running+=("$pid")
      fi
    done
    pids=("${still_running[@]}")
  fi
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

tmp_combined="$COMBINED_JSONL.tmp"
: > "$tmp_combined"
for shard in "$OUT_DIR"/generations_start*_n*.jsonl; do
  [[ -f "$shard" ]] || continue
  cat "$shard" >> "$tmp_combined"
done
mv "$tmp_combined" "$COMBINED_JSONL"

python - "$COMBINED_JSONL" "$TOTAL_ROWS" "$COMPLETIONS_PER_PREFIX" <<'PY'
import sys
path, total, per = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
with open(path, encoding="utf-8") as f:
    n = sum(1 for line in f if line.strip())
expected = total * per
print(f"[sharded-start-policy] combined_rows={n} expected={expected} path={path}")
if n != expected:
    raise SystemExit(f"row count mismatch: got {n}, expected {expected}")
PY

echo "[$(date -Is)] sharded_start_policy done combined=$COMBINED_JSONL"
