#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}
OUT_DIR=${OUT_DIR:-"$ROOT_DIR/runs/imdb_sentiment_env_v0"}
HF_DATASET=${HF_DATASET:-imdb}
HF_SPLIT=${HF_SPLIT:-train}
TRAIN_ROWS=${TRAIN_ROWS:-25000}
VAL_ROWS=${VAL_ROWS:-0}
EVAL_ROWS=${EVAL_ROWS:-0}
SEED=${SEED:-42}
PREFIX_MODE=${PREFIX_MODE:-whitespace}
TOKENIZER=${TOKENIZER:-}

cd "$ROOT_DIR"
export PYTHONPATH="${PYTHONPATH:-src}"

args=(
  -m screscomp.cli.prepare_imdb_sentiment_env
  --hf-dataset "$HF_DATASET"
  --hf-split "$HF_SPLIT"
  --out-dir "$OUT_DIR"
  --train-rows "$TRAIN_ROWS"
  --val-rows "$VAL_ROWS"
  --eval-rows "$EVAL_ROWS"
  --seed "$SEED"
  --prefix-mode "$PREFIX_MODE"
)

if [[ -n "$TOKENIZER" ]]; then
  args+=(--tokenizer "$TOKENIZER")
fi

python "${args[@]}"
