#!/usr/bin/env bash
set -euo pipefail

cd LOCAL_HOME/RPEC/projects/screscomp
if [[ -f LOCAL_HOME/anaconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source LOCAL_HOME/anaconda3/etc/profile.d/conda.sh
  conda activate screscomp
fi
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

GPU="${GPU:-${1:-3}}"
OUT_ROOT="${OUT_ROOT:-${2:-runs/tldr_gptj_dpo_20260608_163146}}"
PPO_MODEL="${PPO_MODEL:-LOCAL_HOME/RPEC/hf_cache/hub/models--CarperAI--openai_summarize_tldr_ppo}"
SFT_TOKENIZER="${SFT_TOKENIZER:-LOCAL_HOME/RPEC/hf_cache/hub/models--CarperAI--openai_summarize_tldr_sft}"
PROMPTS_JSONL="${PROMPTS_JSONL:-LOCAL_HOME/RPEC/projects/screscomp/data_tldr/tldr_gptj_prepared/prompts/test_prompts.jsonl}"
TEST_MAX_ROWS="${TEST_MAX_ROWS:-320}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-100}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"
TOP_K="${TOP_K:-0}"
SEED="${SEED:-42}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"
CRESCOMP_CUDA_MEMORY_LIMIT_GB="${CRESCOMP_CUDA_MEMORY_LIMIT_GB:-30}"
export CRESCOMP_CUDA_MEMORY_LIMIT_GB
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PPO_OUT="${PPO_OUT:-$OUT_ROOT/open_test/ppo}"
OUTPUT="$PPO_OUT/generations.jsonl"
mkdir -p "$PPO_OUT"
existing=0
if [[ -s "$OUTPUT" ]]; then
  existing="$(wc -l < "$OUTPUT" | tr -d ' ')"
fi
if [[ "$existing" -ge "$TEST_MAX_ROWS" ]]; then
  echo "[ppo] already complete rows=$existing/$TEST_MAX_ROWS output=$OUTPUT"
  exit 0
fi

echo "[ppo] model download/load and aligned generation on GPU $GPU"
CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.generate_tldr_ppo \
  --model "$PPO_MODEL" \
  --tokenizer "$SFT_TOKENIZER" \
  --prompts-jsonl "$PROMPTS_JSONL" \
  --out-jsonl "$OUTPUT" \
  --split test \
  --start 0 \
  --max-rows "$TEST_MAX_ROWS" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --top-k "$TOP_K" \
  --seed "$SEED" \
  --torch-dtype "$TORCH_DTYPE" \
  --device "$DEVICE" \
  > "$PPO_OUT/generate.log" 2>&1

echo "[ppo] generation complete output=$OUTPUT"
