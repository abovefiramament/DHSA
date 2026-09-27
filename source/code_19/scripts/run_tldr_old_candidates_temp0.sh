#!/usr/bin/env bash
set -euo pipefail

cd LOCAL_HOME/RPEC/projects/screscomp
if [[ -f LOCAL_HOME/anaconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source LOCAL_HOME/anaconda3/etc/profile.d/conda.sh
  conda activate screscomp
fi
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

GPU="${GPU:-3}"
OUT_ROOT="${OUT_ROOT:-runs/tldr_gptj_dpo_20260608_163146}"
MODEL="${MODEL:-LOCAL_HOME/RPEC/hf_cache/hub/models--CarperAI--openai_summarize_tldr_sft}"
PROMPTS_JSONL="${PROMPTS_JSONL:-LOCAL_HOME/RPEC/projects/screscomp/data_tldr/tldr_gptj_prepared/prompts/test_prompts.jsonl}"
TEST_MAX_ROWS="${TEST_MAX_ROWS:-320}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-1}"
CRESCOMP_CUDA_MEMORY_LIMIT_GB="${CRESCOMP_CUDA_MEMORY_LIMIT_GB:-30}"
export CRESCOMP_CUDA_MEMORY_LIMIT_GB
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

run_candidate() {
  local name="$1"
  local actuator="$2"
  local out_dir="$OUT_ROOT/open_test/${name}_temp0"
  local output="$out_dir/generations.jsonl"
  mkdir -p "$out_dir"

  local existing=0
  if [[ -s "$output" ]]; then
    existing="$(wc -l < "$output" | tr -d ' ')"
  fi
  if [[ "$existing" -ge "$TEST_MAX_ROWS" ]]; then
    echo "[temp0] already complete candidate=$name rows=$existing/$TEST_MAX_ROWS"
    return 0
  fi

  echo "[temp0] generating candidate=$name gpu=$GPU"
  CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.run_imdb_sentiment_actuator_generation \
    --model "$MODEL" \
    --prompts-jsonl "$PROMPTS_JSONL" \
    --out-jsonl "$output" \
    --manifest-json "$out_dir/generation_manifest.json" \
    --head-actuator "$actuator" \
    --control-name "${name}_temp0" \
    --alpha-sweep 1.0 \
    --generation-apply-mode all \
    --head-apply-mode all \
    --split test \
    --start 0 \
    --max-rows "$TEST_MAX_ROWS" \
    --samples-per-prompt 1 \
    --generation-batch-size "$GENERATION_BATCH_SIZE" \
    --max-new-tokens 100 \
    --no-do-sample \
    --temperature 0 \
    --top-p 0.9 \
    --top-k 0 \
    --seed 42 \
    --same-seed-across-alpha \
    --torch-dtype auto \
    --device auto \
    > "$out_dir/generate.log" 2>&1
}

run_candidate \
  old_four \
  "$OUT_ROOT/open_test/head_unified_sweetspot_formal_dpo_aligned/head_actuator_per_head_scaled.pt"
run_candidate \
  old_drop_L9 \
  "$OUT_ROOT/open_test/drop_L9_h4_full320_aligned/head_actuator.pt"

echo "[temp0] done"
