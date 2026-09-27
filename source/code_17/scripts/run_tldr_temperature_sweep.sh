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
SWEEP_ROOT="${SWEEP_ROOT:-$OUT_ROOT/open_test/temp_sweep}"
MODEL="${MODEL:-LOCAL_HOME/RPEC/hf_cache/hub/models--CarperAI--openai_summarize_tldr_sft}"
PPO_MODEL="${PPO_MODEL:-LOCAL_HOME/RPEC/hf_cache/hub/models--CarperAI--openai_summarize_tldr_ppo}"
SFT_TOKENIZER="${SFT_TOKENIZER:-LOCAL_HOME/RPEC/hf_cache/hub/models--CarperAI--openai_summarize_tldr_sft}"
PROMPTS_JSONL="${PROMPTS_JSONL:-LOCAL_HOME/RPEC/projects/screscomp/data_tldr/tldr_gptj_prepared/prompts/test_prompts.jsonl}"
JUDGE_PROMPTS_JSONL="${JUDGE_PROMPTS_JSONL:-data_tldr/tldr_gptj_prepared/prompts/test_prompts.jsonl}"
OLD_FOUR_ACTUATOR="${OLD_FOUR_ACTUATOR:-$OUT_ROOT/open_test/head_unified_sweetspot_formal_dpo_aligned/head_actuator_per_head_scaled.pt}"
TEST_MAX_ROWS="${TEST_MAX_ROWS:-320}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-100}"
TOP_P="${TOP_P:-0.9}"
TOP_K="${TOP_K:-0}"
SEED="${SEED:-42}"
TEMPS="${TEMPS:-0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-2}"
GREEDY_BATCH_SIZE="${GREEDY_BATCH_SIZE:-1}"
RUN_GENERATION="${RUN_GENERATION:-1}"
RUN_JUDGE="${RUN_JUDGE:-1}"
JUDGE_HUMAN_ONLY="${JUDGE_HUMAN_ONLY:-0}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-3}"
REVIEW_FRACTION="${REVIEW_FRACTION:-1.0}"
JUDGE_MODEL="${JUDGE_MODEL:-deepseek-v4-pro}"
THINKING_MODE="${THINKING_MODE:-disabled}"
DEEPSEEK_API_KEY_FILE="${DEEPSEEK_API_KEY_FILE:-$HOME/.config/screscomp/deepseek_api_key}"
CRESCOMP_CUDA_MEMORY_LIMIT_GB="${CRESCOMP_CUDA_MEMORY_LIMIT_GB:-30}"
export CRESCOMP_CUDA_MEMORY_LIMIT_GB
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$SWEEP_ROOT"

temp_tag() {
  local temp="$1"
  python - "$temp" <<'PY'
import sys
t = float(sys.argv[1])
if abs(t) < 1e-9:
    print("temp0")
elif abs(t - round(t)) < 1e-9:
    print(f"temp{int(round(t))}")
else:
    print(("temp%.1f" % t).replace(".", "p"))
PY
}

is_complete_generation() {
  local kind="$1"
  local path="$2"
  local manifest="${3:-}"
  local temp="$4"
  python - "$kind" "$path" "$manifest" "$temp" "$TEST_MAX_ROWS" "$MAX_NEW_TOKENS" "$TOP_P" "$TOP_K" "$OLD_FOUR_ACTUATOR" <<'PY'
import json
import math
import sys
from pathlib import Path

kind, path_s, manifest_s, temp_s, rows_s, max_new_s, top_p_s, top_k_s, old_actuator = sys.argv[1:]
path = Path(path_s)
expected_rows = int(rows_s)
temp = float(temp_s)
expected_do_sample = temp > 0.0
if not path.exists() or path.stat().st_size == 0:
    raise SystemExit(1)
records = []
with path.open("r", encoding="utf-8-sig") as stream:
    for line in stream:
        if line.strip():
            records.append(json.loads(line))
if len(records) != expected_rows:
    raise SystemExit(1)
sample_ids = [str(row.get("sample_id", "")) for row in records]
if len(set(sample_ids)) != expected_rows or any(not sid for sid in sample_ids):
    raise SystemExit(1)
if kind == "old_four":
    manifest = Path(manifest_s)
    if not manifest.exists():
        raise SystemExit(1)
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    checks = [
        int(meta.get("completed_rows", -1)) == expected_rows,
        int(meta.get("max_new_tokens", -1)) == int(max_new_s),
        bool(meta.get("do_sample")) == expected_do_sample,
        abs(float(meta.get("temperature", -999.0)) - temp) < 1e-9,
        abs(float(meta.get("top_p", -999.0)) - float(top_p_s)) < 1e-9,
        int(meta.get("top_k", -999)) == int(top_k_s),
        str(meta.get("head_actuator_path", "")).endswith(str(old_actuator)),
    ]
    if not all(checks):
        raise SystemExit(1)
else:
    for row in records:
        checks = [
            bool(row.get("do_sample")) == expected_do_sample,
            abs(float(row.get("temperature", -999.0)) - temp) < 1e-9,
            abs(float(row.get("top_p", -999.0)) - float(top_p_s)) < 1e-9,
            int(row.get("top_k", -999)) == int(top_k_s),
            int(row.get("max_new_tokens", -1)) == int(max_new_s),
            "openai_summarize_tldr_ppo" in str(row.get("model", "")),
        ]
        if not all(checks):
            raise SystemExit(1)
PY
}

old_four_path_for_temp() {
  local temp="$1"
  local tag
  tag="$(temp_tag "$temp")"
  if [[ "$tag" == "temp0" ]]; then
    echo "$OUT_ROOT/open_test/old_four_temp0/generations.jsonl"
  elif [[ "$tag" == "temp0p7" ]]; then
    echo "$OUT_ROOT/open_test/head_unified_sweetspot_formal_dpo_aligned/generations.jsonl"
  else
    echo "$SWEEP_ROOT/old_four_${tag}/generations.jsonl"
  fi
}

old_four_manifest_for_temp() {
  local temp="$1"
  local path
  path="$(old_four_path_for_temp "$temp")"
  echo "$(dirname "$path")/generation_manifest.json"
}

ppo_path_for_temp() {
  local temp="$1"
  local tag
  tag="$(temp_tag "$temp")"
  if [[ "$tag" == "temp0" ]]; then
    echo "$OUT_ROOT/open_test/ppo_temp0/generations.jsonl"
  elif [[ "$tag" == "temp0p7" ]]; then
    echo "$OUT_ROOT/open_test/ppo/generations.jsonl"
  else
    echo "$SWEEP_ROOT/ppo_${tag}/generations.jsonl"
  fi
}

judge_dir_for_temp() {
  local temp="$1"
  local tag
  tag="$(temp_tag "$temp")"
  if [[ "$tag" == "temp0" ]]; then
    echo "$OUT_ROOT/open_test/ds4_old_four_temp0_vs_ppo_temp0_fullswap"
  else
    echo "$SWEEP_ROOT/ds4_old_four_vs_ppo_${tag}_fullswap"
  fi
}

judge_complete() {
  local out_dir="$1"
  python - "$out_dir" "$TEST_MAX_ROWS" "$REVIEW_FRACTION" "$JUDGE_HUMAN_ONLY" <<'PY'
import csv
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
expected_rows = int(sys.argv[2])
review_fraction = float(sys.argv[3])
human_only = sys.argv[4] == "1"
expected_pairs = 2 if human_only else 3
summary_json = out_dir / "pairwise_summary.json"
summary_csv = out_dir / "pairwise_summary.csv"
manifest = out_dir / "judge_manifest.json"
if not summary_json.exists() or not summary_csv.exists() or not manifest.exists():
    raise SystemExit(1)
summary = json.loads(summary_json.read_text(encoding="utf-8"))
meta = json.loads(manifest.read_text(encoding="utf-8"))
if int(meta.get("sample_rows", -1)) != expected_rows:
    raise SystemExit(1)
if abs(float(meta.get("review_fraction", -1.0)) - review_fraction) > 1e-9:
    raise SystemExit(1)
if bool(meta.get("human_only", False)) != human_only:
    raise SystemExit(1)
if int(summary.get("first_pass_ok", -1)) != expected_rows * expected_pairs:
    raise SystemExit(1)
if review_fraction >= 1.0 and int(summary.get("review_ok", -1)) != expected_rows * expected_pairs:
    raise SystemExit(1)
with summary_csv.open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
if len(rows) != expected_pairs:
    raise SystemExit(1)
for row in rows:
    if int(row["n"]) != expected_rows:
        raise SystemExit(1)
    if review_fraction >= 1.0 and int(row["reviewed"]) != expected_rows:
        raise SystemExit(1)
PY
}

generate_old_four() {
  local temp="$1"
  local tag path manifest out_dir batch sample_flag
  tag="$(temp_tag "$temp")"
  path="$(old_four_path_for_temp "$temp")"
  manifest="$(old_four_manifest_for_temp "$temp")"
  if is_complete_generation old_four "$path" "$manifest" "$temp"; then
    echo "[sweep] reuse old_four $tag path=$path"
    return 0
  fi
  out_dir="$(dirname "$path")"
  mkdir -p "$out_dir"
  if [[ "$temp" == "0" || "$temp" == "0.0" ]]; then
    sample_flag=(--no-do-sample)
    batch="$GREEDY_BATCH_SIZE"
  else
    sample_flag=(--do-sample)
    batch="$GENERATION_BATCH_SIZE"
  fi
  echo "[sweep] generate old_four $tag temp=$temp gpu=$GPU out=$path"
  CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.run_imdb_sentiment_actuator_generation \
    --model "$MODEL" \
    --prompts-jsonl "$PROMPTS_JSONL" \
    --out-jsonl "$path" \
    --manifest-json "$manifest" \
    --head-actuator "$OLD_FOUR_ACTUATOR" \
    --control-name "old_four_${tag}" \
    --alpha-sweep 1.0 \
    --generation-apply-mode all \
    --head-apply-mode all \
    --split test \
    --start 0 \
    --max-rows "$TEST_MAX_ROWS" \
    --samples-per-prompt 1 \
    --generation-batch-size "$batch" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    "${sample_flag[@]}" \
    --temperature "$temp" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --seed "$SEED" \
    --same-seed-across-alpha \
    --torch-dtype auto \
    --device auto \
    > "$out_dir/generate.log" 2>&1
}

generate_ppo() {
  local temp="$1"
  local tag path out_dir
  tag="$(temp_tag "$temp")"
  path="$(ppo_path_for_temp "$temp")"
  if is_complete_generation ppo "$path" "" "$temp"; then
    echo "[sweep] reuse ppo $tag path=$path"
    return 0
  fi
  out_dir="$(dirname "$path")"
  mkdir -p "$out_dir"
  echo "[sweep] generate ppo $tag temp=$temp gpu=$GPU out=$path"
  CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.generate_tldr_ppo \
    --model "$PPO_MODEL" \
    --tokenizer "$SFT_TOKENIZER" \
    --prompts-jsonl "$PROMPTS_JSONL" \
    --out-jsonl "$path" \
    --split test \
    --start 0 \
    --max-rows "$TEST_MAX_ROWS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature "$temp" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --seed "$SEED" \
    --torch-dtype bfloat16 \
    --device cuda \
    > "$out_dir/generate.log" 2>&1
}

run_judge_for_temp() {
  local temp="$1"
  local tag old_path ppo_path out_dir
  tag="$(temp_tag "$temp")"
  old_path="$(old_four_path_for_temp "$temp")"
  ppo_path="$(ppo_path_for_temp "$temp")"
  out_dir="$(judge_dir_for_temp "$temp")"
  if judge_complete "$out_dir"; then
    echo "[sweep] reuse judge $tag out=$out_dir"
    return 0
  fi
  if ! is_complete_generation old_four "$old_path" "$(old_four_manifest_for_temp "$temp")" "$temp"; then
    echo "[sweep] old_four generation incomplete for $tag: $old_path" >&2
    return 1
  fi
  if ! is_complete_generation ppo "$ppo_path" "" "$temp"; then
    echo "[sweep] ppo generation incomplete for $tag: $ppo_path" >&2
    return 1
  fi
  mkdir -p "$out_dir"
  judge_extra=()
  if [[ "$JUDGE_HUMAN_ONLY" == "1" ]]; then
    judge_extra+=(--human-only)
  fi
  if [[ -z "${DEEPSEEK_API_KEY:-}" && -f "$DEEPSEEK_API_KEY_FILE" ]]; then
    export DEEPSEEK_API_KEY
    DEEPSEEK_API_KEY="$(<"$DEEPSEEK_API_KEY_FILE")"
  fi
  echo "[sweep] judge $tag out=$out_dir"
  python -m screscomp.cli.evaluate_tldr_dpo_ds4 \
    --prompts-jsonl "$JUDGE_PROMPTS_JSONL" \
    --candidate "old_four_${tag}=$old_path" \
    --candidate "ppo_${tag}=$ppo_path" \
    --out-dir "$out_dir" \
    --expected-rows "$TEST_MAX_ROWS" \
    --concurrency "$JUDGE_CONCURRENCY" \
    --review-fraction "$REVIEW_FRACTION" \
    --model "$JUDGE_MODEL" \
    --thinking-mode "$THINKING_MODE" \
    --allow-batch-seed-mismatch \
    "${judge_extra[@]}" \
    > "$out_dir/judge.log" 2>&1
}

if [[ "$RUN_GENERATION" == "1" ]]; then
  for temp in $TEMPS; do
    tag="$(temp_tag "$temp")"
    echo "[sweep] ===== generate $tag temp=$temp ====="
    generate_old_four "$temp"
    generate_ppo "$temp"
  done
fi

if [[ "$RUN_JUDGE" == "1" ]]; then
  for temp in $TEMPS; do
    tag="$(temp_tag "$temp")"
    echo "[sweep] ===== judge $tag temp=$temp ====="
    run_judge_for_temp "$temp"
  done
fi

python scripts/summarize_tldr_temperature_sweep.py \
  --out-root "$OUT_ROOT" \
  --sweep-root "$SWEEP_ROOT" \
  --temps $TEMPS \
  --out-csv "$SWEEP_ROOT/temperature_sweep_balanced_summary.csv"

echo "[sweep] done summary=$SWEEP_ROOT/temperature_sweep_balanced_summary.csv"
