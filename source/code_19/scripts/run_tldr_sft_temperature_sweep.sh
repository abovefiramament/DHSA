#!/usr/bin/env bash
set -euo pipefail

cd LOCAL_HOME/RPEC/projects/screscomp
if [[ -f LOCAL_HOME/anaconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source LOCAL_HOME/anaconda3/etc/profile.d/conda.sh
  conda activate screscomp
fi
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

GPU="${GPU:-0}"
OUT_ROOT="${OUT_ROOT:-runs/tldr_gptj_dpo_20260608_163146}"
SWEEP_ROOT="${SWEEP_ROOT:-$OUT_ROOT/open_test/sft_temp_sweep}"
MODEL="${MODEL:-LOCAL_HOME/RPEC/hf_cache/hub/models--CarperAI--openai_summarize_tldr_sft}"
PROMPTS_JSONL="${PROMPTS_JSONL:-LOCAL_HOME/RPEC/projects/screscomp/data_tldr/tldr_gptj_prepared/prompts/test_prompts.jsonl}"
JUDGE_PROMPTS_JSONL="${JUDGE_PROMPTS_JSONL:-data_tldr/tldr_gptj_prepared/prompts/test_prompts.jsonl}"
TEST_MAX_ROWS="${TEST_MAX_ROWS:-320}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-100}"
TOP_P="${TOP_P:-0.9}"
TOP_K="${TOP_K:-0}"
SEED="${SEED:-42}"
TEMPS="${TEMPS:-0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-1}"
GREEDY_BATCH_SIZE="${GREEDY_BATCH_SIZE:-1}"
RUN_GENERATION="${RUN_GENERATION:-1}"
RUN_JUDGE="${RUN_JUDGE:-1}"
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

sft_path_for_temp() {
  local temp="$1"
  local tag
  tag="$(temp_tag "$temp")"
  if [[ "$tag" == "temp0p7" ]]; then
    echo "$OUT_ROOT/open_test/base/generations.jsonl"
  else
    echo "$SWEEP_ROOT/sft_${tag}/generations.jsonl"
  fi
}

sft_manifest_for_temp() {
  local temp="$1"
  local path
  path="$(sft_path_for_temp "$temp")"
  echo "$(dirname "$path")/generation_manifest.json"
}

sft_judge_dir_for_temp() {
  local temp="$1"
  local tag
  tag="$(temp_tag "$temp")"
  echo "$SWEEP_ROOT/ds4_sft_${tag}_fullswap"
}

is_complete_sft_generation() {
  local path="$1"
  local manifest="$2"
  local temp="$3"
  python - "$path" "$manifest" "$temp" "$TEST_MAX_ROWS" "$MAX_NEW_TOKENS" "$TOP_P" "$TOP_K" <<'PY'
import json
import sys
from pathlib import Path

path_s, manifest_s, temp_s, rows_s, max_new_s, top_p_s, top_k_s = sys.argv[1:]
path = Path(path_s)
manifest = Path(manifest_s)
expected_rows = int(rows_s)
temp = float(temp_s)
expected_do_sample = temp > 0.0
if not path.exists() or path.stat().st_size == 0 or not manifest.exists():
    raise SystemExit(1)
records = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
if len(records) != expected_rows:
    raise SystemExit(1)
sample_ids = [str(row.get("sample_id", "")) for row in records]
if len(set(sample_ids)) != expected_rows or any(not sid for sid in sample_ids):
    raise SystemExit(1)
meta = json.loads(manifest.read_text(encoding="utf-8"))
checks = [
    int(meta.get("completed_rows", -1)) == expected_rows,
    int(meta.get("max_new_tokens", -1)) == int(max_new_s),
    bool(meta.get("do_sample")) == expected_do_sample,
    abs(float(meta.get("temperature", -999.0)) - temp) < 1e-9,
    abs(float(meta.get("top_p", -999.0)) - float(top_p_s)) < 1e-9,
    int(meta.get("top_k", -999)) == int(top_k_s),
    str(meta.get("actuator_path", "")) == "",
    str(meta.get("head_actuator_path", "")) == "",
]
if not all(checks):
    raise SystemExit(1)
PY
}

judge_complete() {
  local out_dir="$1"
  python - "$out_dir" "$TEST_MAX_ROWS" "$REVIEW_FRACTION" <<'PY'
import csv
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
expected_rows = int(sys.argv[2])
review_fraction = float(sys.argv[3])
manifest_path = out_dir / "judge_manifest.json"
summary_path = out_dir / "pairwise_summary.csv"
first_path = out_dir / "judge_first_pass.jsonl"
review_path = out_dir / "judge_order_swap_review.jsonl"
if not manifest_path.exists() or not summary_path.exists() or not first_path.exists():
    raise SystemExit(1)
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if not bool(manifest.get("human_only", False)):
    raise SystemExit(1)
if int(manifest.get("sample_rows", -1)) != expected_rows:
    raise SystemExit(1)
if int(manifest.get("judge_packets", -1)) != expected_rows:
    raise SystemExit(1)
with summary_path.open("r", encoding="utf-8-sig", newline="") as stream:
    rows = list(csv.DictReader(stream))
if len(rows) != 1:
    raise SystemExit(1)
row = rows[0]
if row.get("right") != "human" or not str(row.get("left", "")).startswith("sft_"):
    raise SystemExit(1)
if int(row.get("n", -1)) != expected_rows:
    raise SystemExit(1)
first_lines = sum(1 for line in first_path.read_text(encoding="utf-8-sig").splitlines() if line.strip())
if first_lines < expected_rows:
    raise SystemExit(1)
if review_fraction >= 0.999:
    if not review_path.exists():
        raise SystemExit(1)
    review_lines = sum(1 for line in review_path.read_text(encoding="utf-8-sig").splitlines() if line.strip())
    if review_lines < expected_rows:
        raise SystemExit(1)
PY
}

generate_sft_for_temp() {
  local temp="$1"
  local tag path manifest out_dir sample_flag batch
  tag="$(temp_tag "$temp")"
  path="$(sft_path_for_temp "$temp")"
  manifest="$(sft_manifest_for_temp "$temp")"
  if is_complete_sft_generation "$path" "$manifest" "$temp"; then
    echo "[sft-sweep] reuse $tag path=$path"
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
  echo "[sft-sweep] generate $tag temp=$temp gpu=$GPU out=$path"
  CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.run_imdb_sentiment_actuator_generation \
    --model "$MODEL" \
    --prompts-jsonl "$PROMPTS_JSONL" \
    --out-jsonl "$path" \
    --manifest-json "$manifest" \
    --control-name "sft_${tag}" \
    --alpha-sweep 0.0 \
    --generation-apply-mode all \
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

run_judge_for_temp() {
  local temp="$1"
  local tag path manifest out_dir
  tag="$(temp_tag "$temp")"
  path="$(sft_path_for_temp "$temp")"
  manifest="$(sft_manifest_for_temp "$temp")"
  out_dir="$(sft_judge_dir_for_temp "$temp")"
  if judge_complete "$out_dir"; then
    echo "[sft-sweep] reuse judge $tag out=$out_dir"
    return 0
  fi
  if ! is_complete_sft_generation "$path" "$manifest" "$temp"; then
    echo "[sft-sweep] sft generation incomplete for $tag: $path" >&2
    return 1
  fi
  mkdir -p "$out_dir"
  if [[ -z "${DEEPSEEK_API_KEY:-}" && -f "$DEEPSEEK_API_KEY_FILE" ]]; then
    export DEEPSEEK_API_KEY
    DEEPSEEK_API_KEY="$(<"$DEEPSEEK_API_KEY_FILE")"
  fi
  echo "[sft-sweep] judge $tag out=$out_dir"
  python -m screscomp.cli.evaluate_tldr_dpo_ds4 \
    --prompts-jsonl "$JUDGE_PROMPTS_JSONL" \
    --candidate "sft_${tag}=$path" \
    --out-dir "$out_dir" \
    --expected-rows "$TEST_MAX_ROWS" \
    --concurrency "$JUDGE_CONCURRENCY" \
    --review-fraction "$REVIEW_FRACTION" \
    --model "$JUDGE_MODEL" \
    --thinking-mode "$THINKING_MODE" \
    --human-only \
    --allow-batch-seed-mismatch \
    > "$out_dir/judge.log" 2>&1
}

if [[ "$RUN_GENERATION" == "1" ]]; then
  for temp in $TEMPS; do
    tag="$(temp_tag "$temp")"
    echo "[sft-sweep] ===== generate $tag temp=$temp ====="
    generate_sft_for_temp "$temp"
  done
fi

if [[ "$RUN_JUDGE" == "1" ]]; then
  for temp in $TEMPS; do
    tag="$(temp_tag "$temp")"
    echo "[sft-sweep] ===== judge $tag temp=$temp ====="
    run_judge_for_temp "$temp"
  done
fi

python scripts/audit_tldr_text_health.py \
  --out-root "$OUT_ROOT" \
  --sweep-root "$SWEEP_ROOT" \
  --methods sft \
  --temps $TEMPS \
  --out-csv "$SWEEP_ROOT/sft_text_health_template_metrics.csv" \
  --templates-json "$SWEEP_ROOT/sft_text_health_template_examples.json"

echo "[sft-sweep] done"
