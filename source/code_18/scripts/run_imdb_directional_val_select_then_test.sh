#!/usr/bin/env bash
set -euo pipefail

# Locked IMDb evaluation:
#   1) measure the full pre-registered validation response curve
#   2) evaluate baseline + the pre-registered alpha on held-out test prompts

cd LOCAL_HOME/RPEC/projects/screscomp

GPU="${GPU:-0}"
MODEL="edbeeching/gpt2-large-imdb"
EVENT="imdb_positive_sentiment"
CONDA_ENV="${CONDA_ENV:-screscomp}"

TRAIN_ROOT="${TRAIN_ROOT:?TRAIN_ROOT is required by the locked evaluator}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT is required by the locked evaluator}"
VAL_ENV_DIR="${VAL_ENV_DIR:?VAL_ENV_DIR is required by the locked evaluator}"
TEST_ENV_DIR="${TEST_ENV_DIR:?TEST_ENV_DIR is required by the locked evaluator}"

EVAL_GROUPS="mlp_positive,mlp_negative,att_positive,att_negative,full_positive,full_negative"
ALPHA_SWEEP="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"
METRIC="mean_positive_sentiment_score"
ALPHA_KL_FIELD="mean_sequence_kl"
FIXED_ALPHA="0.3"
SHARED_BASE_CONTROL_NAME="shared_base"

VAL_ROWS=1024
TEST_ROWS=2048
SEED=42

EVAL_SPLIT="eval"
EVAL_SAMPLES_PER_PROMPT=1
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-16}"
MAX_NEW_TOKENS=256
DO_SAMPLE=1
TEMPERATURE=1.0
TOP_P=1.0
TOP_K=50
SAME_SEED_ACROSS_ALPHA=1

EVAL_GENERATION_APPLY_MODE="all"
EVAL_COMPONENT_APPLY_MODE="prefill"
EVAL_HEAD_APPLY_MODE="all"

SCORER_MODEL="siebert/sentiment-roberta-large-english"
SCORER_DEVICE="${SCORER_DEVICE:-0}"
SCORE_TEXT="completion"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-16}"
SCORER_MAX_LENGTH=512
SCORE_STREAM_SUMMARY_EVERY_BATCHES="${SCORE_STREAM_SUMMARY_EVERY_BATCHES:-16}"
KL_STREAM_SUMMARY_EVERY_ROWS="${KL_STREAM_SUMMARY_EVERY_ROWS:-32}"
KL_BATCH_SIZE="${KL_BATCH_SIZE:-16}"
RESUME="${RESUME:-1}"

TORCH_DTYPE="bfloat16"
DEVICE="${DEVICE:-cuda}"

mkdir -p "$OUT_ROOT"
STATUS="$OUT_ROOT/status.tsv"
MASTER_LOG="$OUT_ROOT/master.log"

if [[ -f LOCAL_HOME/anaconda3/etc/profile.d/conda.sh && -z "${SKIP_CONDA:-}" ]]; then
  # shellcheck disable=SC1091
  source LOCAL_HOME/anaconda3/etc/profile.d/conda.sh
  conda activate "$CONDA_ENV"
fi
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

{
  echo -e "time\tstage\tstatus"
  echo -e "$(date -Is)\tinit\tstarting"
} > "$STATUS"

echo "[$(date -Is)] val-select-then-test train_root=$TRAIN_ROOT out_root=$OUT_ROOT" | tee -a "$MASTER_LOG"
echo "[$(date -Is)] groups=$EVAL_GROUPS alpha_sweep=$ALPHA_SWEEP val_rows=$VAL_ROWS test_rows=$TEST_ROWS fixed_alpha=$FIXED_ALPHA shared_base=required" | tee -a "$MASTER_LOG"

run_kl_with_backoff() {
  local out_dir="$1"
  shift
  local first_attempt="$1"
  shift
  local kl_args=("$@")
  kl_args+=(--batch-size "$KL_BATCH_SIZE")
  if [[ "$first_attempt" == "1" ]]; then
    kl_args+=(--overwrite)
  else
    kl_args+=(--resume)
  fi
  echo "[$(date -Is)] kl-run out=$out_dir batch_size=$KL_BATCH_SIZE mode=$(if [[ "$first_attempt" == "1" ]]; then echo overwrite; else echo resume; fi)" | tee -a "$MASTER_LOG"
  CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.compute_imdb_generation_kl "${kl_args[@]}" > "$out_dir/kl.log" 2>&1
}

filter_nonzero_alphas() {
  local alpha_sweep="$1"
  python - "$alpha_sweep" <<'PY'
import sys

items = []
for raw in str(sys.argv[1]).split(","):
    text = raw.strip()
    if not text:
        continue
    try:
        value = float(text)
    except Exception:
        items.append(text)
        continue
    if abs(value) > 1e-12:
        items.append(text)
print(",".join(items))
PY
}

run_shared_base() {
  local prompts_jsonl="$1"
  local out_dir="$2"
  mkdir -p "$out_dir"
  local gen_jsonl="$out_dir/generations.jsonl"
  local scored_jsonl="$out_dir/scored_generations.jsonl"
  local scored_csv="$out_dir/scored_generations.csv"
  local summary_csv="$out_dir/score_summary.csv"
  local score_manifest="$out_dir/score_manifest.json"
  local kl_jsonl="$out_dir/kl_scored_generations.jsonl"
  local kl_csv="$out_dir/kl_scored_generations.csv"
  local kl_manifest="$out_dir/kl_manifest.json"
  local kl_progress="$out_dir/kl_progress.json"

  if [[ "$RESUME" == "1" && -f "$score_manifest" && -f "$kl_manifest" && -f "$gen_jsonl" && -f "$scored_jsonl" && -f "$kl_jsonl" ]]; then
    echo "[$(date -Is)] resume-skip shared-base out=$out_dir" | tee -a "$MASTER_LOG"
    return 0
  fi

  if [[ "$RESUME" != "1" || ! -f "$score_manifest" || ! -f "$gen_jsonl" || ! -f "$scored_jsonl" ]]; then
    gen_args=(
      python -m screscomp.cli.run_imdb_sentiment_actuator_generation
      --model "$MODEL"
      --prompts-jsonl "$prompts_jsonl"
      --out-jsonl "$gen_jsonl"
      --control-name "$SHARED_BASE_CONTROL_NAME"
      --alpha-sweep "0"
      --generation-apply-mode "$EVAL_GENERATION_APPLY_MODE"
      --component-apply-mode "$EVAL_COMPONENT_APPLY_MODE"
      --head-apply-mode "$EVAL_HEAD_APPLY_MODE"
      --split "$EVAL_SPLIT"
      --samples-per-prompt "$EVAL_SAMPLES_PER_PROMPT"
      --generation-batch-size "$GENERATION_BATCH_SIZE"
      --max-new-tokens "$MAX_NEW_TOKENS"
      --temperature "$TEMPERATURE"
      --top-p "$TOP_P"
      --top-k "$TOP_K"
      --seed "$SEED"
      --same-seed-across-alpha
      --torch-dtype "$TORCH_DTYPE"
      --device "$DEVICE"
    )
    if [[ "$RESUME" != "1" ]]; then
      gen_args+=(--overwrite)
    fi
    CUDA_VISIBLE_DEVICES="$GPU" "${gen_args[@]}" > "$out_dir/generate.log" 2>&1

    CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.score_imdb_sentiment_generations \
      --input "$gen_jsonl" \
      --out-jsonl "$scored_jsonl" \
      --out-csv "$scored_csv" \
      --summary-csv "$summary_csv" \
      --progress-json "$out_dir/score_progress.json" \
      --score-text "$SCORE_TEXT" \
      --batch-size "$SCORE_BATCH_SIZE" \
      --scorer-model "$SCORER_MODEL" \
      --scorer-max-length "$SCORER_MAX_LENGTH" \
      --stream-summary-every-batches "$SCORE_STREAM_SUMMARY_EVERY_BATCHES" \
      --overwrite \
      --device "$SCORER_DEVICE" \
      > "$out_dir/score.log" 2>&1
  fi

  shared_kl_args=(
    --input-jsonl "$scored_jsonl"
    --out-jsonl "$kl_jsonl"
    --out-csv "$kl_csv"
    --summary-csv "$summary_csv"
    --progress-json "$kl_progress"
    --model "$MODEL"
    --torch-dtype "$TORCH_DTYPE"
    --device "$DEVICE"
    --stream-summary-every-rows "$KL_STREAM_SUMMARY_EVERY_ROWS"
  )
  if [[ "$RESUME" == "1" ]]; then
    run_kl_with_backoff "$out_dir" 0 "${shared_kl_args[@]}"
  else
    run_kl_with_backoff "$out_dir" 1 "${shared_kl_args[@]}"
  fi
}

merge_group_rows_with_shared_base() {
  local shared_jsonl="$1"
  local nonzero_jsonl="$2"
  local control_name="$3"
  local out_jsonl="$4"
  python - "$shared_jsonl" "$nonzero_jsonl" "$control_name" "$out_jsonl" <<'PY'
import json
import sys
from pathlib import Path

shared_path = Path(sys.argv[1])
nonzero_path = Path(sys.argv[2])
control_name = sys.argv[3]
out_path = Path(sys.argv[4])

def rows(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", encoding="utf-8") as out:
    for row in rows(shared_path):
        row["control_name"] = control_name
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
    for row in rows(nonzero_path):
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
PY
}

seed_group_kl_from_shared_base() {
  local shared_kl_jsonl="$1"
  local control_name="$2"
  local out_jsonl="$3"
  if [[ "$RESUME" == "1" && -s "$out_jsonl" ]]; then
    return 0
  fi
  merge_group_rows_with_shared_base "$shared_kl_jsonl" "/dev/null" "$control_name" "$out_jsonl"
}

check_shared_base_consistency() {
  local eval_root="$1"
  local label="$2"
  python - "$eval_root" "$label" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
label = sys.argv[2]
group_dirs = sorted(path for path in root.iterdir() if path.is_dir() and (path / "generations.jsonl").exists())
if len(group_dirs) <= 1:
    print(f"[shared-base-check] skip {label}: groups={len(group_dirs)}", flush=True)
    raise SystemExit(0)

reference_group = None
reference_rows = None
for group_dir in group_dirs:
    rows = {}
    with (group_dir / "generations.jsonl").open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            try:
                alpha = float(row.get("alpha", 0.0))
            except Exception:
                alpha = 0.0
            if abs(alpha) > 1e-12:
                continue
            key = (str(row.get("sample_id", "")), int(row.get("sample_index", 0)))
            rows[key] = (
                str(row.get("prompt", "")),
                str(row.get("completion", "")),
                str(row.get("full_text", "")),
            )
    if reference_rows is None:
        reference_group = group_dir.name
        reference_rows = rows
        continue
    if rows.keys() != reference_rows.keys():
        missing = sorted(reference_rows.keys() - rows.keys())[:5]
        extra = sorted(rows.keys() - reference_rows.keys())[:5]
        raise SystemExit(
            f"[shared-base-check] {label} mismatch keys: ref={reference_group} group={group_dir.name} "
            f"missing={missing} extra={extra}"
        )
    for key, reference_value in reference_rows.items():
        current_value = rows[key]
        if current_value != reference_value:
            raise SystemExit(
                f"[shared-base-check] {label} mismatch: ref={reference_group} group={group_dir.name} "
                f"sample_id={key[0]} sample_index={key[1]}"
            )
print(f"[shared-base-check] ok {label}: groups={len(group_dirs)} rows={len(reference_rows or {})}", flush=True)
PY
}

run_eval_group() {
  local prompts_jsonl="$1"
  local control_name="$2"
  local alpha_sweep="$3"
  local out_dir="$4"
  local component_actuator="$5"
  local head_actuator="$6"
  local shared_base_dir="${7:-}"
  mkdir -p "$out_dir"
  local gen_jsonl="$out_dir/generations.jsonl"
  local scored_jsonl="$out_dir/scored_generations.jsonl"
  local scored_csv="$out_dir/scored_generations.csv"
  local summary_csv="$out_dir/score_summary.csv"
  local score_manifest="$out_dir/score_manifest.json"
  local nonzero_gen_jsonl="$out_dir/nonzero_generations.jsonl"
  local nonzero_scored_jsonl="$out_dir/nonzero_scored_generations.jsonl"
  local nonzero_scored_csv="$out_dir/nonzero_scored_generations.csv"
  local nonzero_summary_csv="$out_dir/nonzero_score_summary.csv"
  local kl_jsonl="$out_dir/kl_scored_generations.jsonl"
  local kl_csv="$out_dir/kl_scored_generations.csv"
  local kl_manifest="$out_dir/kl_manifest.json"
  local kl_progress="$out_dir/kl_progress.json"

  if [[ -z "$shared_base_dir" ]]; then
    echo "Shared base directory missing for $control_name" >&2
    exit 2
  fi
  run_shared_base "$prompts_jsonl" "$shared_base_dir"

  local need_generate_score="1"
  if [[ "$RESUME" == "1" && -f "$score_manifest" && -f "$gen_jsonl" && -f "$scored_jsonl" ]]; then
    need_generate_score="0"
    echo "[$(date -Is)] resume-skip score group=$control_name out=$out_dir" | tee -a "$MASTER_LOG"
  fi

  if [[ "$need_generate_score" == "1" ]]; then
    local nonzero_alpha_sweep
    nonzero_alpha_sweep="$(filter_nonzero_alphas "$alpha_sweep")"
    if [[ -n "$nonzero_alpha_sweep" ]]; then
      gen_args=(
        python -m screscomp.cli.run_imdb_sentiment_actuator_generation
        --model "$MODEL"
        --prompts-jsonl "$prompts_jsonl"
        --out-jsonl "$nonzero_gen_jsonl"
        --control-name "$control_name"
        --alpha-sweep "$nonzero_alpha_sweep"
        --generation-apply-mode "$EVAL_GENERATION_APPLY_MODE"
        --component-apply-mode "$EVAL_COMPONENT_APPLY_MODE"
        --head-apply-mode "$EVAL_HEAD_APPLY_MODE"
        --split "$EVAL_SPLIT"
        --samples-per-prompt "$EVAL_SAMPLES_PER_PROMPT"
        --generation-batch-size "$GENERATION_BATCH_SIZE"
        --max-new-tokens "$MAX_NEW_TOKENS"
        --temperature "$TEMPERATURE"
        --top-p "$TOP_P"
        --top-k "$TOP_K"
        --seed "$SEED"
        --same-seed-across-alpha
        --torch-dtype "$TORCH_DTYPE"
        --device "$DEVICE"
      )
      if [[ -n "$component_actuator" ]]; then
        gen_args+=(--actuator "$component_actuator")
      fi
      if [[ -n "$head_actuator" ]]; then
        gen_args+=(--head-actuator "$head_actuator")
      fi
      if [[ "$RESUME" != "1" ]]; then
        gen_args+=(--overwrite)
      fi
      CUDA_VISIBLE_DEVICES="$GPU" "${gen_args[@]}" > "$out_dir/generate.log" 2>&1

      CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.score_imdb_sentiment_generations \
        --input "$nonzero_gen_jsonl" \
        --out-jsonl "$nonzero_scored_jsonl" \
        --out-csv "$nonzero_scored_csv" \
        --summary-csv "$nonzero_summary_csv" \
        --progress-json "$out_dir/score_progress.json" \
        --score-text "$SCORE_TEXT" \
        --batch-size "$SCORE_BATCH_SIZE" \
        --scorer-model "$SCORER_MODEL" \
        --scorer-max-length "$SCORER_MAX_LENGTH" \
        --stream-summary-every-batches "$SCORE_STREAM_SUMMARY_EVERY_BATCHES" \
        --overwrite \
        --device "$SCORER_DEVICE" \
        > "$out_dir/score.log" 2>&1
    else
      : > "$out_dir/generate.log"
      : > "$nonzero_gen_jsonl"
      : > "$nonzero_scored_jsonl"
    fi
    merge_group_rows_with_shared_base "$shared_base_dir/generations.jsonl" "$nonzero_gen_jsonl" "$control_name" "$gen_jsonl"
    merge_group_rows_with_shared_base "$shared_base_dir/scored_generations.jsonl" "$nonzero_scored_jsonl" "$control_name" "$scored_jsonl"
  fi
  seed_group_kl_from_shared_base "$shared_base_dir/kl_scored_generations.jsonl" "$control_name" "$kl_jsonl"

  kl_args=(
    --input-jsonl "$scored_jsonl"
    --out-jsonl "$kl_jsonl"
    --out-csv "$kl_csv"
    --summary-csv "$summary_csv"
    --progress-json "$kl_progress"
    --model "$MODEL"
    --torch-dtype "$TORCH_DTYPE"
    --device "$DEVICE"
    --stream-summary-every-rows "$KL_STREAM_SUMMARY_EVERY_ROWS"
  )
  # Group KL is always resumed from the single shared-base KL cache.
  run_kl_with_backoff "$out_dir" 0 "${kl_args[@]}"
}

resolve_group_paths() {
  local group="$1"
  case "$group" in
    mlp_positive)
      echo "$TRAIN_ROOT/train_mlp_positive/fixed_actuator.pt|"
      ;;
    mlp_negative)
      echo "$TRAIN_ROOT/train_mlp_negative/fixed_actuator.pt|"
      ;;
    att_positive)
      echo "|$TRAIN_ROOT/train_head_positive/head_actuator.pt"
      ;;
    att_negative)
      echo "|$TRAIN_ROOT/train_head_negative/head_actuator.pt"
      ;;
    full_positive)
      echo "$TRAIN_ROOT/train_mlp_positive/fixed_actuator.pt|$TRAIN_ROOT/train_head_positive/head_actuator.pt"
      ;;
    full_negative)
      echo "$TRAIN_ROOT/train_mlp_negative/fixed_actuator.pt|$TRAIN_ROOT/train_head_negative/head_actuator.pt"
      ;;
    *)
      echo "Unknown group: $group" >&2
      exit 2
      ;;
  esac
}

test -f "$VAL_ENV_DIR/prompts.jsonl" || { echo "Missing locked validation prompts: $VAL_ENV_DIR/prompts.jsonl" >&2; exit 2; }
test -f "$TEST_ENV_DIR/prompts.jsonl" || { echo "Missing locked test prompts: $TEST_ENV_DIR/prompts.jsonl" >&2; exit 2; }

BEST_ALPHA_CSV="$OUT_ROOT/best_alpha_by_group.csv"
VAL_SHARED_BASE_DIR="$OUT_ROOT/val_select/shared_base"
TEST_SHARED_BASE_DIR="$OUT_ROOT/test_eval/shared_base"

IFS=',' read -r -a GROUP_LIST <<< "$EVAL_GROUPS"
if [[ "$RESUME" != "1" || ! -f "$BEST_ALPHA_CSV" ]]; then
  echo -e "$(date -Is)\tval_select\trunning" >> "$STATUS"
  for raw_group in "${GROUP_LIST[@]}"; do
    group="$(echo "$raw_group" | xargs)"
    [[ -z "$group" ]] && continue
    pair="$(resolve_group_paths "$group")"
    component_actuator="${pair%%|*}"
    head_actuator="${pair#*|}"
    run_eval_group "$VAL_ENV_DIR/prompts.jsonl" "$group" "$ALPHA_SWEEP" "$OUT_ROOT/val_select/$group" "$component_actuator" "$head_actuator" "$VAL_SHARED_BASE_DIR"
  done
  check_shared_base_consistency "$OUT_ROOT/val_select" "val_select"
  python -m screscomp.cli.summarize_imdb_sentiment_matrix \
    --root "$OUT_ROOT/val_select" \
    --out-csv "$OUT_ROOT/val_select/matrix_score_summary.csv" \
    > "$OUT_ROOT/val_select/summarize.log" 2>&1
  python - "$OUT_ROOT/val_select/matrix_score_summary.csv" "$BEST_ALPHA_CSV" "$EVAL_GROUPS" "$FIXED_ALPHA" "$METRIC" "$ALPHA_KL_FIELD" <<'PY'
import csv
import sys
from pathlib import Path

matrix_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
groups = [item.strip() for item in sys.argv[3].split(",") if item.strip()]
fixed_alpha = float(sys.argv[4])
metric = sys.argv[5]
kl_field = sys.argv[6]
with matrix_path.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))
by_key = {}
for row in rows:
    try:
        alpha = round(float(row.get("alpha", "0") or 0.0), 8)
    except Exception:
        continue
    by_key[(str(row.get("control_name", "")), alpha)] = row
out_rows = []
for group in groups:
    base = by_key.get((group, 0.0))
    chosen = by_key.get((group, round(fixed_alpha, 8)))
    if base is None or chosen is None:
        raise SystemExit(f"missing fixed-alpha summary for group={group} alpha={fixed_alpha}")
    base_metric = float(base.get(metric, "0") or 0.0)
    chosen_metric = float(chosen.get(metric, "0") or 0.0)
    out_rows.append(
        {
            "control_name": group,
            "best_alpha": fixed_alpha,
            "best_metric": chosen_metric,
            "baseline_metric": base_metric,
            "delta_vs_baseline": chosen_metric - base_metric,
            "metric": metric,
            "selection_mode": "fixed_preregistered",
            "reward_ratio": "",
            "kl_field": kl_field,
            "best_kl": float(chosen.get(kl_field, "0") or 0.0),
            "split": chosen.get("split", ""),
            "baseline_n": base.get("n", ""),
            "best_n": chosen.get("n", ""),
        }
    )
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(out_rows[0]))
    writer.writeheader()
    writer.writerows(out_rows)
PY
  echo -e "$(date -Is)\tval_select\tdone" >> "$STATUS"
else
  echo "[$(date -Is)] resume-skip val_select existing_best_alpha=$BEST_ALPHA_CSV" | tee -a "$MASTER_LOG"
fi

if [[ "$RESUME" != "1" || ! -f "$OUT_ROOT/test_eval/matrix_score_summary.csv" ]]; then
  echo -e "$(date -Is)\ttest_eval\trunning" >> "$STATUS"
  for raw_group in "${GROUP_LIST[@]}"; do
    group="$(echo "$raw_group" | xargs)"
    [[ -z "$group" ]] && continue
    best_alpha="$FIXED_ALPHA"
    alpha_sweep="0"
    if [[ "$best_alpha" != "0" && "$best_alpha" != "0.0" ]]; then
      alpha_sweep="0,$best_alpha"
    fi
    pair="$(resolve_group_paths "$group")"
    component_actuator="${pair%%|*}"
    head_actuator="${pair#*|}"
    run_eval_group "$TEST_ENV_DIR/prompts.jsonl" "$group" "$alpha_sweep" "$OUT_ROOT/test_eval/$group" "$component_actuator" "$head_actuator" "$TEST_SHARED_BASE_DIR"
  done
  check_shared_base_consistency "$OUT_ROOT/test_eval" "test_eval"
  python -m screscomp.cli.summarize_imdb_sentiment_matrix \
    --root "$OUT_ROOT/test_eval" \
    --out-csv "$OUT_ROOT/test_eval/matrix_score_summary.csv" \
    > "$OUT_ROOT/test_eval/summarize.log" 2>&1
  echo -e "$(date -Is)\ttest_eval\tdone" >> "$STATUS"
else
  echo "[$(date -Is)] resume-skip test_eval existing_summary=$OUT_ROOT/test_eval/matrix_score_summary.csv" | tee -a "$MASTER_LOG"
fi

echo -e "$(date -Is)\tall\tdone" >> "$STATUS"
echo "[$(date -Is)] val-select-then-test done out=$OUT_ROOT" | tee -a "$MASTER_LOG"
