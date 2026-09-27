#!/usr/bin/env bash
set -euo pipefail

# Open-generation site-specificity evaluation for the post-O PM site run.
# This script reuses actuators produced by 07_run_post_o_pm_site_shift.sh.
# It does not retrain; it compares natural-distribution generation under:
#   base, selected-site actuator, shifted-site same-vector, and shifted-site retrain.
# Native-write zeroing belongs to the separate causal-substitution protocol, not site testing.

export LC_ALL=C.UTF-8
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

REPO_ROOT="${REPO_ROOT:-LOCAL_HOME/RPEC/projects/screscomp}"
cd "$REPO_ROOT"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

PY="${PY:-LOCAL_HOME/anaconda3/envs/screscomp/bin/python}"
SITE_RUN_ROOT="${SITE_RUN_ROOT:?Set SITE_RUN_ROOT to the 07_run_post_o_pm_site_shift.sh output root}"
RUN_ROOT="${RUN_ROOT:-/dev/shm/screscomp_runs/post_o_pm_site_open_generation_$(date -u +%Y%m%d_%H%M%S)}"
GPU="${GPU:-0}"
TASKS="${TASKS:-imdb qa mr mc}"
FAMILIES="${FAMILIES:-mlp_pm head_pm}"
OPEN_SHIFTS="${OPEN_SHIFTS:-1}"
ALPHA_SWEEP="${ALPHA_SWEEP:-0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}"
BASELINE_ONCE="${BASELINE_ONCE:-${ZERO_BASELINE_ONCE:-1}}"
INCLUDE_BASE="${INCLUDE_BASE:-1}"
INCLUDE_SELECTED="${INCLUDE_SELECTED:-1}"

IMDB_ROOT="${IMDB_ROOT:-/dev/shm/screscomp_runs/imdb_ma921_expanded1024_sampled_paperkl_20260626_055222}"
IMDB_MODEL="${IMDB_MODEL:-LOCAL_HOME/.cache/huggingface/hub/models--ma921--gpt2-large-sft-imdb/snapshots/f480190690d5abfc0e003ccb4f7e650626019bb9}"
IMDB_EVAL_SPLIT="${IMDB_EVAL_SPLIT:-eval}"
IMDB_PROMPTS_SPLIT="${IMDB_PROMPTS_SPLIT:-test}"
IMDB_PROMPTS_JSONL="${IMDB_PROMPTS_JSONL:-$IMDB_ROOT/eval_env/$IMDB_PROMPTS_SPLIT/prompts.jsonl}"
IMDB_EVAL_ROWS="${IMDB_EVAL_ROWS:-2048}"
IMDB_GEN_BATCH="${IMDB_GEN_BATCH:-32}"
IMDB_SCORE_BATCH="${IMDB_SCORE_BATCH:-32}"
IMDB_KL_BATCH="${IMDB_KL_BATCH:-32}"
IMDB_SCORER_MODEL="${IMDB_SCORER_MODEL:-LOCAL_HOME/.cache/huggingface/hub/models--siebert--sentiment-roberta-large-english/snapshots/74cea614e245b0832c770ec9aa51bd58df965b9c}"

CONFIQA_MODEL="${CONFIQA_MODEL:-LOCAL_HOME/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct}"
FETCH_DIR="${FETCH_DIR:-data_ckplug/context_dpo_confiqa_official}"
CONFIQA_EVAL_SOURCE_START="${CONFIQA_EVAL_SOURCE_START:-1000}"
CONFIQA_EVAL_SOURCE_ROWS="${CONFIQA_EVAL_SOURCE_ROWS:-}"
CONFIQA_EVAL_ROWS="${CONFIQA_EVAL_ROWS:-}"
CONFIQA_QA_EVAL_SOURCE_ROWS="${CONFIQA_QA_EVAL_SOURCE_ROWS:-${CONFIQA_EVAL_SOURCE_ROWS:-4997}}"
CONFIQA_QA_EVAL_ROWS="${CONFIQA_QA_EVAL_ROWS:-${CONFIQA_EVAL_ROWS:-4997}}"
CONFIQA_MR_EVAL_SOURCE_ROWS="${CONFIQA_MR_EVAL_SOURCE_ROWS:-${CONFIQA_EVAL_SOURCE_ROWS:-500}}"
CONFIQA_MR_EVAL_ROWS="${CONFIQA_MR_EVAL_ROWS:-${CONFIQA_EVAL_ROWS:-500}}"
CONFIQA_MC_EVAL_SOURCE_ROWS="${CONFIQA_MC_EVAL_SOURCE_ROWS:-${CONFIQA_EVAL_SOURCE_ROWS:-500}}"
CONFIQA_MC_EVAL_ROWS="${CONFIQA_MC_EVAL_ROWS:-${CONFIQA_EVAL_ROWS:-500}}"
CONFIQA_EVAL_SPLIT="${CONFIQA_EVAL_SPLIT:-all}"
CONFIQA_MAX_NEW_TOKENS="${CONFIQA_MAX_NEW_TOKENS:-64}"

export CUDA_VISIBLE_DEVICES="$GPU"

mkdir -p "$RUN_ROOT/logs"
STATUS="$RUN_ROOT/status.tsv"
SUMMARY="$RUN_ROOT/open_generation_summary.tsv"
MASTER_LOG="$RUN_ROOT/open_generation.log"
printf 'time\tstage\tstatus\n' > "$STATUS"
printf 'task\tfamily\tcontrol_name\tn\tmetric1\tmetric2\tmetric3\tmean_chars\tout_dir\n' > "$SUMMARY"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$MASTER_LOG"
}

stage() {
  printf '%s\t%s\t%s\n' "$(date -Is)" "$1" "$2" >> "$STATUS"
  log "stage=$1 $2"
}

contains_word() {
  local haystack="$1"
  local needle="$2"
  for item in $haystack; do
    [[ "$item" == "$needle" ]] && return 0
  done
  return 1
}

family_apply_mode() {
  case "$1" in
    mlp_pm) echo prefill ;;
    head_pm) echo all ;;
    *) echo "unknown family: $1" >&2; exit 2 ;;
  esac
}

control_alpha_sweep() {
  if [[ "$BASELINE_ONCE" == "1" ]]; then
    echo "$ALPHA_SWEEP"
  else
    echo "0,$ALPHA_SWEEP"
  fi
}

data_json_for_task() {
  case "$1" in
    qa) echo "$FETCH_DIR/ConFiQA-QA.json" ;;
    mr) echo "$FETCH_DIR/ConFiQA-MR.json" ;;
    mc) echo "$FETCH_DIR/ConFiQA-MC.json" ;;
    *) echo "unsupported ConFiQA task: $1" >&2; exit 2 ;;
  esac
}

confiqa_source_rows_for_task() {
  case "$1" in
    qa) echo "$CONFIQA_QA_EVAL_SOURCE_ROWS" ;;
    mr) echo "$CONFIQA_MR_EVAL_SOURCE_ROWS" ;;
    mc) echo "$CONFIQA_MC_EVAL_SOURCE_ROWS" ;;
    *) echo "unsupported ConFiQA task: $1" >&2; exit 2 ;;
  esac
}

confiqa_eval_rows_for_task() {
  case "$1" in
    qa) echo "$CONFIQA_QA_EVAL_ROWS" ;;
    mr) echo "$CONFIQA_MR_EVAL_ROWS" ;;
    mc) echo "$CONFIQA_MC_EVAL_ROWS" ;;
    *) echo "unsupported ConFiQA task: $1" >&2; exit 2 ;;
  esac
}

ensure_confiqa_eval_rows() {
  local short="$1"
  local task_out="$RUN_ROOT/confiqa_$short/common"
  local out_jsonl="$task_out/eval_open_rows.jsonl"
  local data_json
  local source_rows
  data_json="$(data_json_for_task "$short")"
  source_rows="$(confiqa_source_rows_for_task "$short")"
  mkdir -p "$task_out"
  if [[ -f "$out_jsonl" ]]; then
    echo "$out_jsonl"
    return
  fi
  stage "prepare_open_${short}" running >&2
  "$PY" -m screscomp.cli.cecm_fetch_confiqa --out-dir "$FETCH_DIR" --tasks "$short" \
    >"$task_out/fetch.log" 2>&1
  "$PY" -m screscomp.cli.prepare_ckplug_open \
    --dataset confiqa \
    --data_json "$data_json" \
    --out_jsonl "$out_jsonl" \
    --schema base \
    --alias_policy raw \
    --start "$CONFIQA_EVAL_SOURCE_START" \
    --max_rows "$source_rows" \
    >"$task_out/prepare_eval_open.log" 2>&1
  stage "prepare_open_${short}" done >&2
  echo "$out_jsonl"
}

score_imdb_dir() {
  local out_dir="$1"
  "$PY" -m screscomp.cli.score_imdb_sentiment_generations \
    --input "$out_dir/generations.jsonl" \
    --out-jsonl "$out_dir/scored_generations.jsonl" \
    --summary-csv "$out_dir/score_pre_kl_summary.csv" \
    --scorer-model "$IMDB_SCORER_MODEL" \
    --batch-size "$IMDB_SCORE_BATCH" --device 0 --overwrite \
    >"$out_dir/score.log" 2>&1
  "$PY" -m screscomp.cli.compute_imdb_generation_kl \
    --input-jsonl "$out_dir/scored_generations.jsonl" \
    --out-jsonl "$out_dir/kl_scored_generations.jsonl" \
    --summary-csv "$out_dir/score_summary.csv" \
    --model "$IMDB_MODEL" --batch-size "$IMDB_KL_BATCH" --device cuda --torch-dtype bfloat16 \
    --sampled-only --overwrite \
    >"$out_dir/kl.log" 2>&1
}

append_imdb_summary() {
  local family="$1"
  local out_dir="$2"
  TASK_VALUE="imdb" FAMILY_VALUE="$family" OUT_DIR_VALUE="$out_dir" SUMMARY="$SUMMARY" "$PY" - <<'PY'
import csv, os
from pathlib import Path
summary = Path(os.environ["SUMMARY"])
path = Path(os.environ["OUT_DIR_VALUE"]) / "score_summary.csv"
if not path.exists():
    raise SystemExit(f"missing {path}")
with path.open(encoding="utf-8", newline="") as fp, summary.open("a", encoding="utf-8") as out:
    for row in csv.DictReader(fp):
        control = row.get("control_name", "")
        alpha = row.get("alpha", "")
        if alpha != "":
            control = f"{control}_a{alpha.replace('.', 'p')}"
        out.write("\t".join(str(x) for x in [
            os.environ["TASK_VALUE"],
            os.environ["FAMILY_VALUE"],
            control,
            row.get("n", ""),
            row.get("mean_positive_sentiment_score", ""),
            row.get("paper_mean_sampled_sequence_kl", ""),
            row.get("positive_rate_ge_0p5", ""),
            row.get("mean_completion_chars", ""),
            os.environ["OUT_DIR_VALUE"],
        ]) + "\n")
PY
}

run_imdb_family() {
  local family="$1"
  local apply_mode
  apply_mode="$(family_apply_mode "$family")"
  local site_task_root="$SITE_RUN_ROOT/imdb"
  local selected_csv="$site_task_root/site_specs/$family/selected_components.csv"
  local selected_payload="$site_task_root/train/$family/selected/fixed_actuator.pt"
  test -f "$selected_csv"
  test -f "$selected_payload"

  if [[ "$BASELINE_ONCE" == "1" ]]; then
    local base_out_dir="$RUN_ROOT/imdb/$family/base"
    if [[ -f "$base_out_dir/score_summary.csv" ]]; then
      stage "imdb_${family}_base" reused
      append_imdb_summary "$family" "$base_out_dir"
    else
      stage "imdb_${family}_base" running
      mkdir -p "$base_out_dir"
      "$PY" -m screscomp.cli.run_imdb_sentiment_actuator_generation \
        --model "$IMDB_MODEL" \
        --prompts-jsonl "$IMDB_PROMPTS_JSONL" \
        --out-jsonl "$base_out_dir/generations.jsonl" \
        --actuator "$selected_payload" \
        --control-name base \
        --alpha-sweep "0" \
        --generation-apply-mode "$apply_mode" --component-apply-mode "$apply_mode" \
        --split "$IMDB_EVAL_SPLIT" --max-rows "$IMDB_EVAL_ROWS" \
        --samples-per-prompt 1 \
        --generation-batch-size "$IMDB_GEN_BATCH" --max-new-tokens 256 \
        --temperature 1.0 --top-p 1.0 --top-k 50 --seed 42 --same-seed-across-alpha \
        --device cuda --torch-dtype bfloat16 \
        >"$base_out_dir/generate.log" 2>&1
      score_imdb_dir "$base_out_dir"
      append_imdb_summary "$family" "$base_out_dir"
      stage "imdb_${family}_base" done
    fi
  fi

  local controls=()
  if [[ "$INCLUDE_SELECTED" == "1" ]]; then
    controls+=("selected:$selected_payload")
  fi
  for shift in $OPEN_SHIFTS; do
    local same_payload="$site_task_root/train/$family/shift${shift}_same_vector/fixed_actuator.pt"
    local retrained_payload="$site_task_root/train/$family/shift${shift}_retrained/fixed_actuator.pt"
    [[ -f "$same_payload" ]] && controls+=("shift${shift}_same_vector:$same_payload")
    [[ -f "$retrained_payload" ]] && controls+=("shift${shift}_retrained:$retrained_payload")
  done

  for spec in "${controls[@]}"; do
    local name="${spec%%:*}"
    local payload="${spec#*:}"
    local out_dir="$RUN_ROOT/imdb/$family/$name"
    if [[ -f "$out_dir/score_summary.csv" ]]; then
      stage "imdb_${family}_${name}" reused
      append_imdb_summary "$family" "$out_dir"
      continue
    fi
    stage "imdb_${family}_${name}" running
    mkdir -p "$out_dir"
    "$PY" -m screscomp.cli.run_imdb_sentiment_actuator_generation \
      --model "$IMDB_MODEL" \
      --prompts-jsonl "$IMDB_PROMPTS_JSONL" \
      --out-jsonl "$out_dir/generations.jsonl" \
      --actuator "$payload" \
      --control-name "$name" \
      --alpha-sweep "$(control_alpha_sweep)" \
      --generation-apply-mode "$apply_mode" --component-apply-mode "$apply_mode" \
      --split "$IMDB_EVAL_SPLIT" --max-rows "$IMDB_EVAL_ROWS" \
      --samples-per-prompt 1 \
      --generation-batch-size "$IMDB_GEN_BATCH" --max-new-tokens 256 \
      --temperature 1.0 --top-p 1.0 --top-k 50 --seed 42 --same-seed-across-alpha \
      --device cuda --torch-dtype bfloat16 \
      >"$out_dir/generate.log" 2>&1
    score_imdb_dir "$out_dir"
    append_imdb_summary "$family" "$out_dir"
    stage "imdb_${family}_${name}" done
  done
}

control_parts_for_family() {
  local family="$1"
  local site_task_root="$2"
  local apply_mode="$3"
  local controls=()
  if [[ "$INCLUDE_BASE" == "1" ]]; then
    controls+=("base=")
  fi
  if [[ "$INCLUDE_SELECTED" == "1" && "$BASELINE_ONCE" != "1" ]]; then
    controls+=("selected_a0=")
  fi
  if [[ "$INCLUDE_SELECTED" == "1" ]]; then
    for alpha in ${ALPHA_SWEEP//,/ }; do
      controls+=("selected_a${alpha//./p}=comp:selected:${alpha}:${apply_mode}")
    done
  fi
  for shift in $OPEN_SHIFTS; do
    if [[ -f "$site_task_root/train/$family/shift${shift}_same_vector/fixed_actuator.pt" ]]; then
      if [[ "$BASELINE_ONCE" != "1" ]]; then
        controls+=("shift${shift}_same_a0=")
      fi
      for alpha in ${ALPHA_SWEEP//,/ }; do
        controls+=("shift${shift}_same_a${alpha//./p}=comp:shift${shift}_same:${alpha}:${apply_mode}")
      done
    fi
    if [[ -f "$site_task_root/train/$family/shift${shift}_retrained/fixed_actuator.pt" ]]; then
      if [[ "$BASELINE_ONCE" != "1" ]]; then
        controls+=("shift${shift}_retrained_a0=")
      fi
      for alpha in ${ALPHA_SWEEP//,/ }; do
        controls+=("shift${shift}_retrained_a${alpha//./p}=comp:shift${shift}_retrained:${alpha}:${apply_mode}")
      done
    fi
  done
  local joined
  joined="$(IFS=';'; echo "${controls[*]}")"
  echo "$joined"
}

component_actuators_for_family() {
  local family="$1"
  local site_task_root="$2"
  local specs=()
  if [[ "$INCLUDE_SELECTED" == "1" ]]; then
    specs+=("selected=$site_task_root/train/$family/selected/fixed_actuator.pt")
  fi
  for shift in $OPEN_SHIFTS; do
    [[ -f "$site_task_root/train/$family/shift${shift}_same_vector/fixed_actuator.pt" ]] && \
      specs+=("shift${shift}_same=$site_task_root/train/$family/shift${shift}_same_vector/fixed_actuator.pt")
    [[ -f "$site_task_root/train/$family/shift${shift}_retrained/fixed_actuator.pt" ]] && \
      specs+=("shift${shift}_retrained=$site_task_root/train/$family/shift${shift}_retrained/fixed_actuator.pt")
  done
  local joined
  joined="$(IFS=';'; echo "${specs[*]}")"
  echo "$joined"
}

append_confiqa_summary() {
  local task="$1"
  local family="$2"
  local out_dir="$3"
  TASK_VALUE="$task" FAMILY_VALUE="$family" OUT_DIR_VALUE="$out_dir" SUMMARY="$SUMMARY" "$PY" - <<'PY'
import csv, os
from pathlib import Path
summary = Path(os.environ["SUMMARY"])
path = Path(os.environ["OUT_DIR_VALUE"]) / "generation_summary.csv"
if not path.exists():
    raise SystemExit(f"missing {path}")
def pick(row, *names):
    for name in names:
        value = row.get(name, "")
        if value != "":
            return value
    return ""
with path.open(encoding="utf-8", newline="") as fp, summary.open("a", encoding="utf-8") as out:
    for row in csv.DictReader(fp):
        out.write("\t".join(str(x) for x in [
            os.environ["TASK_VALUE"],
            os.environ["FAMILY_VALUE"],
            row.get("control_name", ""),
            row.get("n", ""),
            pick(row, "pc", "context_hit_rate"),
            pick(row, "em", "exact_match_rate"),
            pick(row, "po", "prior_hit_rate"),
            row.get("mean_chars", ""),
            os.environ["OUT_DIR_VALUE"],
        ]) + "\n")
PY
}

run_confiqa_family() {
  local short="$1"
  local family="$2"
  local task="confiqa_$short"
  local apply_mode
  apply_mode="$(family_apply_mode "$family")"
  local eval_rows
  local task_eval_rows
  eval_rows="$(ensure_confiqa_eval_rows "$short")"
  task_eval_rows="$(confiqa_eval_rows_for_task "$short")"
  local site_task_root="$SITE_RUN_ROOT/$task"
  local selected_csv="$site_task_root/site_specs/$family/selected_components.csv"
  test -f "$selected_csv"
  test -f "$site_task_root/train/$family/selected/fixed_actuator.pt"
  local actuators
  actuators="$(component_actuators_for_family "$family" "$site_task_root")"
  local controls
  controls="$(control_parts_for_family "$family" "$site_task_root" "$apply_mode")"
  local out_dir="$RUN_ROOT/$task/$family"
  if [[ -f "$out_dir/generation_summary.csv" ]]; then
    stage "${task}_${family}_open" reused
    append_confiqa_summary "$task" "$family" "$out_dir"
    return
  fi
  stage "${task}_${family}_open" running
  mkdir -p "$out_dir"
  "$PY" -m screscomp.cli.cecm_run_joint_actuator_generation \
    --model "$CONFIQA_MODEL" \
    --eval-open-rows "$eval_rows" \
    --component-actuators "$actuators" \
    --controls "$controls" \
    --generation-prompt-key base_rag \
    --prior-source dataset_orig \
    --split "$CONFIQA_EVAL_SPLIT" \
    --start 0 \
    --max-rows "$task_eval_rows" \
    --generation-apply-mode prefill \
    --max-new-tokens "$CONFIQA_MAX_NEW_TOKENS" \
    --stop-strings "Q:" \
    --empty-cache-every 25 \
    --summary-every 50 \
    --torch-dtype bfloat16 \
    --device cuda \
    --out-dir "$out_dir" \
    >"$out_dir/generation.log" 2>&1
  append_confiqa_summary "$task" "$family" "$out_dir"
  stage "${task}_${family}_open" done
}

stage preflight running
"$PY" -m py_compile \
  src/screscomp/cli/run_imdb_sentiment_actuator_generation.py \
  src/screscomp/cli/score_imdb_sentiment_generations.py \
  src/screscomp/cli/compute_imdb_generation_kl.py \
  src/screscomp/cli/cecm_run_joint_actuator_generation.py \
  src/screscomp/cli/prepare_ckplug_open.py
stage preflight done

for task in $TASKS; do
  case "$task" in
    imdb)
      for family in $FAMILIES; do
        run_imdb_family "$family"
      done
      ;;
    qa|mr|mc)
      for family in $FAMILIES; do
        run_confiqa_family "$task" "$family"
      done
      ;;
    *)
      echo "Unknown task: $task" >&2
      exit 2
      ;;
  esac
done

stage complete done
log "done run_root=$RUN_ROOT"
cat "$SUMMARY"
