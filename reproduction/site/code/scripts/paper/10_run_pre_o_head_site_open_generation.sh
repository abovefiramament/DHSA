#!/usr/bin/env bash
set -euo pipefail

# Open-generation evaluation for 09_run_pre_o_head_site_shift.sh.
#
# It evaluates head-only role combinations:
#   IMDb: positive head actuator + negative head actuator.
#   ConFiQA: suppress head actuator + boost head actuator.
#
# No same-vector controls and no MLP/full controls are evaluated here.

export LC_ALL=C.UTF-8
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

REPO_ROOT="${REPO_ROOT:-LOCAL_HOME/RPEC/projects/screscomp}"
cd "$REPO_ROOT"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

PY="${PY:-LOCAL_HOME/anaconda3/envs/screscomp/bin/python}"
SITE_RUN_ROOT="${SITE_RUN_ROOT:?Set SITE_RUN_ROOT to the 09_run_pre_o_head_site_shift.sh output root}"
RUN_ROOT="${RUN_ROOT:-/dev/shm/screscomp_runs/pre_o_head_site_open_generation_$(date -u +%Y%m%d_%H%M%S)}"
GPU="${GPU:-0}"
TASKS="${TASKS:-imdb qa mr mc}"
OPEN_SHIFTS="${OPEN_SHIFTS:-1 2 3}"
INCLUDE_SHIFT_CONTROLS="${INCLUDE_SHIFT_CONTROLS:-1}"
EXTRA_OPEN_CONTROLS="${EXTRA_OPEN_CONTROLS:-}"
ALPHA_SWEEP="${ALPHA_SWEEP:-0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}"
ALPHA_PLAN_TSV="${ALPHA_PLAN_TSV:-}"
INCLUDE_BASE="${INCLUDE_BASE:-1}"
INCLUDE_SELECTED="${INCLUDE_SELECTED:-1}"
USE_RETRAINED_SELECTED="${USE_RETRAINED_SELECTED:-1}"

IMDB_ROOT="${IMDB_ROOT:-/dev/shm/screscomp_runs/imdb_ma921_expanded1024_sampled_paperkl_20260626_055222}"
IMDB_MODEL="${IMDB_MODEL:-LOCAL_HOME/.cache/huggingface/hub/models--ma921--gpt2-large-sft-imdb/snapshots/f480190690d5abfc0e003ccb4f7e650626019bb9}"
IMDB_SELECTED_POS_OUT="${IMDB_SELECTED_POS_OUT:-$IMDB_ROOT/train/train_head_positive}"
IMDB_SELECTED_NEG_OUT="${IMDB_SELECTED_NEG_OUT:-$IMDB_ROOT/train/train_head_negative}"
IMDB_PROMPTS_SPLIT="${IMDB_PROMPTS_SPLIT:-test}"
IMDB_PROMPTS_JSONL="${IMDB_PROMPTS_JSONL:-$IMDB_ROOT/eval_env/$IMDB_PROMPTS_SPLIT/prompts.jsonl}"
IMDB_EVAL_SPLIT="${IMDB_EVAL_SPLIT:-eval}"
IMDB_EVAL_ROWS="${IMDB_EVAL_ROWS:-2048}"
IMDB_GEN_BATCH="${IMDB_GEN_BATCH:-32}"
IMDB_SCORE_BATCH="${IMDB_SCORE_BATCH:-32}"
IMDB_KL_BATCH="${IMDB_KL_BATCH:-32}"
IMDB_SCORER_MODEL="${IMDB_SCORER_MODEL:-LOCAL_HOME/.cache/huggingface/hub/models--siebert--sentiment-roberta-large-english/snapshots/74cea614e245b0832c770ec9aa51bd58df965b9c}"
IMDB_HEAD_APPLY_MODE="${IMDB_HEAD_APPLY_MODE:-all}"

CONFIQA_MODEL="${CONFIQA_MODEL:-LOCAL_HOME/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct}"
FETCH_DIR="${FETCH_DIR:-data_ckplug/context_dpo_confiqa_official}"
DISCOVERY_BASE="${DISCOVERY_BASE:-data_ckplug/cast_confiqa_all3_disc120_heldout500_v0}"
DISCOVERY_QA="${DISCOVERY_QA:-$DISCOVERY_BASE/qa}"
DISCOVERY_MR="${DISCOVERY_MR:-$DISCOVERY_BASE/mr}"
DISCOVERY_MC="${DISCOVERY_MC:-$DISCOVERY_BASE/mc}"
CONFIQA_EVAL_SOURCE_START="${CONFIQA_EVAL_SOURCE_START:-301}"
CONFIQA_EVAL_SOURCE_ROWS="${CONFIQA_EVAL_SOURCE_ROWS:-120}"
CONFIQA_EVAL_ROWS="${CONFIQA_EVAL_ROWS:-120}"
CONFIQA_EVAL_SPLIT="${CONFIQA_EVAL_SPLIT:-all}"
CONFIQA_MAX_NEW_TOKENS="${CONFIQA_MAX_NEW_TOKENS:-64}"
CONFIQA_HEAD_APPLY_MODE="${CONFIQA_HEAD_APPLY_MODE:-decision_tokens}"

export CUDA_VISIBLE_DEVICES="$GPU"

mkdir -p "$RUN_ROOT/logs"
STATUS="$RUN_ROOT/status.tsv"
SUMMARY="$RUN_ROOT/open_generation_summary.tsv"
MASTER_LOG="$RUN_ROOT/pre_o_head_site_open_generation.log"
printf 'time\tstage\tstatus\n' > "$STATUS"
printf 'task\tcontrol_name\tn\tscore\tkl\tpositive_rate\tpc\tem\tshort\tpo\tmean_chars\tlogic_score\tcontext_only\tprior_only\tboth\tneither\tmr\tout_dir\n' > "$SUMMARY"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$MASTER_LOG"
}

stage() {
  printf '%s\t%s\t%s\n' "$(date -Is)" "$1" "$2" >> "$STATUS"
  log "stage=$1 $2"
}

open_control_names() {
  local controls=()
  if [[ "$INCLUDE_SHIFT_CONTROLS" == "1" ]]; then
    for shift in $OPEN_SHIFTS; do
      controls+=("shift${shift}_retrained")
    done
  fi
  for control in $EXTRA_OPEN_CONTROLS; do
    controls+=("$control")
  done
  printf '%s\n' "${controls[@]}"
}

data_json_for_task() {
  case "$1" in
    qa) echo "$FETCH_DIR/ConFiQA-QA.json" ;;
    mr) echo "$FETCH_DIR/ConFiQA-MR.json" ;;
    mc) echo "$FETCH_DIR/ConFiQA-MC.json" ;;
    *) echo "unsupported ConFiQA task: $1" >&2; exit 2 ;;
  esac
}

discovery_root_for_task() {
  case "$1" in
    qa) echo "$DISCOVERY_QA" ;;
    mr) echo "$DISCOVERY_MR" ;;
    mc) echo "$DISCOVERY_MC" ;;
    *) echo "unsupported ConFiQA task: $1" >&2; exit 2 ;;
  esac
}

alpha_sweep_for_control() {
  local task="$1"
  local control="$2"
  if [[ -n "$ALPHA_PLAN_TSV" ]]; then
    local planned
    planned="$("$PY" - "$ALPHA_PLAN_TSV" "$task" "$control" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
task = sys.argv[2]
control = sys.argv[3]
if not path.exists():
    raise SystemExit(f"missing alpha plan: {path}")
values = []
with path.open(encoding="utf-8", newline="") as fp:
    for row in csv.DictReader(fp, delimiter="\t"):
        if row.get("task") == task and row.get("control") == control:
            alpha = str(row.get("alpha", "")).strip()
            if alpha:
                values.append(alpha)
seen = []
for value in values:
    if value not in seen:
        seen.append(value)
print(",".join(seen))
PY
)"
    if [[ -n "$planned" ]]; then
      echo "$planned"
      return
    fi
  fi
  echo "$ALPHA_SWEEP"
}

alpha_controls_for_heads() {
  local prefix="$1"
  local role_a="$2"
  local role_b="$3"
  local alpha_sweep="${4:-$ALPHA_SWEEP}"
  local apply_mode="${5:-all}"
  local controls=()
  for alpha in ${alpha_sweep//,/ }; do
    controls+=("${prefix}_a${alpha//./p}=head_act:${role_a}:${alpha}:${apply_mode}+head_act:${role_b}:${alpha}:${apply_mode}")
  done
  local joined
  joined="$(IFS=';'; echo "${controls[*]}")"
  echo "$joined"
}

combine_head_payloads() {
  local payload_a="$1"
  local payload_b="$2"
  local dst_payload="$3"
  local label="$4"
  mkdir -p "$(dirname "$dst_payload")"
  PAYLOAD_A="$payload_a" PAYLOAD_B="$payload_b" DST_PAYLOAD="$dst_payload" LABEL="$label" "$PY" - <<'PY'
import os
from pathlib import Path

import torch

paths = [Path(os.environ["PAYLOAD_A"]), Path(os.environ["PAYLOAD_B"])]
payloads = [torch.load(path, map_location="cpu") for path in paths]
combined = dict(payloads[0])
heads = []
vectors = {}
seen = set()
for source_path, payload in zip(paths, payloads, strict=False):
    for row in payload.get("heads", []):
        row = dict(row)
        head_id = str(row["head_id"])
        if head_id in seen:
            raise SystemExit(f"duplicate head in combined payload: {head_id}")
        seen.add(head_id)
        row["source_payload"] = str(source_path)
        heads.append(row)
        vectors[head_id] = payload["vectors"][head_id].detach().clone()
combined["heads"] = heads
combined["vectors"] = vectors
combined["site_control"] = {
    "mode": "role_combination",
    "label": os.environ["LABEL"],
    "source_payloads": [str(path) for path in paths],
}
dst = Path(os.environ["DST_PAYLOAD"])
dst.parent.mkdir(parents=True, exist_ok=True)
torch.save(combined, dst)
print(dst, flush=True)
PY
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
  local out_dir="$1"
  TASK_VALUE="imdb" OUT_DIR_VALUE="$out_dir" SUMMARY="$SUMMARY" "$PY" - <<'PY'
import csv
import os
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
            control,
            row.get("n", ""),
            row.get("mean_positive_sentiment_score", ""),
            row.get("paper_mean_sampled_sequence_kl", row.get("mean_sequence_kl", "")),
            row.get("positive_rate_ge_0p5", ""),
            "",
            "",
            "",
            "",
            row.get("mean_completion_chars", ""),
            "",
            "",
            "",
            "",
            "",
            "",
            os.environ["OUT_DIR_VALUE"],
        ]) + "\n")
PY
}

run_imdb_control() {
  local name="$1"
  local payload="$2"
  local alpha_sweep="$3"
  local out_dir="$RUN_ROOT/imdb/$name"
  if [[ -f "$out_dir/score_summary.csv" ]]; then
    stage "imdb_${name}" reused
    append_imdb_summary "$out_dir"
    return
  fi
  stage "imdb_${name}" running
  mkdir -p "$out_dir"
  "$PY" -m screscomp.cli.run_imdb_sentiment_actuator_generation \
    --model "$IMDB_MODEL" \
    --prompts-jsonl "$IMDB_PROMPTS_JSONL" \
    --out-jsonl "$out_dir/generations.jsonl" \
    --head-actuator "$payload" \
    --control-name "$name" \
    --alpha-sweep "$alpha_sweep" \
    --generation-apply-mode "$IMDB_HEAD_APPLY_MODE" \
    --head-apply-mode "$IMDB_HEAD_APPLY_MODE" \
    --split "$IMDB_EVAL_SPLIT" --max-rows "$IMDB_EVAL_ROWS" \
    --samples-per-prompt 1 \
    --generation-batch-size "$IMDB_GEN_BATCH" --max-new-tokens 256 \
    --temperature 1.0 --top-p 1.0 --top-k 50 --seed 42 --same-seed-across-alpha \
    --device cuda --torch-dtype bfloat16 \
    >"$out_dir/generate.log" 2>&1
  score_imdb_dir "$out_dir"
  append_imdb_summary "$out_dir"
  stage "imdb_${name}" done
}

run_imdb() {
  local task_root="$SITE_RUN_ROOT/imdb/train/head_pre_o"
  local selected_pos_out="$IMDB_SELECTED_POS_OUT"
  local selected_neg_out="$IMDB_SELECTED_NEG_OUT"
  if [[ "$USE_RETRAINED_SELECTED" == "1" && -f "$task_root/selected/positive/head_actuator.pt" && -f "$task_root/selected/negative/head_actuator.pt" ]]; then
    selected_pos_out="$task_root/selected/positive"
    selected_neg_out="$task_root/selected/negative"
  fi
  test -f "$selected_pos_out/head_actuator.pt"
  test -f "$selected_neg_out/head_actuator.pt"
  local payload_root="$RUN_ROOT/imdb/combined_payloads"
  local selected_payload="$payload_root/selected/head_actuator.pt"
  combine_head_payloads \
    "$selected_pos_out/head_actuator.pt" \
    "$selected_neg_out/head_actuator.pt" \
    "$selected_payload" selected

  if [[ "$INCLUDE_BASE" == "1" ]]; then
    run_imdb_control base "$selected_payload" "0"
  fi
  if [[ "$INCLUDE_SELECTED" == "1" ]]; then
    run_imdb_control selected "$selected_payload" "$(alpha_sweep_for_control imdb selected)"
  fi
  for control in $(open_control_names); do
    local dir="$task_root/$control"
    local payload="$payload_root/$control/head_actuator.pt"
    test -f "$dir/positive/head_actuator.pt"
    test -f "$dir/negative/head_actuator.pt"
    combine_head_payloads "$dir/positive/head_actuator.pt" "$dir/negative/head_actuator.pt" "$payload" "$control"
    run_imdb_control "$control" "$payload" "$(alpha_sweep_for_control imdb "$control")"
  done
}

ensure_confiqa_eval_rows() {
  local short="$1"
  local task_out="$RUN_ROOT/confiqa_$short/common"
  local out_jsonl="$task_out/eval_open_rows.jsonl"
  local data_json
  data_json="$(data_json_for_task "$short")"
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
    --max_rows "$CONFIQA_EVAL_SOURCE_ROWS" \
    >"$task_out/prepare_eval_open.log" 2>&1
  stage "prepare_open_${short}" done >&2
  echo "$out_jsonl"
}

append_confiqa_summary() {
  local task="$1"
  local out_dir="$2"
  TASK_VALUE="$task" OUT_DIR_VALUE="$out_dir" SUMMARY="$SUMMARY" "$PY" - <<'PY'
import csv
import os
from pathlib import Path

summary = Path(os.environ["SUMMARY"])
path = Path(os.environ["OUT_DIR_VALUE"]) / "generation_summary.csv"
if not path.exists():
    raise SystemExit(f"missing {path}")
with path.open(encoding="utf-8", newline="") as fp, summary.open("a", encoding="utf-8") as out:
    for row in csv.DictReader(fp):
        def f(key: str) -> float:
            value = row.get(key, "")
            return float(value) if value not in {"", None} else 0.0

        pc = f("pc") if row.get("pc", "") != "" else f("context_hit_rate")
        em = f("em") if row.get("em", "") != "" else f("context_em_rate")
        short = f("short_exact_context_rate")
        po = f("po") if row.get("po", "") != "" else f("prior_hit_rate")
        mean_chars = f("mean_chars")
        context_only = f("context_only_rate")
        prior_only = f("prior_only_rate")
        both = f("both_rate")
        neither = f("neither_rate")
        mr = f("mr")
        # ConFiQA alpha selection uses the same open-generation arbitration axis
        # as the CK factor scorer: source identity plus short exact form, with
        # penalties for prior leakage, abstention/neither, and verbosity.
        logic_score = context_only + short - po - neither - mean_chars / 400.0

        out.write("\t".join(str(x) for x in [
            os.environ["TASK_VALUE"],
            row.get("control_name", ""),
            row.get("n", ""),
            logic_score,
            "",
            "",
            pc,
            em,
            short,
            po,
            mean_chars,
            logic_score,
            context_only,
            prior_only,
            both,
            neither,
            mr,
            os.environ["OUT_DIR_VALUE"],
        ]) + "\n")
PY
}

run_confiqa_task() {
  local short="$1"
  local task="confiqa_$short"
  local site_task_root="$SITE_RUN_ROOT/$task/train/head_pre_o"
  local selected_root
  selected_root="$(discovery_root_for_task "$short")"
  local eval_rows
  eval_rows="$(ensure_confiqa_eval_rows "$short")"
  local selected_suppress_payload="$selected_root/train_decision_tokens_head_suppress/head_actuator.pt"
  local selected_boost_payload="$selected_root/train_decision_tokens_head_boost/head_actuator.pt"
  if [[ "$USE_RETRAINED_SELECTED" == "1" && -f "$site_task_root/selected/suppress/head_actuator.pt" && -f "$site_task_root/selected/boost/head_actuator.pt" ]]; then
    selected_suppress_payload="$site_task_root/selected/suppress/head_actuator.pt"
    selected_boost_payload="$site_task_root/selected/boost/head_actuator.pt"
  fi
  test -f "$selected_suppress_payload"
  test -f "$selected_boost_payload"

  local controls=()
  local head_specs=()
  if [[ "$INCLUDE_BASE" == "1" ]]; then
    controls+=("base=")
  fi
  if [[ "$INCLUDE_SELECTED" == "1" ]]; then
    head_specs+=("selected_suppress=$selected_suppress_payload")
    head_specs+=("selected_boost=$selected_boost_payload")
    local selected_alpha_sweep
    selected_alpha_sweep="$(alpha_sweep_for_control "$task" selected)"
    controls+=("$(alpha_controls_for_heads selected selected_suppress selected_boost "$selected_alpha_sweep" "$CONFIQA_HEAD_APPLY_MODE")")
  fi
  for control in $(open_control_names); do
    local dir="$site_task_root/$control"
    test -f "$dir/suppress/head_actuator.pt"
    test -f "$dir/boost/head_actuator.pt"
    head_specs+=("${control}_suppress=$dir/suppress/head_actuator.pt")
    head_specs+=("${control}_boost=$dir/boost/head_actuator.pt")
    local control_alpha_sweep
    control_alpha_sweep="$(alpha_sweep_for_control "$task" "$control")"
    controls+=("$(alpha_controls_for_heads "$control" "${control}_suppress" "${control}_boost" "$control_alpha_sweep" "$CONFIQA_HEAD_APPLY_MODE")")
  done

  local head_actuators
  local control_text
  head_actuators="$(IFS=';'; echo "${head_specs[*]}")"
  control_text="$(IFS=';'; echo "${controls[*]}")"
  local out_dir="$RUN_ROOT/$task/head_pre_o"
  if [[ -f "$out_dir/generation_summary.csv" ]]; then
    stage "${task}_head_pre_o" reused
    append_confiqa_summary "$task" "$out_dir"
    return
  fi
  stage "${task}_head_pre_o" running
  mkdir -p "$out_dir"
  "$PY" -m screscomp.cli.cecm_run_joint_actuator_generation \
    --model "$CONFIQA_MODEL" \
    --eval-open-rows "$eval_rows" \
    --head-actuators "$head_actuators" \
    --controls "$control_text" \
    --generation-prompt-key base_rag \
    --prior-source dataset_orig \
    --split "$CONFIQA_EVAL_SPLIT" \
    --start 0 \
    --max-rows "$CONFIQA_EVAL_ROWS" \
    --generation-apply-mode "$CONFIQA_HEAD_APPLY_MODE" \
    --max-new-tokens "$CONFIQA_MAX_NEW_TOKENS" \
    --stop-strings "Q:" \
    --empty-cache-every 25 \
    --summary-every 50 \
    --torch-dtype bfloat16 \
    --device cuda \
    --out-dir "$out_dir" \
    >"$out_dir/generation.log" 2>&1
  append_confiqa_summary "$task" "$out_dir"
  stage "${task}_head_pre_o" done
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
    imdb) run_imdb ;;
    qa|mr|mc) run_confiqa_task "$task" ;;
    *) echo "Unknown task: $task" >&2; exit 2 ;;
  esac
done

stage complete done
log "done run_root=$RUN_ROOT"
