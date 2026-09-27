#!/usr/bin/env bash
set -euo pipefail

# Formal ConFiQA post-O attention-component rerun.
# The former head-space actuator is replaced by fixed residual-dimension
# actuators trained on Lx.attn module outputs after the attention O projection.

export LC_ALL=C.UTF-8
cd LOCAL_HOME/RPEC/projects/screscomp
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

PY="${PY:-LOCAL_HOME/anaconda3/envs/screscomp/bin/python}"
MODEL="${MODEL:-LOCAL_HOME/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct}"
RUN_ROOT="${RUN_ROOT:-/dev/shm/screscomp_runs/confiqa_post_o_attention_all3_$(date -u +%Y%m%d_%H%M%S)}"
TASKS="${TASKS:-qa mr mc}"
GPU="${GPU:-3}"

DISCOVERY_BASE="${DISCOVERY_BASE:-data_ckplug/cast_confiqa_all3_smalltrain_heldout5000_v0}"
DISCOVERY_QA="${DISCOVERY_QA:-$DISCOVERY_BASE/qa}"
DISCOVERY_MR="${DISCOVERY_MR:-data_ckplug/cast_confiqa_mr_smalltrain_disc120_eval500_v0/mr}"
DISCOVERY_MC="${DISCOVERY_MC:-data_ckplug/cast_confiqa_mc_smalltrain_disc120_eval500_v0/mc}"
FETCH_DIR="${FETCH_DIR:-data_ckplug/context_dpo_confiqa_official}"
SPECS="${SPECS:-mlp4_attn4:4:4}"

TRAIN_SOURCE_ROWS="${TRAIN_SOURCE_ROWS:-300}"
EVAL_SOURCE_START="${EVAL_SOURCE_START:-0}"
EVAL_SOURCE_ROWS="${EVAL_SOURCE_ROWS:-}"
VAL_MOD="${VAL_MOD:-5}"
TRAIN_ROWS="${TRAIN_ROWS:-240}"
VAL_ROWS="${VAL_ROWS:-60}"
EVAL_ROWS="${EVAL_ROWS:-}"
EVAL_SPLIT="${EVAL_SPLIT:-all}"

EPOCHS="${EPOCHS:-2}"
MLP_EPOCHS="${MLP_EPOCHS:-$EPOCHS}"
ATTN_EPOCHS="${ATTN_EPOCHS:-$EPOCHS}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
LR="${LR:-0.05}"
LAMBDA_NORM="${LAMBDA_NORM:-1e-4}"
STATE_MARGIN_WEIGHT="${STATE_MARGIN_WEIGHT:-0}"
GAIN_WEIGHT="${GAIN_WEIGHT:-1}"
TARGET_MARGIN="${TARGET_MARGIN:-0.0}"
TARGET_GAIN="${TARGET_GAIN:-0.0}"
SCORE_MODE="${SCORE_MODE:-answer_rest_margin}"
EMPTY_CACHE_EVERY="${EMPTY_CACHE_EVERY:-25}"
MLP_MAX_ALIASES_PER_SIDE="${MLP_MAX_ALIASES_PER_SIDE:-1}"
ATTN_MAX_ALIASES_PER_SIDE="${ATTN_MAX_ALIASES_PER_SIDE:-1}"
ACTUATOR_ALPHA_SWEEP="${ACTUATOR_ALPHA_SWEEP:-0,1}"

MLP_ALPHA="${MLP_ALPHA:-0.05}"
ATTN_ALPHA="${ATTN_ALPHA:-0.5}"
ATTN_TRAIN_APPLY_MODE="${ATTN_TRAIN_APPLY_MODE:-decision_tokens}"
ATTN_EVAL_APPLY_MODE="${ATTN_EVAL_APPLY_MODE:-all}"
MLP_TRAIN_APPLY_MODE="${MLP_TRAIN_APPLY_MODE:-decision_tokens}"
MLP_EVAL_APPLY_MODE="${MLP_EVAL_APPLY_MODE:-prefill}"
CAST_CONTROLS="${CAST_CONTROLS:-base=;cast_attn=comp:attn_suppress:${ATTN_ALPHA}:${ATTN_EVAL_APPLY_MODE}+comp:attn_boost:${ATTN_ALPHA}:${ATTN_EVAL_APPLY_MODE};cast_full=comp:attn_suppress:${ATTN_ALPHA}:${ATTN_EVAL_APPLY_MODE}+comp:attn_boost:${ATTN_ALPHA}:${ATTN_EVAL_APPLY_MODE}+comp:prior_mlp:${MLP_ALPHA}:${MLP_EVAL_APPLY_MODE}}"

mkdir -p "$RUN_ROOT"
SUMMARY="$RUN_ROOT/post_o_attention_summary.tsv"
MASTER_LOG="$RUN_ROOT/post_o_attention_all3.log"
echo -e "task\tconfig\tmlp_topk\tattn_topk\tstate_margin_weight\tgain_weight\tcontrol_name\tn\tpc\tpo\tmr\tem\tcontext_only\tprior_only\tboth\tneither\tmean_chars\tout_dir" > "$SUMMARY"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$MASTER_LOG"
}

data_json_for_task() {
  case "$1" in
    qa) echo "$FETCH_DIR/ConFiQA-QA.json" ;;
    mr) echo "$FETCH_DIR/ConFiQA-MR.json" ;;
    mc) echo "$FETCH_DIR/ConFiQA-MC.json" ;;
    *) echo "TASK must be qa, mr, or mc; got $1" >&2; exit 2 ;;
  esac
}

component_screen_for_task() {
  local task="$1"
  local upper override_var override
  upper="$(printf '%s' "$task" | tr '[:lower:]' '[:upper:]')"
  override_var="COMPONENT_SCREEN_${upper}"
  override="${!override_var:-}"
  if [[ -n "$override" ]]; then
    echo "$override"
    return
  fi
  case "$task" in
    qa) echo "$DISCOVERY_QA/discovery/component_scan/component_screen.csv" ;;
    mr) echo "$DISCOVERY_MR/discovery/component_scan/component_screen.csv" ;;
    mc) echo "$DISCOVERY_MC/discovery/component_scan/component_screen.csv" ;;
    *) echo "TASK must be qa, mr, or mc; got $task" >&2; exit 2 ;;
  esac
}

select_window() {
  local component_screen="$1"
  local out_dir="$2"
  local mlp_topk="$3"
  local attn_topk="$4"
  mkdir -p "$out_dir"
  COMPONENT_SCREEN="$component_screen" SELECT_OUT="$out_dir" MLP_TOPK="$mlp_topk" ATTN_TOPK="$attn_topk" "$PY" - <<'PY'
import csv, json, os
from pathlib import Path

component_screen = Path(os.environ["COMPONENT_SCREEN"])
out = Path(os.environ["SELECT_OUT"])
mlp_topk = int(os.environ["MLP_TOPK"])
attn_topk = int(os.environ["ATTN_TOPK"])

def f(row, key, default=0.0):
    try:
        value = str(row.get(key, "")).strip()
        return float(value) if value else default
    except Exception:
        return default

def write_components(path, rows):
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["component_id", "layer_idx", "component_type"])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "component_id": row["component_id"],
                "layer_idx": row["layer_idx"],
                "component_type": row["component_type"],
            })

rows = list(csv.DictReader(component_screen.open("r", encoding="utf-8", newline="")))
mlp_rows = [r for r in rows if r.get("component_type") == "mlp"]
attn_rows = [r for r in rows if r.get("component_type") == "attn"]
if not mlp_rows:
    raise SystemExit(f"No MLP component rows in {component_screen}")
if not attn_rows:
    raise SystemExit(f"No attention component rows in {component_screen}")

mlp_neg = [r for r in mlp_rows if f(r, "mean_delta") < 0]
selected_mlp = sorted(mlp_neg or mlp_rows, key=lambda r: (f(r, "mean_delta"), -f(r, "sign_consistency")))[:mlp_topk]
if len(selected_mlp) < mlp_topk:
    raise SystemExit(f"Only selected {len(selected_mlp)} MLP rows for topk={mlp_topk}")

def select_attn(role, used=()):
    used_ids = set(used)
    if role == "suppress":
        signed = [r for r in attn_rows if f(r, "mean_delta") < 0 and r.get("component_id") not in used_ids]
        primary = sorted(signed, key=lambda r: (f(r, "mean_delta"), -f(r, "sign_consistency")))
    else:
        signed = [r for r in attn_rows if f(r, "mean_delta") > 0 and r.get("component_id") not in used_ids]
        primary = sorted(signed, key=lambda r: (f(r, "mean_delta"), f(r, "sign_consistency")), reverse=True)
    fallback = sorted(
        [r for r in attn_rows if r.get("component_id") not in used_ids and r not in primary],
        key=lambda r: (f(r, "abs_mean_delta"), f(r, "sign_consistency")),
        reverse=True,
    )
    return (primary + fallback)[:attn_topk]

selected_suppress = select_attn("suppress")
selected_boost = select_attn("boost", used=[r["component_id"] for r in selected_suppress])
if len(selected_suppress) < attn_topk or len(selected_boost) < attn_topk:
    raise SystemExit(
        f"Insufficient attention rows: suppress={len(selected_suppress)} boost={len(selected_boost)} topk={attn_topk}"
    )

prior_csv = out / "prior_mlp_components.csv"
suppress_csv = out / "attn_suppress_components.csv"
boost_csv = out / "attn_boost_components.csv"
write_components(prior_csv, selected_mlp)
write_components(suppress_csv, selected_suppress)
write_components(boost_csv, selected_boost)
(out / "selected.env").write_text(
    f"PRIOR_MLP_COMPONENTS={prior_csv}\n"
    f"SUPPRESS_ATTN_COMPONENTS={suppress_csv}\n"
    f"BOOST_ATTN_COMPONENTS={boost_csv}\n",
    encoding="utf-8",
)
(out / "selected_window.json").write_text(json.dumps({
    "component_screen": str(component_screen),
    "mlp_topk": mlp_topk,
    "attn_topk": attn_topk,
    "mlp_selection": "top negative MLP by mean_delta, fallback same ordering over all MLP",
    "attention_selection": "post-O attn components split by signed mean_delta; suppress<0, boost>0, non-overlap enforced",
    "mlp_components": selected_mlp,
    "attn_suppress_components": selected_suppress,
    "attn_boost_components": selected_boost,
}, ensure_ascii=False, indent=2), encoding="utf-8")
PY
}

append_summary() {
  local task="$1"
  local config="$2"
  local mlp_topk="$3"
  local attn_topk="$4"
  local gen_summary="$5"
  local out_dir="$6"
  TASK_VALUE="$task" CONFIG="$config" MLP_TOPK="$mlp_topk" ATTN_TOPK="$attn_topk" STATE_MARGIN_WEIGHT_VALUE="$STATE_MARGIN_WEIGHT" GAIN_WEIGHT_VALUE="$GAIN_WEIGHT" GEN_SUMMARY="$gen_summary" OUT_DIR_VALUE="$out_dir" SUMMARY="$SUMMARY" "$PY" - <<'PY'
import csv, os
from pathlib import Path

def pick(row, *names):
    for name in names:
        value = row.get(name, "")
        if value != "":
            return value
    return ""

rows = list(csv.DictReader(Path(os.environ["GEN_SUMMARY"]).open("r", encoding="utf-8", newline="")))
with Path(os.environ["SUMMARY"]).open("a", encoding="utf-8") as fp:
    for row in rows:
        fp.write("\t".join(str(v) for v in [
            os.environ["TASK_VALUE"],
            os.environ["CONFIG"],
            os.environ["MLP_TOPK"],
            os.environ["ATTN_TOPK"],
            os.environ["STATE_MARGIN_WEIGHT_VALUE"],
            os.environ["GAIN_WEIGHT_VALUE"],
            row.get("control_name", ""),
            row.get("n", ""),
            row.get("pc", ""),
            row.get("po", ""),
            row.get("mr", ""),
            row.get("em", ""),
            pick(row, "context_only_rate", "context_only"),
            pick(row, "prior_only_rate", "prior_only"),
            pick(row, "both_rate", "both"),
            pick(row, "neither_rate", "neither"),
            row.get("mean_chars", ""),
            os.environ["OUT_DIR_VALUE"],
        ]) + "\n")
PY
}

run_task() {
  local task="$1"
  local gpu="$GPU"
  local base_out="$RUN_ROOT/$task"
  local common="$base_out/common"
  local pairs_dir="$common/pairs"
  local train_open_rows="$common/train_open_rows.jsonl"
  local eval_open_rows="$common/eval_open_rows.jsonl"
  local data_json
  data_json="$(data_json_for_task "$task")"
  local component_screen
  component_screen="$(component_screen_for_task "$task")"
  local task_log="$base_out/post_o_attention.log"

  mkdir -p "$common"
  test -f "$component_screen"

  log "task=$task gpu=$gpu base_out=$base_out component_screen=$component_screen"
  if [[ ! -f "$pairs_dir/pairs.csv" || ! -f "$eval_open_rows" ]]; then
    echo "[$(date -Is)] prepare task=$task rows/pairs" | tee -a "$task_log"
    "$PY" -m screscomp.cli.cecm_fetch_confiqa --out-dir "$FETCH_DIR" --tasks "$task" | tee -a "$task_log"
    prepare_train_args=("$PY" -m screscomp.cli.prepare_ckplug_open --dataset confiqa --data_json "$data_json" --out_jsonl "$train_open_rows" --schema base --alias_policy raw)
    if [[ -n "$TRAIN_SOURCE_ROWS" ]]; then prepare_train_args+=(--max_rows "$TRAIN_SOURCE_ROWS"); fi
    "${prepare_train_args[@]}" | tee -a "$task_log"
    prepare_eval_args=("$PY" -m screscomp.cli.prepare_ckplug_open --dataset confiqa --data_json "$data_json" --out_jsonl "$eval_open_rows" --schema base --alias_policy raw --start "$EVAL_SOURCE_START")
    if [[ -n "$EVAL_SOURCE_ROWS" ]]; then prepare_eval_args+=(--max_rows "$EVAL_SOURCE_ROWS"); fi
    "${prepare_eval_args[@]}" | tee -a "$task_log"
    "$PY" -m screscomp.cli.cecm_build_preference_pairs \
      --input-jsonl "$train_open_rows" \
      --out-dir "$pairs_dir" \
      --event source_context_over_prior \
      --prompt-key base_rag \
      --prior-source dataset_orig \
      --val-mod "$VAL_MOD" \
      | tee -a "$task_log"
  fi

  local unique_mlp unique_attn
  unique_mlp="$(for spec in $SPECS; do IFS=: read -r _name m _a <<< "$spec"; echo "$m"; done | sort -n | uniq | tr '\n' ' ')"
  unique_attn="$(for spec in $SPECS; do IFS=: read -r _name _m a <<< "$spec"; echo "$a"; done | sort -n | uniq | tr '\n' ' ')"

  for k in $unique_mlp; do
    local select_dir="$base_out/select_mlp${k}_attn1"
    select_window "$component_screen" "$select_dir" "$k" 1
    # shellcheck disable=SC1090
    source "$select_dir/selected.env"
    local out="$base_out/actuators/mlp${k}"
    if [[ -f "$out/fixed_actuator.pt" ]]; then
      echo "[$(date -Is)] reuse task=$task mlp top$k $out" | tee -a "$task_log"
      continue
    fi
    mkdir -p "$out"
    echo "[$(date -Is)] train task=$task mlp top$k" | tee -a "$task_log"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m screscomp.cli.cecm_train_fixed_actuator \
      --model "$MODEL" \
      --pairs-csv "$pairs_dir/pairs.csv" \
      --components-csv "$PRIOR_MLP_COMPONENTS" \
      --event source_context_over_prior \
      --train-split train \
      --val-split val \
      --max-train-rows "$TRAIN_ROWS" \
      --max-val-rows "$VAL_ROWS" \
      --epochs "$MLP_EPOCHS" \
      --train-batch-size "$TRAIN_BATCH_SIZE" \
      --lr "$LR" \
      --lambda-norm "$LAMBDA_NORM" \
      --alpha-train 1.0 \
      --alpha-sweep "$ACTUATOR_ALPHA_SWEEP" \
      --state-margin-weight "$STATE_MARGIN_WEIGHT" \
      --gain-weight "$GAIN_WEIGHT" \
      --target-margin "$TARGET_MARGIN" \
      --target-gain "$TARGET_GAIN" \
      --apply-mode "$MLP_TRAIN_APPLY_MODE" \
      --score-mode "$SCORE_MODE" \
      --max-aliases-per-side "$MLP_MAX_ALIASES_PER_SIDE" \
      --empty-cache-every "$EMPTY_CACHE_EVERY" \
      --torch-dtype bfloat16 \
      --device cuda \
      --out-dir "$out" \
      > "$out/train.log" 2>&1
  done

  for k in $unique_attn; do
    local select_dir="$base_out/select_mlp1_attn${k}"
    select_window "$component_screen" "$select_dir" 1 "$k"
    # shellcheck disable=SC1090
    source "$select_dir/selected.env"
    for role in suppress boost; do
      local components_var="SUPPRESS_ATTN_COMPONENTS"
      if [[ "$role" == "boost" ]]; then components_var="BOOST_ATTN_COMPONENTS"; fi
      local components_csv="${!components_var}"
      local out="$base_out/actuators/attn${k}_${role}"
      if [[ -f "$out/fixed_actuator.pt" ]]; then
        echo "[$(date -Is)] reuse task=$task attn $role top$k $out" | tee -a "$task_log"
        continue
      fi
      mkdir -p "$out"
      echo "[$(date -Is)] train task=$task post-O attn $role top$k" | tee -a "$task_log"
      CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m screscomp.cli.cecm_train_fixed_actuator \
        --model "$MODEL" \
        --pairs-csv "$pairs_dir/pairs.csv" \
        --components-csv "$components_csv" \
        --event source_context_over_prior \
        --train-split train \
        --val-split val \
        --max-train-rows "$TRAIN_ROWS" \
        --max-val-rows "$VAL_ROWS" \
        --epochs "$ATTN_EPOCHS" \
        --train-batch-size "$TRAIN_BATCH_SIZE" \
        --lr "$LR" \
        --lambda-norm "$LAMBDA_NORM" \
        --alpha-train 1.0 \
        --alpha-sweep "$ACTUATOR_ALPHA_SWEEP" \
        --state-margin-weight "$STATE_MARGIN_WEIGHT" \
        --gain-weight "$GAIN_WEIGHT" \
        --target-margin "$TARGET_MARGIN" \
        --target-gain "$TARGET_GAIN" \
        --apply-mode "$ATTN_TRAIN_APPLY_MODE" \
        --score-mode "$SCORE_MODE" \
        --max-aliases-per-side "$ATTN_MAX_ALIASES_PER_SIDE" \
        --empty-cache-every "$EMPTY_CACHE_EVERY" \
        --torch-dtype bfloat16 \
        --device cuda \
        --out-dir "$out" \
        > "$out/train.log" 2>&1
    done
  done

  for spec in $SPECS; do
    IFS=: read -r name mlp_topk attn_topk <<< "$spec"
    local out="$base_out/$name/eval_cast_base_rag"
    mkdir -p "$out"
    echo "[$(date -Is)] eval task=$task $name mlp=$mlp_topk attn=$attn_topk controls=$CAST_CONTROLS" | tee -a "$task_log"
    gen_args=(
      "$PY" -m screscomp.cli.cecm_run_joint_actuator_generation
      --model "$MODEL"
      --eval-open-rows "$eval_open_rows"
      --component-actuators "prior_mlp=$base_out/actuators/mlp${mlp_topk}/fixed_actuator.pt;attn_suppress=$base_out/actuators/attn${attn_topk}_suppress/fixed_actuator.pt;attn_boost=$base_out/actuators/attn${attn_topk}_boost/fixed_actuator.pt"
      --controls "$CAST_CONTROLS"
      --generation-prompt-key base_rag
      --prior-source dataset_orig
      --split "$EVAL_SPLIT"
      --start 0
      --generation-apply-mode prefill
      --max-new-tokens 64
      --stop-strings "Q:"
      --empty-cache-every "$EMPTY_CACHE_EVERY"
      --torch-dtype bfloat16
      --device cuda
      --out-dir "$out"
    )
    if [[ -n "$EVAL_ROWS" ]]; then
      gen_args+=(--max-rows "$EVAL_ROWS")
    else
      gen_args+=(--max-rows 999999999)
    fi
    CUDA_VISIBLE_DEVICES="$gpu" "${gen_args[@]}" > "$out/generation.log" 2>&1
    append_summary "$task" "$name" "$mlp_topk" "$attn_topk" "$out/generation_summary.csv" "$base_out/$name"
  done
}

log "preflight"
"$PY" -m py_compile \
  src/screscomp/cli/cecm_train_fixed_actuator.py \
  src/screscomp/cli/cecm_run_joint_actuator_generation.py \
  src/screscomp/cli/prepare_ckplug_open.py \
  src/screscomp/cli/cecm_build_preference_pairs.py

idx=0
for task in $TASKS; do
  run_task "$task"
  idx=$((idx + 1))
done

log "done run_root=$RUN_ROOT"
cat "$SUMMARY"
