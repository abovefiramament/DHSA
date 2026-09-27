#!/usr/bin/env bash
set -euo pipefail

# Exact cached top-k window sweep:
#   - reuse existing component/head scan rankings
#   - train each unique MLP top-k once
#   - train each unique suppress/boost head top-k once
#   - compose the cached actuators into the requested windows
#
# This avoids both repeated scans and repeated actuator training.

cd "$(dirname "${BASH_SOURCE[0]}")/.."

GPU="${GPU:-3}"
TASK="${TASK:-qa}"
MODEL="${MODEL:-LOCAL_HOME/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct}"
BASE_OUT="${BASE_OUT:-data_ckplug/cast_confiqa_topk_window_cached_${TASK}_v0}"

DISCOVERY_SOURCE_ROOT="${DISCOVERY_SOURCE_ROOT:-data_ckplug/cast_confiqa_all3_smalltrain_heldout5000_v0/qa}"
COMPONENT_SCREEN="${COMPONENT_SCREEN:-$DISCOVERY_SOURCE_ROOT/discovery/component_scan/component_screen.csv}"
HEAD_SCAN="${HEAD_SCAN:-$DISCOVERY_SOURCE_ROOT/discovery/head_refine/head_scan.csv}"
SPECS="${SPECS:-mlp4_head4:4:4 mlp2_head4:2:4 mlp4_head2:4:2 mlp4_head6:4:6 mlp2_head6:2:6}"
EXCLUDE_HEAD_LAYERS="${EXCLUDE_HEAD_LAYERS:-}"

FETCH_DIR="${FETCH_DIR:-data_ckplug/context_dpo_confiqa_official}"
TRAIN_SOURCE_ROWS="${TRAIN_SOURCE_ROWS:-300}"
EVAL_SOURCE_START="${EVAL_SOURCE_START:-1000}"
EVAL_SOURCE_ROWS="${EVAL_SOURCE_ROWS:-}"
VAL_MOD="${VAL_MOD:-5}"
TRAIN_ROWS="${TRAIN_ROWS:-240}"
VAL_ROWS="${VAL_ROWS:-60}"
EVAL_ROWS="${EVAL_ROWS:-500}"
EVAL_SPLIT="${EVAL_SPLIT:-all}"

EPOCHS="${EPOCHS:-2}"
MLP_EPOCHS="${MLP_EPOCHS:-$EPOCHS}"
HEAD_EPOCHS="${HEAD_EPOCHS:-$EPOCHS}"
LR="${LR:-0.05}"
LAMBDA_NORM="${LAMBDA_NORM:-1e-4}"
STATE_MARGIN_WEIGHT="${STATE_MARGIN_WEIGHT:-0}"
GAIN_WEIGHT="${GAIN_WEIGHT:-1}"
TARGET_MARGIN="${TARGET_MARGIN:-0.0}"
TARGET_GAIN="${TARGET_GAIN:-0.0}"
SCORE_MODE="${SCORE_MODE:-answer_rest_margin}"
EMPTY_CACHE_EVERY="${EMPTY_CACHE_EVERY:-25}"
MLP_MAX_ALIASES_PER_SIDE="${MLP_MAX_ALIASES_PER_SIDE:-1}"
HEAD_MAX_ALIASES_PER_SIDE="${HEAD_MAX_ALIASES_PER_SIDE:-1}"
ACTUATOR_ALPHA_SWEEP="${ACTUATOR_ALPHA_SWEEP:-0,1}"

MLP_ALPHA="${MLP_ALPHA:-0.05}"
HEAD_ALPHA="${HEAD_ALPHA:-0.5}"
CAST_CONTROLS="${CAST_CONTROLS:-cast_attn=head_act:suppress:${HEAD_ALPHA}:all+head_act:boost:${HEAD_ALPHA}:all;cast_full=head_act:suppress:${HEAD_ALPHA}:all+head_act:boost:${HEAD_ALPHA}:all+comp:prior_mlp:${MLP_ALPHA}:prefill}"

case "$TASK" in
  qa) DATA_JSON="$FETCH_DIR/ConFiQA-QA.json" ;;
  mr) DATA_JSON="$FETCH_DIR/ConFiQA-MR.json" ;;
  mc) DATA_JSON="$FETCH_DIR/ConFiQA-MC.json" ;;
  *) echo "TASK must be qa, mr, or mc; got $TASK" >&2; exit 2 ;;
esac

mkdir -p "$BASE_OUT"
MASTER_LOG="$BASE_OUT/topk_window_cached.log"
SUMMARY="$BASE_OUT/topk_window_cached_summary.tsv"
COMMON="$BASE_OUT/common"
PAIRS_DIR="$COMMON/pairs"
TRAIN_OPEN_ROWS="$COMMON/train_open_rows.jsonl"
EVAL_OPEN_ROWS="$COMMON/eval_open_rows.jsonl"

test -f "$COMPONENT_SCREEN"
test -f "$HEAD_SCAN"

echo -e "config\tmlp_topk\thead_topk\tstate_margin_weight\tgain_weight\tcontrol_name\tn\tpc\tpo\tmr\tem\tcontext_only\tprior_only\tboth\tneither\tmean_chars\tout_dir" > "$SUMMARY"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  deactivate || true
fi
source LOCAL_HOME/anaconda3/etc/profile.d/conda.sh
conda activate screscomp
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

unique_mlp="$(for spec in $SPECS; do IFS=: read -r _name m _h <<< "$spec"; echo "$m"; done | sort -n | uniq | tr '\n' ' ')"
unique_head="$(for spec in $SPECS; do IFS=: read -r _name _m h <<< "$spec"; echo "$h"; done | sort -n | uniq | tr '\n' ' ')"

select_window() {
  local out_dir="$1"
  local mlp_topk="$2"
  local head_topk="$3"
  mkdir -p "$out_dir"
  COMPONENT_SCREEN="$COMPONENT_SCREEN" HEAD_SCAN="$HEAD_SCAN" SELECT_OUT="$out_dir" MLP_TOPK="$mlp_topk" HEAD_TOPK="$head_topk" EXCLUDE_HEAD_LAYERS="$EXCLUDE_HEAD_LAYERS" python - <<'PY'
import csv, json, os
from pathlib import Path

component_screen = Path(os.environ["COMPONENT_SCREEN"])
head_scan = Path(os.environ["HEAD_SCAN"])
out = Path(os.environ["SELECT_OUT"])
mlp_topk = int(os.environ["MLP_TOPK"])
head_topk = int(os.environ["HEAD_TOPK"])
excluded_head_layers = {
    value.strip()
    for value in os.environ.get("EXCLUDE_HEAD_LAYERS", "").split(",")
    if value.strip()
}

def f(row, key, default=0.0):
    try:
        value = str(row.get(key, "")).strip()
        return float(value) if value else default
    except Exception:
        return default

component_rows = list(csv.DictReader(component_screen.open("r", encoding="utf-8", newline="")))
mlp_rows = [r for r in component_rows if r.get("component_type") == "mlp"]
mlp_neg = [r for r in mlp_rows if f(r, "mean_delta") < 0]
selected_mlp = sorted(mlp_neg or mlp_rows, key=lambda r: (f(r, "mean_delta"), -f(r, "sign_consistency")))[:mlp_topk]
if not selected_mlp:
    raise SystemExit("No MLP components selected")
prior_csv = out / "prior_mlp_components.csv"
with prior_csv.open("w", encoding="utf-8", newline="") as fp:
    writer = csv.DictWriter(fp, fieldnames=["component_id", "layer_idx", "component_type"])
    writer.writeheader()
    for row in selected_mlp:
        writer.writerow({"component_id": row["component_id"], "layer_idx": row["layer_idx"], "component_type": row["component_type"]})

head_rows = list(csv.DictReader(head_scan.open("r", encoding="utf-8", newline="")))
def select_heads(role):
    role_rows = [
        r for r in head_rows
        if r.get("role") == role and str(r.get("layer_idx", "")).strip() not in excluded_head_layers
    ]
    pool = [r for r in role_rows if f(r, "mean_margin_gain") > 0] or role_rows
    best = {}
    for row in pool:
        head = row["head_id"]
        if head not in best or f(row, "mean_margin_gain") > f(best[head], "mean_margin_gain"):
            best[head] = row
    selected = sorted(best.values(), key=lambda r: (f(r, "mean_margin_gain"), f(r, "gain_positive_rate")), reverse=True)[:head_topk]
    if not selected:
        raise SystemExit(f"No {role} heads selected")
    return selected

suppress = select_heads("suppress")
boost = select_heads("boost")
head_overlap = sorted({r["head_id"] for r in suppress} & {r["head_id"] for r in boost})
if head_overlap:
    raise SystemExit(
        "Ambiguous cross-role head selection: suppress and boost share "
        + ",".join(head_overlap)
        + ". Audit head_scan or lower HEAD_TOPK before training."
    )
suppress_heads = ",".join(r["head_id"] for r in suppress)
boost_heads = ",".join(r["head_id"] for r in boost)
(out / "selected.env").write_text(
    f"PRIOR_MLP_COMPONENTS={prior_csv}\nSUPPRESS_HEADS={suppress_heads}\nBOOST_HEADS={boost_heads}\n",
    encoding="utf-8",
)
(out / "selected_window.json").write_text(json.dumps({
    "mlp_topk": mlp_topk,
    "head_topk": head_topk,
    "excluded_head_layers": sorted(excluded_head_layers, key=lambda value: int(value) if value.isdigit() else value),
    "mlp_components": selected_mlp,
    "suppress_heads": suppress,
    "boost_heads": boost,
}, ensure_ascii=False, indent=2), encoding="utf-8")
PY
}

append_summary() {
  local config="$1"
  local mlp_topk="$2"
  local head_topk="$3"
  local gen_summary="$4"
  local out_dir="$5"
  CONFIG="$config" MLP_TOPK="$mlp_topk" HEAD_TOPK_VALUE="$head_topk" STATE_MARGIN_WEIGHT_VALUE="$STATE_MARGIN_WEIGHT" GAIN_WEIGHT_VALUE="$GAIN_WEIGHT" GEN_SUMMARY="$gen_summary" OUT_DIR_VALUE="$out_dir" SUMMARY="$SUMMARY" python - <<'PY'
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
            os.environ["CONFIG"],
            os.environ["MLP_TOPK"],
            os.environ["HEAD_TOPK_VALUE"],
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

mkdir -p "$COMMON"
if [[ ! -f "$PAIRS_DIR/pairs.csv" || ! -f "$EVAL_OPEN_ROWS" ]]; then
  echo "[$(date -Is)] prepare common rows/pairs" | tee -a "$MASTER_LOG"
  python -m screscomp.cli.cecm_fetch_confiqa --out-dir "$FETCH_DIR" --tasks "$TASK" | tee -a "$MASTER_LOG"
  prepare_train_args=(python -m screscomp.cli.prepare_ckplug_open --dataset confiqa --data_json "$DATA_JSON" --out_jsonl "$TRAIN_OPEN_ROWS" --schema base --alias_policy raw)
  if [[ -n "$TRAIN_SOURCE_ROWS" ]]; then prepare_train_args+=(--max_rows "$TRAIN_SOURCE_ROWS"); fi
  "${prepare_train_args[@]}" | tee -a "$MASTER_LOG"
  prepare_eval_args=(python -m screscomp.cli.prepare_ckplug_open --dataset confiqa --data_json "$DATA_JSON" --out_jsonl "$EVAL_OPEN_ROWS" --schema base --alias_policy raw --start "$EVAL_SOURCE_START")
  if [[ -n "$EVAL_SOURCE_ROWS" ]]; then prepare_eval_args+=(--max_rows "$EVAL_SOURCE_ROWS"); fi
  "${prepare_eval_args[@]}" | tee -a "$MASTER_LOG"
  python -m screscomp.cli.cecm_build_preference_pairs \
    --input-jsonl "$TRAIN_OPEN_ROWS" \
    --out-dir "$PAIRS_DIR" \
    --event source_context_over_prior \
    --prompt-key base_rag \
    --prior-source dataset_orig \
    --val-mod "$VAL_MOD" \
    | tee -a "$MASTER_LOG"
fi

for k in $unique_mlp; do
  select_dir="$BASE_OUT/select_mlp${k}_head1"
  select_window "$select_dir" "$k" 1
  # shellcheck disable=SC1090
  source "$select_dir/selected.env"
  out="$BASE_OUT/actuators/mlp${k}"
  if [[ -f "$out/fixed_actuator.pt" ]]; then
    echo "[$(date -Is)] reuse mlp top$k $out" | tee -a "$MASTER_LOG"
    continue
  fi
  mkdir -p "$out"
  echo "[$(date -Is)] train mlp top$k" | tee -a "$MASTER_LOG"
  CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.cecm_train_fixed_actuator \
    --model "$MODEL" \
    --pairs-csv "$PAIRS_DIR/pairs.csv" \
    --components-csv "$PRIOR_MLP_COMPONENTS" \
    --event source_context_over_prior \
    --train-split train \
    --val-split val \
    --max-train-rows "$TRAIN_ROWS" \
    --max-val-rows "$VAL_ROWS" \
    --epochs "$MLP_EPOCHS" \
    --lr "$LR" \
    --lambda-norm "$LAMBDA_NORM" \
    --alpha-train 1.0 \
    --alpha-sweep "$ACTUATOR_ALPHA_SWEEP" \
    --state-margin-weight "$STATE_MARGIN_WEIGHT" \
    --gain-weight "$GAIN_WEIGHT" \
    --target-margin "$TARGET_MARGIN" \
    --target-gain "$TARGET_GAIN" \
    --apply-mode decision_tokens \
    --score-mode "$SCORE_MODE" \
    --max-aliases-per-side "$MLP_MAX_ALIASES_PER_SIDE" \
    --empty-cache-every "$EMPTY_CACHE_EVERY" \
    --torch-dtype bfloat16 \
    --device cuda \
    --out-dir "$out" \
    > "$out/train.log" 2>&1
done

for k in $unique_head; do
  select_dir="$BASE_OUT/select_mlp1_head${k}"
  select_window "$select_dir" 1 "$k"
  # shellcheck disable=SC1090
  source "$select_dir/selected.env"
  for role in suppress boost; do
    heads_var="SUPPRESS_HEADS"
    if [[ "$role" == "boost" ]]; then heads_var="BOOST_HEADS"; fi
    heads="${!heads_var}"
    out="$BASE_OUT/actuators/head${k}_${role}"
    if [[ -f "$out/head_actuator.pt" ]]; then
      echo "[$(date -Is)] reuse head $role top$k $out" | tee -a "$MASTER_LOG"
      continue
    fi
    mkdir -p "$out"
    echo "[$(date -Is)] train head $role top$k" | tee -a "$MASTER_LOG"
    CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.cecm_train_attention_head_actuator \
      --model "$MODEL" \
      --pairs-csv "$PAIRS_DIR/pairs.csv" \
      --event source_context_over_prior \
      --heads "$heads" \
      --train-split train \
      --val-split val \
      --max-train-rows "$TRAIN_ROWS" \
      --max-val-rows "$VAL_ROWS" \
      --epochs "$HEAD_EPOCHS" \
      --lr "$LR" \
      --lambda-norm "$LAMBDA_NORM" \
      --alpha-train 1.0 \
      --alpha-sweep "$ACTUATOR_ALPHA_SWEEP" \
      --state-margin-weight "$STATE_MARGIN_WEIGHT" \
      --gain-weight "$GAIN_WEIGHT" \
      --target-margin "$TARGET_MARGIN" \
      --target-gain "$TARGET_GAIN" \
      --apply-mode decision_tokens \
      --score-mode "$SCORE_MODE" \
      --max-aliases-per-side "$HEAD_MAX_ALIASES_PER_SIDE" \
      --empty-cache-every "$EMPTY_CACHE_EVERY" \
      --torch-dtype bfloat16 \
      --device cuda \
      --out-dir "$out" \
      > "$out/train.log" 2>&1
  done
done

for spec in $SPECS; do
  IFS=: read -r name mlp_topk head_topk <<< "$spec"
  out="$BASE_OUT/$name/eval_cast_base_rag"
  mkdir -p "$out"
  echo "[$(date -Is)] eval $name mlp=$mlp_topk head=$head_topk controls=$CAST_CONTROLS" | tee -a "$MASTER_LOG"
  CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.cecm_run_joint_actuator_generation \
    --model "$MODEL" \
    --eval-open-rows "$EVAL_OPEN_ROWS" \
    --component-actuators "prior_mlp=$BASE_OUT/actuators/mlp${mlp_topk}/fixed_actuator.pt" \
    --head-actuators "suppress=$BASE_OUT/actuators/head${head_topk}_suppress/head_actuator.pt;boost=$BASE_OUT/actuators/head${head_topk}_boost/head_actuator.pt" \
    --controls "$CAST_CONTROLS" \
    --generation-prompt-key base_rag \
    --prior-source dataset_orig \
    --split "$EVAL_SPLIT" \
    --start 0 \
    --max-rows "$EVAL_ROWS" \
    --generation-apply-mode prefill \
    --max-new-tokens 64 \
    --stop-strings "Q:" \
    --empty-cache-every "$EMPTY_CACHE_EVERY" \
    --torch-dtype bfloat16 \
    --device cuda \
    --out-dir "$out" \
    > "$out/generation.log" 2>&1
  append_summary "$name" "$mlp_topk" "$head_topk" "$out/generation_summary.csv" "$BASE_OUT/$name"
done

echo "[$(date -Is)] done" | tee -a "$MASTER_LOG"
cat "$SUMMARY"
