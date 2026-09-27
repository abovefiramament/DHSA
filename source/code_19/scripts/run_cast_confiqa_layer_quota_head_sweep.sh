#!/usr/bin/env bash
set -euo pipefail

# Layer-quota head sweep for CAST / ConFiQA:
#   - select top-A attention components/layers from component_scan
#   - within each selected layer, select top-H suppress heads and top-H boost heads
#   - train one pooled suppress actuator and one pooled boost actuator per (A,H)
#   - keep MLP top4 fixed
#
# This is different from global head top-k: every selected attention component
# contributes the same number of heads, so the experiment matches the
# "component -> internal heads" semantics more directly.

cd "$(dirname "${BASH_SOURCE[0]}")/.."

GPU="${GPU:-3}"
TASK="${TASK:-qa}"
MODEL="${MODEL:-LOCAL_HOME/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct}"
BASE_OUT="${BASE_OUT:-data_ckplug/cast_confiqa_layer_quota_head_sweep_no_l0_${TASK}_v0}"

DISCOVERY_SOURCE_ROOT="${DISCOVERY_SOURCE_ROOT:-data_ckplug/cast_confiqa_all3_smalltrain_heldout5000_v0/qa}"
COMPONENT_SCREEN="${COMPONENT_SCREEN:-$DISCOVERY_SOURCE_ROOT/discovery/component_scan/component_screen.csv}"
EXCLUDE_HEAD_LAYERS="${EXCLUDE_HEAD_LAYERS-0}"

FETCH_DIR="${FETCH_DIR:-data_ckplug/context_dpo_confiqa_official}"
TRAIN_SOURCE_ROWS="${TRAIN_SOURCE_ROWS:-300}"
EVAL_SOURCE_START="${EVAL_SOURCE_START:-1000}"
EVAL_SOURCE_ROWS="${EVAL_SOURCE_ROWS:-}"
VAL_MOD="${VAL_MOD:-5}"
TRAIN_ROWS="${TRAIN_ROWS:-240}"
VAL_ROWS="${VAL_ROWS:-60}"
EVAL_ROWS="${EVAL_ROWS:-500}"
EVAL_SPLIT="${EVAL_SPLIT:-all}"

MLP_TOPK="${MLP_TOPK:-4}"
ATTN_TOPKS="${ATTN_TOPKS:-4 6}"
HEADS_PER_LAYER_SWEEP="${HEADS_PER_LAYER_SWEEP:-2 4 6}"
MAX_ATTN_TOPK="${MAX_ATTN_TOPK:-6}"

HEAD_SCAN_ROWS="${HEAD_SCAN_ROWS:-24}"
HEAD_SCAN_FACTORS="${HEAD_SCAN_FACTORS:-0.0,1.5}"

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
MASTER_LOG="$BASE_OUT/layer_quota_head_sweep.log"
SUMMARY="$BASE_OUT/layer_quota_head_sweep_summary.tsv"
COMMON="$BASE_OUT/common"
PAIRS_DIR="$COMMON/pairs"
TRAIN_OPEN_ROWS="$COMMON/train_open_rows.jsonl"
EVAL_OPEN_ROWS="$COMMON/eval_open_rows.jsonl"
HEAD_REFINE_DIR="$BASE_OUT/head_refine_top${MAX_ATTN_TOPK}"
HEAD_SCAN="$HEAD_REFINE_DIR/head_scan.csv"

test -f "$COMPONENT_SCREEN"

echo -e "config\tattn_topk\theads_per_layer\tmlp_topk\tstate_margin_weight\tgain_weight\tcontrol_name\tn\tpc\tpo\tmr\tem\tcontext_only\tprior_only\tboth\tneither\tmean_chars\tout_dir" > "$SUMMARY"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  deactivate || true
fi
source LOCAL_HOME/anaconda3/etc/profile.d/conda.sh
conda activate screscomp
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

append_summary() {
  local config="$1"
  local attn_topk="$2"
  local heads_per_layer="$3"
  local gen_summary="$4"
  local out_dir="$5"
  CONFIG="$config" ATTN_TOPK_VALUE="$attn_topk" HEADS_PER_LAYER_VALUE="$heads_per_layer" MLP_TOPK_VALUE="$MLP_TOPK" STATE_MARGIN_WEIGHT_VALUE="$STATE_MARGIN_WEIGHT" GAIN_WEIGHT_VALUE="$GAIN_WEIGHT" GEN_SUMMARY="$gen_summary" OUT_DIR_VALUE="$out_dir" SUMMARY="$SUMMARY" python - <<'PY'
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
            os.environ["ATTN_TOPK_VALUE"],
            os.environ["HEADS_PER_LAYER_VALUE"],
            os.environ["MLP_TOPK_VALUE"],
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

select_mlp_topk() {
  local out_dir="$1"
  mkdir -p "$out_dir"
  COMPONENT_SCREEN="$COMPONENT_SCREEN" SELECT_OUT="$out_dir" MLP_TOPK="$MLP_TOPK" python - <<'PY'
import csv, os
from pathlib import Path

component_screen = Path(os.environ["COMPONENT_SCREEN"])
out = Path(os.environ["SELECT_OUT"])
mlp_topk = int(os.environ["MLP_TOPK"])

def f(row, key, default=0.0):
    try:
        value = str(row.get(key, "")).strip()
        return float(value) if value else default
    except Exception:
        return default

rows = list(csv.DictReader(component_screen.open("r", encoding="utf-8", newline="")))
mlp_rows = [r for r in rows if r.get("component_type") == "mlp"]
mlp_neg = [r for r in mlp_rows if f(r, "mean_delta") < 0]
selected = sorted(mlp_neg or mlp_rows, key=lambda r: (f(r, "mean_delta"), -f(r, "sign_consistency")))[:mlp_topk]
if not selected:
    raise SystemExit("No MLP components selected")
prior_csv = out / "prior_mlp_components.csv"
with prior_csv.open("w", encoding="utf-8", newline="") as fp:
    writer = csv.DictWriter(fp, fieldnames=["component_id", "layer_idx", "component_type"])
    writer.writeheader()
    for row in selected:
        writer.writerow({
            "component_id": row["component_id"],
            "layer_idx": row["layer_idx"],
            "component_type": row["component_type"],
        })
(out / "selected.env").write_text(f"PRIOR_MLP_COMPONENTS={prior_csv}\n", encoding="utf-8")
PY
}

select_attn_layers() {
  local attn_topk="$1"
  COMPONENT_SCREEN="$COMPONENT_SCREEN" ATTN_TOPK="$attn_topk" EXCLUDE_HEAD_LAYERS="$EXCLUDE_HEAD_LAYERS" python - <<'PY'
import csv, os
from pathlib import Path

def f(row, key, default=0.0):
    try:
        value = str(row.get(key, "")).strip()
        return float(value) if value else default
    except Exception:
        return default

rows = list(csv.DictReader(Path(os.environ["COMPONENT_SCREEN"]).open("r", encoding="utf-8", newline="")))
excluded_head_layers = {
    value.strip()
    for value in os.environ.get("EXCLUDE_HEAD_LAYERS", "").split(",")
    if value.strip()
}
attn = [
    r for r in rows
    if r.get("component_type") == "attn" and str(r.get("layer_idx", "")).strip() not in excluded_head_layers
]
selected = []
seen = set()
for row in sorted(attn, key=lambda r: (-f(r, "abs_mean_delta"), -f(r, "sign_consistency"))):
    layer = str(row["layer_idx"])
    if layer in seen:
        continue
    seen.add(layer)
    selected.append(layer)
    if len(selected) >= int(os.environ["ATTN_TOPK"]):
        break
if len(selected) < int(os.environ["ATTN_TOPK"]):
    raise SystemExit(f"Only selected {len(selected)} attention layers")
print(",".join(selected))
PY
}

select_layer_quota_heads() {
  local out_dir="$1"
  local attn_topk="$2"
  local heads_per_layer="$3"
  mkdir -p "$out_dir"
  COMPONENT_SCREEN="$COMPONENT_SCREEN" HEAD_SCAN="$HEAD_SCAN" SELECT_OUT="$out_dir" ATTN_TOPK="$attn_topk" HEADS_PER_LAYER="$heads_per_layer" EXCLUDE_HEAD_LAYERS="$EXCLUDE_HEAD_LAYERS" python - <<'PY'
import csv, json, os
from pathlib import Path

component_screen = Path(os.environ["COMPONENT_SCREEN"])
head_scan = Path(os.environ["HEAD_SCAN"])
out = Path(os.environ["SELECT_OUT"])
attn_topk = int(os.environ["ATTN_TOPK"])
heads_per_layer = int(os.environ["HEADS_PER_LAYER"])
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
attn_rows = [
    r for r in component_rows
    if r.get("component_type") == "attn" and str(r.get("layer_idx", "")).strip() not in excluded_head_layers
]
selected_layers = []
seen_layers = set()
for row in sorted(attn_rows, key=lambda r: (-f(r, "abs_mean_delta"), -f(r, "sign_consistency"))):
    layer = str(row["layer_idx"])
    if layer in seen_layers:
        continue
    seen_layers.add(layer)
    selected_layers.append(layer)
    if len(selected_layers) >= attn_topk:
        break
if len(selected_layers) < attn_topk:
    raise SystemExit(f"Need {attn_topk} attention layers, got {len(selected_layers)}")

head_rows = list(csv.DictReader(head_scan.open("r", encoding="utf-8", newline="")))
selected_by_role = {}
for role in ("suppress", "boost"):
    role_selected = []
    role_audit = []
    for layer in selected_layers:
        layer_rows = [r for r in head_rows if r.get("role") == role and str(r.get("layer_idx")) == layer]
        positive = [r for r in layer_rows if f(r, "mean_margin_gain") > 0]
        pool = positive or layer_rows
        best_by_head = {}
        for row in pool:
            head = row["head_id"]
            if head not in best_by_head or f(row, "mean_margin_gain") > f(best_by_head[head], "mean_margin_gain"):
                best_by_head[head] = row
        selected = sorted(
            best_by_head.values(),
            key=lambda r: (f(r, "mean_margin_gain"), f(r, "gain_positive_rate")),
            reverse=True,
        )[:heads_per_layer]
        if len(selected) < heads_per_layer:
            raise SystemExit(
                f"Layer {layer} role {role} only has {len(selected)} heads for H={heads_per_layer}"
            )
        role_selected.extend(selected)
        role_audit.append({"layer_idx": layer, "heads": selected})
    selected_by_role[role] = role_selected
    (out / f"{role}_heads.txt").write_text(
        ",".join(row["head_id"] for row in role_selected) + "\n",
        encoding="utf-8",
    )

head_overlap = sorted(
    {row["head_id"] for row in selected_by_role["suppress"]}
    & {row["head_id"] for row in selected_by_role["boost"]}
)
if head_overlap:
    raise SystemExit(
        "Ambiguous cross-role head selection: suppress and boost share "
        + ",".join(head_overlap)
        + ". Audit head_scan or lower the layer-quota budget before training."
    )

(out / "selected.env").write_text(
    "SUPPRESS_HEADS=%s\nBOOST_HEADS=%s\n" % (
        ",".join(row["head_id"] for row in selected_by_role["suppress"]),
        ",".join(row["head_id"] for row in selected_by_role["boost"]),
    ),
    encoding="utf-8",
)
(out / "selected_layer_quota_heads.json").write_text(json.dumps({
    "attn_topk": attn_topk,
    "heads_per_layer": heads_per_layer,
    "excluded_head_layers": sorted(excluded_head_layers, key=lambda value: int(value) if value.isdigit() else value),
    "selected_layers": selected_layers,
    "selection_rule": (
        "select top attention layers by abs_mean_delta; within every selected layer, "
        "select top-H suppress heads and top-H boost heads by positive mean_margin_gain"
    ),
    "suppress_heads": selected_by_role["suppress"],
    "boost_heads": selected_by_role["boost"],
}, ensure_ascii=False, indent=2), encoding="utf-8")
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

if [[ ! -f "$HEAD_SCAN" ]]; then
  ATTN_LAYERS="$(select_attn_layers "$MAX_ATTN_TOPK")"
  mkdir -p "$HEAD_REFINE_DIR"
  echo "[$(date -Is)] scan heads for max attn top$MAX_ATTN_TOPK layers=$ATTN_LAYERS" | tee -a "$MASTER_LOG"
  CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.cecm_run_attention_head_refine \
    --model "$MODEL" \
    --pairs-csv "$PAIRS_DIR/pairs.csv" \
    --event source_context_over_prior \
    --eval-open-rows "$TRAIN_OPEN_ROWS" \
    --attn-layers "$ATTN_LAYERS" \
    --scan-factors "$HEAD_SCAN_FACTORS" \
    --topks "1" \
    --generation-kinds suppress,boost \
    --generation-apply-modes prefill \
    --generation-prompt-key base_rag \
    --prior-source dataset_orig \
    --split train \
    --scan-start 0 \
    --scan-max-rows "$HEAD_SCAN_ROWS" \
    --start 0 \
    --max-rows 1 \
    --val-mod "$VAL_MOD" \
    --score-mode "$SCORE_MODE" \
    --score-apply-mode decision_tokens \
    --max-aliases-per-side 0 \
    --max-new-tokens 64 \
    --stop-strings "Q:" \
    --torch-dtype bfloat16 \
    --device cuda \
    --out-dir "$HEAD_REFINE_DIR" \
    > "$HEAD_REFINE_DIR/head_refine.log" 2>&1
fi

MLP_SELECT="$BASE_OUT/select_mlp${MLP_TOPK}"
select_mlp_topk "$MLP_SELECT"
# shellcheck disable=SC1090
source "$MLP_SELECT/selected.env"
MLP_OUT="$BASE_OUT/actuators/mlp${MLP_TOPK}"
if [[ ! -f "$MLP_OUT/fixed_actuator.pt" ]]; then
  mkdir -p "$MLP_OUT"
  echo "[$(date -Is)] train mlp top$MLP_TOPK" | tee -a "$MASTER_LOG"
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
    --out-dir "$MLP_OUT" \
    > "$MLP_OUT/train.log" 2>&1
fi

for attn_topk in $ATTN_TOPKS; do
  for heads_per_layer in $HEADS_PER_LAYER_SWEEP; do
    name="att${attn_topk}_h${heads_per_layer}"
    select_dir="$BASE_OUT/select_${name}"
    select_layer_quota_heads "$select_dir" "$attn_topk" "$heads_per_layer"
    # shellcheck disable=SC1090
    source "$select_dir/selected.env"

    for role in suppress boost; do
      heads_var="SUPPRESS_HEADS"
      if [[ "$role" == "boost" ]]; then heads_var="BOOST_HEADS"; fi
      heads="${!heads_var}"
      out="$BASE_OUT/actuators/${name}_${role}"
      if [[ -f "$out/head_actuator.pt" ]]; then
        echo "[$(date -Is)] reuse head $role $name $out" | tee -a "$MASTER_LOG"
        continue
      fi
      mkdir -p "$out"
      echo "[$(date -Is)] train head $role $name heads=$heads" | tee -a "$MASTER_LOG"
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

    eval_out="$BASE_OUT/$name/eval_cast_base_rag"
    mkdir -p "$eval_out"
    echo "[$(date -Is)] eval $name controls=$CAST_CONTROLS" | tee -a "$MASTER_LOG"
    CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.cecm_run_joint_actuator_generation \
      --model "$MODEL" \
      --eval-open-rows "$EVAL_OPEN_ROWS" \
      --component-actuators "prior_mlp=$MLP_OUT/fixed_actuator.pt" \
      --head-actuators "suppress=$BASE_OUT/actuators/${name}_suppress/head_actuator.pt;boost=$BASE_OUT/actuators/${name}_boost/head_actuator.pt" \
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
      --out-dir "$eval_out" \
      > "$eval_out/generation.log" 2>&1
    append_summary "$name" "$attn_topk" "$heads_per_layer" "$eval_out/generation_summary.csv" "$BASE_OUT/$name"
  done
done

echo "[$(date -Is)] done" | tee -a "$MASTER_LOG"
cat "$SUMMARY"
