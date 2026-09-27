#!/usr/bin/env bash
set -euo pipefail

# Fast CAST top-k window sweep.
#
# Reuses an existing component/head scan and only changes the top-k window:
#   - MLP_TOPK: top negative MLP components from component_screen.csv
#   - HEAD_TOPK: top suppress/boost heads from head_scan.csv
#
# No component scan and no head scan are rerun here. Each selected window first
# trains three role-wise actuator groups through run_cast_confiqa_min_loop.sh:
# prior MLP, suppress heads, and boost heads. RUN_JOINT_UNFREEZE=1 then runs one
# global joint refinement from those role-wise initializations.

cd "$(dirname "${BASH_SOURCE[0]}")/.."

GPU="${GPU:-3}"
TASK="${TASK:-qa}"
MODEL="${MODEL:-LOCAL_HOME/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct}"
BASE_OUT="${BASE_OUT:-data_ckplug/cast_confiqa_topk_window_fast_${TASK}_v0}"

DISCOVERY_SOURCE_ROOT="${DISCOVERY_SOURCE_ROOT:-data_ckplug/cast_confiqa_all3_smalltrain_heldout5000_v0/qa}"
COMPONENT_SCREEN="${COMPONENT_SCREEN:-$DISCOVERY_SOURCE_ROOT/discovery/component_scan/component_screen.csv}"
HEAD_SCAN="${HEAD_SCAN:-$DISCOVERY_SOURCE_ROOT/discovery/head_refine/head_scan.csv}"

SPECS="${SPECS:-mlp4_head4:4:4 mlp2_head4:2:4 mlp4_head2:4:2 mlp4_head6:4:6 mlp2_head6:2:6}"
EXCLUDE_HEAD_LAYERS="${EXCLUDE_HEAD_LAYERS:-}"

TRAIN_SOURCE_ROWS="${TRAIN_SOURCE_ROWS:-300}"
VAL_MOD="${VAL_MOD:-5}"
TRAIN_ROWS="${TRAIN_ROWS:-240}"
VAL_ROWS="${VAL_ROWS:-60}"
EVAL_SOURCE_START="${EVAL_SOURCE_START:-1000}"
EVAL_SOURCE_ROWS="${EVAL_SOURCE_ROWS:-}"
EVAL_ROWS="${EVAL_ROWS:-500}"
EVAL_SPLIT="${EVAL_SPLIT:-all}"

EPOCHS="${EPOCHS:-2}"
LR="${LR:-0.05}"
LAMBDA_NORM="${LAMBDA_NORM:-1e-4}"
STATE_MARGIN_WEIGHT="${STATE_MARGIN_WEIGHT:-0}"
GAIN_WEIGHT="${GAIN_WEIGHT:-1}"
TARGET_MARGIN="${TARGET_MARGIN:-0.0}"
TARGET_GAIN="${TARGET_GAIN:-0.0}"
EMPTY_CACHE_EVERY="${EMPTY_CACHE_EVERY:-25}"

MLP_ALPHA="${MLP_ALPHA:-0.05}"
HEAD_ALPHA="${HEAD_ALPHA:-0.5}"
RUN_JOINT_UNFREEZE="${RUN_JOINT_UNFREEZE:-0}"
JOINT_EPOCHS="${JOINT_EPOCHS:-2}"
JOINT_EVAL_ROWS="${JOINT_EVAL_ROWS:-$EVAL_ROWS}"

mkdir -p "$BASE_OUT"
MASTER_LOG="$BASE_OUT/topk_window_fast.log"
SUMMARY="$BASE_OUT/topk_window_fast_summary.tsv"

test -f "$COMPONENT_SCREEN"
test -f "$HEAD_SCAN"

echo -e "config\tstage\tmlp_topk\thead_topk\tstate_margin_weight\tgain_weight\tcontrol_name\tn\tpc\tpo\tmr\tem\tcontext_only\tprior_only\tboth\tneither\tmean_chars\tout_dir" > "$SUMMARY"

append_summary() {
  local config="$1"
  local stage="$2"
  local mlp_topk="$3"
  local head_topk="$4"
  local gen_summary="$5"
  local out_dir="$6"

  CONFIG="$config" \
  STAGE="$stage" \
  MLP_TOPK="$mlp_topk" \
  HEAD_TOPK_VALUE="$head_topk" \
  STATE_MARGIN_WEIGHT_VALUE="$STATE_MARGIN_WEIGHT" \
  GAIN_WEIGHT_VALUE="$GAIN_WEIGHT" \
  GEN_SUMMARY="$gen_summary" \
  OUT_DIR_VALUE="$out_dir" \
  SUMMARY="$SUMMARY" \
  python - <<'PY'
import csv
import os
from pathlib import Path

path = Path(os.environ["GEN_SUMMARY"])
if not path.exists():
    raise SystemExit(f"missing generation summary: {path}")

def pick(row, *names):
    for name in names:
        value = row.get(name, "")
        if value != "":
            return value
    return ""

rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
with Path(os.environ["SUMMARY"]).open("a", encoding="utf-8") as fp:
    for row in rows:
        values = [
            os.environ["CONFIG"],
            os.environ["STAGE"],
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
        ]
        fp.write("\t".join(str(v) for v in values) + "\n")
PY
}

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  deactivate || true
fi
source LOCAL_HOME/anaconda3/etc/profile.d/conda.sh
conda activate screscomp
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

for spec in $SPECS; do
  IFS=: read -r name mlp_topk head_topk <<< "$spec"
  if [[ -z "${name:-}" || -z "${mlp_topk:-}" || -z "${head_topk:-}" ]]; then
    echo "Bad spec '$spec'; expected name:mlp_topk:head_topk" >&2
    exit 2
  fi

  root="$BASE_OUT/$name"
  selected="$root/window_selected"
  mkdir -p "$selected"

  echo "[$(date -Is)] config=$name mlp_topk=$mlp_topk head_topk=$head_topk root=$root" | tee -a "$MASTER_LOG"
  echo "[$(date -Is)] reuse component_screen=$COMPONENT_SCREEN head_scan=$HEAD_SCAN" | tee -a "$MASTER_LOG"

  COMPONENT_SCREEN="$COMPONENT_SCREEN" \
  HEAD_SCAN="$HEAD_SCAN" \
  SELECT_OUT="$selected" \
  MLP_TOPK="$mlp_topk" \
  HEAD_TOPK="$head_topk" \
  EXCLUDE_HEAD_LAYERS="$EXCLUDE_HEAD_LAYERS" \
  python - <<'PY'
import csv
import json
import os
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
out.mkdir(parents=True, exist_ok=True)

def f(row, key, default=0.0):
    try:
        value = str(row.get(key, "")).strip()
        return float(value) if value else default
    except Exception:
        return default

rows = list(csv.DictReader(component_screen.open("r", encoding="utf-8", newline="")))
mlp_rows = [r for r in rows if r.get("component_type") == "mlp"]
mlp_neg = [r for r in mlp_rows if f(r, "mean_delta") < 0]
if mlp_neg:
    selected_mlp = sorted(mlp_neg, key=lambda r: (f(r, "mean_delta"), -f(r, "sign_consistency")))[:mlp_topk]
else:
    selected_mlp = sorted(mlp_rows, key=lambda r: (-f(r, "abs_mean_delta"), -f(r, "sign_consistency")))[:mlp_topk]
if not selected_mlp:
    raise SystemExit("Could not select MLP components.")

prior_csv = out / "prior_mlp_components.csv"
with prior_csv.open("w", encoding="utf-8", newline="") as fp:
    writer = csv.DictWriter(fp, fieldnames=["component_id", "layer_idx", "component_type"])
    writer.writeheader()
    for row in selected_mlp:
        writer.writerow({
            "component_id": row["component_id"],
            "layer_idx": row["layer_idx"],
            "component_type": row["component_type"],
        })

head_rows = list(csv.DictReader(head_scan.open("r", encoding="utf-8", newline="")))

def select_heads(role):
    role_rows = [
        r for r in head_rows
        if r.get("role") == role and str(r.get("layer_idx", "")).strip() not in excluded_head_layers
    ]
    positive = [r for r in role_rows if f(r, "mean_margin_gain") > 0]
    pool = positive or role_rows
    best_by_head = {}
    for row in pool:
        head = row["head_id"]
        prev = best_by_head.get(head)
        if prev is None or f(row, "mean_margin_gain") > f(prev, "mean_margin_gain"):
            best_by_head[head] = row
    selected = sorted(
        best_by_head.values(),
        key=lambda r: (f(r, "mean_margin_gain"), f(r, "gain_positive_rate")),
        reverse=True,
    )[:head_topk]
    if not selected:
        raise SystemExit(f"Could not select {role} heads.")
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
    f"PRIOR_MLP_COMPONENTS={prior_csv}\n"
    f"SUPPRESS_HEADS={suppress_heads}\n"
    f"BOOST_HEADS={boost_heads}\n",
    encoding="utf-8",
)
(out / "selected_window.json").write_text(json.dumps({
    "selection_scope": "reused existing scan; top-k window only",
    "component_screen": str(component_screen),
    "head_scan": str(head_scan),
    "mlp_topk": mlp_topk,
    "head_topk": head_topk,
    "excluded_head_layers": sorted(excluded_head_layers, key=lambda value: int(value) if value.isdigit() else value),
    "mlp_components": selected_mlp,
    "suppress_heads": suppress,
    "boost_heads": boost,
}, ensure_ascii=False, indent=2), encoding="utf-8")
print("PRIOR_MLP_COMPONENTS", prior_csv)
print("SUPPRESS_HEADS", suppress_heads)
print("BOOST_HEADS", boost_heads)
PY

  # shellcheck disable=SC1090
  source "$selected/selected.env"

  GPU="$GPU" \
    TASK="$TASK" \
    MODEL="$MODEL" \
    ROOT="$root" \
    TRAIN_SOURCE_ROWS="$TRAIN_SOURCE_ROWS" \
    EVAL_SOURCE_START="$EVAL_SOURCE_START" \
    EVAL_SOURCE_ROWS="$EVAL_SOURCE_ROWS" \
    VAL_MOD="$VAL_MOD" \
    TRAIN_ROWS="$TRAIN_ROWS" \
    VAL_ROWS="$VAL_ROWS" \
    MLP_TRAIN_ROWS="$TRAIN_ROWS" \
    MLP_VAL_ROWS="$VAL_ROWS" \
    HEAD_TRAIN_ROWS="$TRAIN_ROWS" \
    HEAD_VAL_ROWS="$VAL_ROWS" \
    EPOCHS="$EPOCHS" \
    MLP_EPOCHS="$EPOCHS" \
    HEAD_EPOCHS="$EPOCHS" \
    LR="$LR" \
    LAMBDA_NORM="$LAMBDA_NORM" \
    STATE_MARGIN_WEIGHT="$STATE_MARGIN_WEIGHT" \
    GAIN_WEIGHT="$GAIN_WEIGHT" \
    TARGET_MARGIN="$TARGET_MARGIN" \
    TARGET_GAIN="$TARGET_GAIN" \
    DISCOVER_COMPONENTS=0 \
    PRIOR_MLP_COMPONENTS="$PRIOR_MLP_COMPONENTS" \
    SUPPRESS_HEADS="$SUPPRESS_HEADS" \
    BOOST_HEADS="$BOOST_HEADS" \
    MLP_ALPHA="$MLP_ALPHA" \
    HEAD_ALPHA="$HEAD_ALPHA" \
    RUN_BASE_PROMPTS=0 \
    RUN_CONTEXT_DPO=0 \
    AUTO_TUNE=0 \
    EVAL_ROWS="$EVAL_ROWS" \
    EVAL_SPLIT="$EVAL_SPLIT" \
    EMPTY_CACHE_EVERY="$EMPTY_CACHE_EVERY" \
    bash scripts/run_cast_confiqa_min_loop.sh \
    > "$root/nohup.log" 2>&1

  append_summary "$name" fixed "$mlp_topk" "$head_topk" "$root/eval_cast_base_rag/generation_summary.csv" "$root"

  if [[ "$RUN_JOINT_UNFREEZE" == "1" ]]; then
    joint_out="$BASE_OUT/${name}_joint"
    mkdir -p "$joint_out"
    GPU="$GPU" \
      MODEL="$MODEL" \
      ROOT="$root" \
      OUT="$joint_out" \
      TRAIN_ROWS="$TRAIN_ROWS" \
      VAL_ROWS="$VAL_ROWS" \
      EPOCHS="$JOINT_EPOCHS" \
      STATE_MARGIN_WEIGHT="$STATE_MARGIN_WEIGHT" \
      GAIN_WEIGHT="$GAIN_WEIGHT" \
      TARGET_MARGIN="$TARGET_MARGIN" \
      TARGET_GAIN="$TARGET_GAIN" \
      EVAL_ROWS="$JOINT_EVAL_ROWS" \
      EVAL_SPLIT="$EVAL_SPLIT" \
      EMPTY_CACHE_EVERY="$EMPTY_CACHE_EVERY" \
      bash scripts/run_cast_confiqa_joint_unfreeze_smoke.sh \
      > "$joint_out/nohup.log" 2>&1
    append_summary "$name" joint_unfreeze "$mlp_topk" "$head_topk" "$joint_out/eval_joint_unfreeze_base_rag/generation_summary.csv" "$joint_out"
  fi
done

echo "[$(date -Is)] done" | tee -a "$MASTER_LOG"
cat "$SUMMARY"
