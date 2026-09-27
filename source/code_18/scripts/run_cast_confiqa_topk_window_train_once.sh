#!/usr/bin/env bash
set -euo pipefail

# Fast top-k window ablation:
#   1) reuse existing component/head scan rankings
#   2) train only the maximum requested MLP/head windows once
#   3) prune the trained payloads to each smaller top-k window
#   4) run generation for each pruned window
#
# This is a window ablation, not a per-window retraining study.

cd "$(dirname "${BASH_SOURCE[0]}")/.."

GPU="${GPU:-3}"
TASK="${TASK:-qa}"
MODEL="${MODEL:-LOCAL_HOME/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct}"
BASE_OUT="${BASE_OUT:-data_ckplug/cast_confiqa_topk_window_train_once_${TASK}_v0}"

DISCOVERY_SOURCE_ROOT="${DISCOVERY_SOURCE_ROOT:-data_ckplug/cast_confiqa_all3_smalltrain_heldout5000_v0/qa}"
COMPONENT_SCREEN="${COMPONENT_SCREEN:-$DISCOVERY_SOURCE_ROOT/discovery/component_scan/component_screen.csv}"
HEAD_SCAN="${HEAD_SCAN:-$DISCOVERY_SOURCE_ROOT/discovery/head_refine/head_scan.csv}"

SPECS="${SPECS:-mlp4_head4:4:4 mlp2_head4:2:4 mlp4_head2:4:2 mlp4_head6:4:6 mlp2_head6:2:6}"

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
CAST_CONTROLS="${CAST_CONTROLS:-base=;cast_attn=head_act:suppress:${HEAD_ALPHA}:all+head_act:boost:${HEAD_ALPHA}:all;cast_full=head_act:suppress:${HEAD_ALPHA}:all+head_act:boost:${HEAD_ALPHA}:all+comp:prior_mlp:${MLP_ALPHA}:prefill}"

mkdir -p "$BASE_OUT"
MASTER_LOG="$BASE_OUT/topk_window_train_once.log"
SUMMARY="$BASE_OUT/topk_window_train_once_summary.tsv"

test -f "$COMPONENT_SCREEN"
test -f "$HEAD_SCAN"

max_mlp=0
max_head=0
for spec in $SPECS; do
  IFS=: read -r _name mlp_topk head_topk <<< "$spec"
  if [[ -z "${mlp_topk:-}" || -z "${head_topk:-}" ]]; then
    echo "Bad spec '$spec'; expected name:mlp_topk:head_topk" >&2
    exit 2
  fi
  if (( mlp_topk > max_mlp )); then max_mlp="$mlp_topk"; fi
  if (( head_topk > max_head )); then max_head="$head_topk"; fi
done

TRAIN_ROOT="$BASE_OUT/train_max_mlp${max_mlp}_head${max_head}"
MAX_SELECT="$TRAIN_ROOT/window_selected"
mkdir -p "$MAX_SELECT"

echo -e "config\tstage\tmlp_topk\thead_topk\ttrained_mlp_topk\ttrained_head_topk\tstate_margin_weight\tgain_weight\tcontrol_name\tn\tpc\tpo\tmr\tem\tcontext_only\tprior_only\tboth\tneither\tmean_chars\tout_dir" > "$SUMMARY"

select_window() {
  local out_dir="$1"
  local mlp_topk="$2"
  local head_topk="$3"
  mkdir -p "$out_dir"
  COMPONENT_SCREEN="$COMPONENT_SCREEN" \
  HEAD_SCAN="$HEAD_SCAN" \
  SELECT_OUT="$out_dir" \
  MLP_TOPK="$mlp_topk" \
  HEAD_TOPK="$head_topk" \
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
out.mkdir(parents=True, exist_ok=True)

def f(row, key, default=0.0):
    try:
        value = str(row.get(key, "")).strip()
        return float(value) if value else default
    except Exception:
        return default

component_rows = list(csv.DictReader(component_screen.open("r", encoding="utf-8", newline="")))
mlp_rows = [r for r in component_rows if r.get("component_type") == "mlp"]
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
    role_rows = [r for r in head_rows if r.get("role") == role]
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
    "mlp_components": selected_mlp,
    "suppress_heads": suppress,
    "boost_heads": boost,
}, ensure_ascii=False, indent=2), encoding="utf-8")
print("PRIOR_MLP_COMPONENTS", prior_csv)
print("SUPPRESS_HEADS", suppress_heads)
print("BOOST_HEADS", boost_heads)
PY
}

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
  TRAINED_MLP_TOPK="$max_mlp" \
  TRAINED_HEAD_TOPK="$max_head" \
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
            os.environ["TRAINED_MLP_TOPK"],
            os.environ["TRAINED_HEAD_TOPK"],
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

select_window "$MAX_SELECT" "$max_mlp" "$max_head"
# shellcheck disable=SC1090
source "$MAX_SELECT/selected.env"

if [[ ! -f "$TRAIN_ROOT/train_decision_tokens_prior_mlp/fixed_actuator.pt" \
      || ! -f "$TRAIN_ROOT/train_decision_tokens_head_suppress/head_actuator.pt" \
      || ! -f "$TRAIN_ROOT/train_decision_tokens_head_boost/head_actuator.pt" ]]; then
  echo "[$(date -Is)] train once max_mlp=$max_mlp max_head=$max_head root=$TRAIN_ROOT" | tee -a "$MASTER_LOG"
  GPU="$GPU" \
    TASK="$TASK" \
    MODEL="$MODEL" \
    ROOT="$TRAIN_ROOT" \
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
    EVAL_ROWS=1 \
    EVAL_SPLIT="$EVAL_SPLIT" \
    EMPTY_CACHE_EVERY="$EMPTY_CACHE_EVERY" \
    bash scripts/run_cast_confiqa_min_loop.sh \
    > "$TRAIN_ROOT/nohup.log" 2>&1
else
  echo "[$(date -Is)] reuse trained max payloads from $TRAIN_ROOT" | tee -a "$MASTER_LOG"
fi

for spec in $SPECS; do
  IFS=: read -r name mlp_topk head_topk <<< "$spec"
  root="$BASE_OUT/$name"
  selected="$root/window_selected"
  mkdir -p "$root"
  select_window "$selected" "$mlp_topk" "$head_topk"

  SPEC_ROOT="$root" \
  SELECTED="$selected" \
  TRAIN_ROOT="$TRAIN_ROOT" \
  python - <<'PY'
import csv
import os
import shutil
from pathlib import Path
import torch

spec_root = Path(os.environ["SPEC_ROOT"])
selected = Path(os.environ["SELECTED"])
train_root = Path(os.environ["TRAIN_ROOT"])

mlp_out = spec_root / "train_decision_tokens_prior_mlp"
sup_out = spec_root / "train_decision_tokens_head_suppress"
boost_out = spec_root / "train_decision_tokens_head_boost"
mlp_out.mkdir(parents=True, exist_ok=True)
sup_out.mkdir(parents=True, exist_ok=True)
boost_out.mkdir(parents=True, exist_ok=True)

def load_component_ids(path):
    with path.open("r", encoding="utf-8", newline="") as fp:
        return [row["component_id"] for row in csv.DictReader(fp)]

env = {}
for line in (selected / "selected.env").read_text(encoding="utf-8").splitlines():
    if "=" in line:
        key, value = line.split("=", 1)
        env[key] = value

component_ids = set(load_component_ids(Path(env["PRIOR_MLP_COMPONENTS"])))
suppress_ids = set(x for x in env["SUPPRESS_HEADS"].split(",") if x)
boost_ids = set(x for x in env["BOOST_HEADS"].split(",") if x)

def prune_fixed(src, dst, wanted):
    payload = torch.load(src, map_location="cpu")
    payload["components"] = [row for row in payload["components"] if row["component_id"] in wanted]
    payload["vectors"] = {k: v for k, v in payload["vectors"].items() if k in wanted}
    torch.save(payload, dst)

def prune_head(src, dst, wanted):
    payload = torch.load(src, map_location="cpu")
    payload["heads"] = [row for row in payload["heads"] if row["head_id"] in wanted]
    payload["vectors"] = {k: v for k, v in payload["vectors"].items() if k in wanted}
    torch.save(payload, dst)

prune_fixed(train_root / "train_decision_tokens_prior_mlp" / "fixed_actuator.pt", mlp_out / "fixed_actuator.pt", component_ids)
prune_head(train_root / "train_decision_tokens_head_suppress" / "head_actuator.pt", sup_out / "head_actuator.pt", suppress_ids)
prune_head(train_root / "train_decision_tokens_head_boost" / "head_actuator.pt", boost_out / "head_actuator.pt", boost_ids)

for rel in ("pairs", "eval_open_rows.jsonl", "train_open_rows.jsonl"):
    src = train_root / rel
    dst = spec_root / rel
    if src.is_dir():
        if dst.exists():
            continue
        shutil.copytree(src, dst)
    elif src.exists() and not dst.exists():
        shutil.copy2(src, dst)
PY

  gen_out="$root/eval_cast_base_rag"
  mkdir -p "$gen_out"
  echo "[$(date -Is)] eval window config=$name mlp_topk=$mlp_topk head_topk=$head_topk" | tee -a "$MASTER_LOG"
  CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.cecm_run_joint_actuator_generation \
    --model "$MODEL" \
    --eval-open-rows "$TRAIN_ROOT/eval_open_rows.jsonl" \
    --component-actuators "prior_mlp=$root/train_decision_tokens_prior_mlp/fixed_actuator.pt" \
    --head-actuators "suppress=$root/train_decision_tokens_head_suppress/head_actuator.pt;boost=$root/train_decision_tokens_head_boost/head_actuator.pt" \
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
    --out-dir "$gen_out" \
    > "$gen_out/generation.log" 2>&1

  append_summary "$name" pruned_from_max "$mlp_topk" "$head_topk" "$gen_out/generation_summary.csv" "$root"
done

echo "[$(date -Is)] done" | tee -a "$MASTER_LOG"
cat "$SUMMARY"
