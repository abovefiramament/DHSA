#!/usr/bin/env bash
set -euo pipefail

# Minimal same-base ConFiQA loop:
#   1) fetch one official Context-DPO ConFiQA subset
#   2) convert it to open-generation rows with official aliases
#   3) build target-vs-competing pairs
#   4) discover component/head pools on the train slice
#   5) train component-local CAST/CECM actuators
#   6) evaluate Base / Strong Prompt / CAST, plus optional released Context-DPO adapter
#
# This is intentionally not a tuning sweep. It gives a reproducible first closed loop.

cd "$(dirname "${BASH_SOURCE[0]}")/.."

GPU="${GPU:-0}"
TASK="${TASK:-qa}"  # qa, mr, or mc
MODEL="${MODEL:-LOCAL_HOME/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct}"
CONTEXT_DPO_MODEL="${CONTEXT_DPO_MODEL:-LOCAL_HOME/.cache/huggingface/hub/models--Bibaolong--Context-Faithful-LLaMA-3-8b-instruct}"
ROOT="${ROOT:-data_ckplug/cast_confiqa_min_loop_${TASK}_llama3_v0}"

FETCH_DIR="${FETCH_DIR:-data_ckplug/context_dpo_confiqa_official}"
VAL_MOD="${VAL_MOD:-5}"
MAX_SOURCE_ROWS="${MAX_SOURCE_ROWS:-}"   # legacy: empty = full official subset
TRAIN_SOURCE_ROWS="${TRAIN_SOURCE_ROWS:-$MAX_SOURCE_ROWS}"
EVAL_SOURCE_START="${EVAL_SOURCE_START:-0}"
EVAL_SOURCE_ROWS="${EVAL_SOURCE_ROWS:-$MAX_SOURCE_ROWS}"
TRAIN_ROWS="${TRAIN_ROWS:-160}"
VAL_ROWS="${VAL_ROWS:-40}"
EVAL_ROWS="${EVAL_ROWS:-600}"
EVAL_SPLIT="${EVAL_SPLIT:-val}"
EPOCHS="${EPOCHS:-2}"
MLP_EPOCHS="${MLP_EPOCHS:-$EPOCHS}"
HEAD_EPOCHS="${HEAD_EPOCHS:-$EPOCHS}"
MLP_TRAIN_ROWS="${MLP_TRAIN_ROWS:-$TRAIN_ROWS}"
MLP_VAL_ROWS="${MLP_VAL_ROWS:-$VAL_ROWS}"
HEAD_TRAIN_ROWS="${HEAD_TRAIN_ROWS:-$TRAIN_ROWS}"
HEAD_VAL_ROWS="${HEAD_VAL_ROWS:-$VAL_ROWS}"
LR="${LR:-0.05}"
LAMBDA_NORM="${LAMBDA_NORM:-1e-4}"
STATE_MARGIN_WEIGHT="${STATE_MARGIN_WEIGHT:-0.0}"
GAIN_WEIGHT="${GAIN_WEIGHT:-1.0}"
TARGET_MARGIN="${TARGET_MARGIN:-0.0}"
TARGET_GAIN="${TARGET_GAIN:-0.0}"
SCORE_MODE="${SCORE_MODE:-answer_rest_margin}"
EMPTY_CACHE_EVERY="${EMPTY_CACHE_EVERY:-25}"
MLP_MAX_ALIASES_PER_SIDE="${MLP_MAX_ALIASES_PER_SIDE:-1}"

DISCOVER_COMPONENTS="${DISCOVER_COMPONENTS:-1}"
REUSE_DISCOVERY="${REUSE_DISCOVERY:-0}"
DISCOVERY_ROWS="${DISCOVERY_ROWS:-60}"
DISCOVERY_TOPK_MLP="${DISCOVERY_TOPK_MLP:-4}"
DISCOVERY_TOPK_ATTN_LAYERS="${DISCOVERY_TOPK_ATTN_LAYERS:-4}"
HEAD_SCAN_ROWS="${HEAD_SCAN_ROWS:-24}"
HEAD_SCAN_FACTORS="${HEAD_SCAN_FACTORS:-0.0,1.5}"
HEAD_TOPK="${HEAD_TOPK:-4}"
HEAD_REFINE_EVAL_ROWS="${HEAD_REFINE_EVAL_ROWS:-1}"

SUPPRESS_HEADS="${SUPPRESS_HEADS:-L9.attn.h17,L31.attn.h11,L9.attn.h12,L9.attn.h10}"
BOOST_HEADS="${BOOST_HEADS:-L31.attn.h14,L31.attn.h5,L9.attn.h3,L9.attn.h29}"
PRIOR_MLP_COMPONENTS="${PRIOR_MLP_COMPONENTS:-configs/cecm_hard_prior_mlp_components.csv}"
MLP_ALPHA="${MLP_ALPHA:-0.05}"
HEAD_ALPHA="${HEAD_ALPHA:-0.5}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
RUN_CONTEXT_DPO="${RUN_CONTEXT_DPO:-1}"
AUTO_TUNE="${AUTO_TUNE:-0}"
RUN_BASE_PROMPTS="${RUN_BASE_PROMPTS:-1}"

case "$TASK" in
  qa) DATA_JSON="$FETCH_DIR/ConFiQA-QA.json" ;;
  mr) DATA_JSON="$FETCH_DIR/ConFiQA-MR.json" ;;
  mc) DATA_JSON="$FETCH_DIR/ConFiQA-MC.json" ;;
  *) echo "TASK must be qa, mr, or mc; got $TASK" >&2; exit 2 ;;
esac

mkdir -p "$ROOT"
MASTER_LOG="$ROOT/master.log"
STATUS="$ROOT/status.tsv"
TRAIN_OPEN_ROWS="$ROOT/train_open_rows.jsonl"
EVAL_OPEN_ROWS="$ROOT/eval_open_rows.jsonl"
PAIRS_DIR="$ROOT/pairs"
DISCOVERY_DIR="$ROOT/discovery"

test -d "$MODEL"
if [[ "$DISCOVER_COMPONENTS" != "1" ]]; then
  test -f "$PRIOR_MLP_COMPONENTS"
fi

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  deactivate || true
fi
source LOCAL_HOME/anaconda3/etc/profile.d/conda.sh
conda activate screscomp
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

{
  echo -e "time\tstage\tstatus"
  echo -e "$(date -Is)\tinit\tstarting"
} > "$STATUS"

echo "[$(date -Is)] root=$ROOT task=$TASK model=$MODEL eval_rows=$EVAL_ROWS train_rows=$TRAIN_ROWS" | tee -a "$MASTER_LOG"
echo "[$(date -Is)] train_source_rows=${TRAIN_SOURCE_ROWS:-all} eval_source_start=$EVAL_SOURCE_START eval_source_rows=${EVAL_SOURCE_ROWS:-all}" | tee -a "$MASTER_LOG"

echo -e "$(date -Is)\tfetch_confiqa\trunning" >> "$STATUS"
python -m screscomp.cli.cecm_fetch_confiqa \
  --out-dir "$FETCH_DIR" \
  --tasks "$TASK" \
  | tee -a "$MASTER_LOG"
echo -e "$(date -Is)\tfetch_confiqa\tdone" >> "$STATUS"

echo -e "$(date -Is)\tprepare_train_open_rows\trunning" >> "$STATUS"
prepare_train_args=(
  python -m screscomp.cli.prepare_ckplug_open
  --dataset confiqa
  --data_json "$DATA_JSON"
  --out_jsonl "$TRAIN_OPEN_ROWS"
  --schema base
  --alias_policy raw
)
if [[ -n "$TRAIN_SOURCE_ROWS" ]]; then
  prepare_train_args+=(--max_rows "$TRAIN_SOURCE_ROWS")
fi
"${prepare_train_args[@]}" | tee -a "$MASTER_LOG"
echo -e "$(date -Is)\tprepare_train_open_rows\tdone" >> "$STATUS"

echo -e "$(date -Is)\tprepare_eval_open_rows\trunning" >> "$STATUS"
prepare_eval_args=(
  python -m screscomp.cli.prepare_ckplug_open
  --dataset confiqa
  --data_json "$DATA_JSON"
  --out_jsonl "$EVAL_OPEN_ROWS"
  --schema base
  --alias_policy raw
  --start "$EVAL_SOURCE_START"
)
if [[ -n "$EVAL_SOURCE_ROWS" ]]; then
  prepare_eval_args+=(--max_rows "$EVAL_SOURCE_ROWS")
fi
"${prepare_eval_args[@]}" | tee -a "$MASTER_LOG"
echo -e "$(date -Is)\tprepare_eval_open_rows\tdone" >> "$STATUS"

echo -e "$(date -Is)\tbuild_pairs\trunning" >> "$STATUS"
python -m screscomp.cli.cecm_build_preference_pairs \
  --input-jsonl "$TRAIN_OPEN_ROWS" \
  --out-dir "$PAIRS_DIR" \
  --event source_context_over_prior \
  --prompt-key base_rag \
  --prior-source dataset_orig \
  --val-mod "$VAL_MOD" \
  | tee -a "$MASTER_LOG"
echo -e "$(date -Is)\tbuild_pairs\tdone" >> "$STATUS"

if [[ "$DISCOVER_COMPONENTS" == "1" && "$REUSE_DISCOVERY" == "1" && -f "$DISCOVERY_DIR/selected/selected.env" ]]; then
  # shellcheck disable=SC1090
  source "$DISCOVERY_DIR/selected/selected.env"
  echo "[$(date -Is)] reuse discovery prior_mlp=$PRIOR_MLP_COMPONENTS suppress=$SUPPRESS_HEADS boost=$BOOST_HEADS" | tee -a "$MASTER_LOG"
  echo -e "$(date -Is)\tdiscover_components\treused" >> "$STATUS"
elif [[ "$DISCOVER_COMPONENTS" == "1" ]]; then
  echo -e "$(date -Is)\tdiscover_components\trunning" >> "$STATUS"
  mkdir -p "$DISCOVERY_DIR/component_scan" "$DISCOVERY_DIR/selected" "$DISCOVERY_DIR/head_refine"

  CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.cecm_scan_component_contributions \
    --model "$MODEL" \
    --pairs-csv "$PAIRS_DIR/pairs.csv" \
    --event source_context_over_prior \
    --split train \
    --start 0 \
    --max-rows "$DISCOVERY_ROWS" \
    --component-types attn,mlp \
    --apply-mode decision_tokens \
    --score-mode "$SCORE_MODE" \
    --max-aliases-per-side 0 \
    --min-abs-delta 0.0 \
    --min-sign-consistency 0.50 \
    --allow-ci-cross-zero \
    --max-components-per-direction 16 \
    --torch-dtype bfloat16 \
    --device cuda \
    --out-dir "$DISCOVERY_DIR/component_scan" \
    > "$DISCOVERY_DIR/component_scan.log" 2>&1

  COMPONENT_SCREEN="$DISCOVERY_DIR/component_scan/component_screen.csv" \
  SELECT_OUT="$DISCOVERY_DIR/selected" \
  DISCOVERY_TOPK_MLP="$DISCOVERY_TOPK_MLP" \
  DISCOVERY_TOPK_ATTN_LAYERS="$DISCOVERY_TOPK_ATTN_LAYERS" \
  python - <<'PY'
import csv
import json
import os
from pathlib import Path

screen = Path(os.environ["COMPONENT_SCREEN"])
out = Path(os.environ["SELECT_OUT"])
topk_mlp = int(os.environ["DISCOVERY_TOPK_MLP"])
topk_attn = int(os.environ["DISCOVERY_TOPK_ATTN_LAYERS"])
out.mkdir(parents=True, exist_ok=True)

rows = list(csv.DictReader(screen.open("r", encoding="utf-8", newline="")))
if not rows:
    raise SystemExit(f"No component rows found: {screen}")

def f(row, key, default=0.0):
    try:
        value = str(row.get(key, "")).strip()
        return float(value) if value else default
    except Exception:
        return default

mlp_rows = [r for r in rows if r.get("component_type") == "mlp"]
attn_rows = [r for r in rows if r.get("component_type") == "attn"]
if not mlp_rows:
    raise SystemExit("Component discovery found no MLP rows.")
if not attn_rows:
    raise SystemExit("Component discovery found no attention rows.")

# MLP actuator is the small prior-resistance actuator, so use the strongest
# negative contributors first. If the subset has no negative MLP contribution,
# fall back to absolute contribution and record it.
mlp_neg = [r for r in mlp_rows if f(r, "mean_delta") < 0]
mlp_policy = "top negative MLP by mean_delta"
if mlp_neg:
    selected_mlp = sorted(mlp_neg, key=lambda r: (f(r, "mean_delta"), -f(r, "sign_consistency")))[:topk_mlp]
else:
    mlp_policy = "fallback top absolute MLP by abs_mean_delta"
    selected_mlp = sorted(mlp_rows, key=lambda r: (-f(r, "abs_mean_delta"), -f(r, "sign_consistency")))[:topk_mlp]

# Attention layers are a pool for head-level refinement; keep this selection
# deliberately simple and signed-agnostic, then let head scanning split
# suppress vs boost.
selected_attn = []
seen_layers = set()
for row in sorted(attn_rows, key=lambda r: (-f(r, "abs_mean_delta"), -f(r, "sign_consistency"))):
    layer = str(row["layer_idx"])
    if layer in seen_layers:
        continue
    seen_layers.add(layer)
    selected_attn.append(row)
    if len(selected_attn) >= topk_attn:
        break

if not selected_mlp:
    raise SystemExit("Could not select MLP components.")
if not selected_attn:
    raise SystemExit("Could not select attention layers.")

with (out / "prior_mlp_components.csv").open("w", encoding="utf-8", newline="") as fp:
    writer = csv.DictWriter(fp, fieldnames=["component_id", "layer_idx", "component_type"])
    writer.writeheader()
    for row in selected_mlp:
        writer.writerow({
            "component_id": row["component_id"],
            "layer_idx": row["layer_idx"],
            "component_type": row["component_type"],
        })

attn_layers = ",".join(str(r["layer_idx"]) for r in selected_attn)
(out / "selected.env").write_text(
    "PRIOR_MLP_COMPONENTS=%s\nDISCOVERED_ATTN_LAYERS=%s\n" % (
        out / "prior_mlp_components.csv",
        attn_layers,
    ),
    encoding="utf-8",
)
(out / "selected_components.json").write_text(json.dumps({
    "selection_scope": "train split only",
    "selection_rule": {
        "mlp": mlp_policy,
        "attention_layers": "top unique attention layers by abs_mean_delta; head roles selected by head scan",
    },
    "mlp_components": [
        {k: row[k] for k in row.keys() if k in {"component_id", "layer_idx", "component_type", "mean_delta", "abs_mean_delta", "sign_consistency", "ci95_low", "ci95_high"}}
        for row in selected_mlp
    ],
    "attention_layers": [
        {k: row[k] for k in row.keys() if k in {"component_id", "layer_idx", "component_type", "mean_delta", "abs_mean_delta", "sign_consistency", "ci95_low", "ci95_high"}}
        for row in selected_attn
    ],
}, ensure_ascii=False, indent=2), encoding="utf-8")

print("PRIOR_MLP_COMPONENTS", out / "prior_mlp_components.csv")
print("DISCOVERED_ATTN_LAYERS", attn_layers)
PY

  # shellcheck disable=SC1090
  source "$DISCOVERY_DIR/selected/selected.env"

  CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.cecm_run_attention_head_refine \
    --model "$MODEL" \
    --pairs-csv "$PAIRS_DIR/pairs.csv" \
    --event source_context_over_prior \
    --eval-open-rows "$TRAIN_OPEN_ROWS" \
    --attn-layers "$DISCOVERED_ATTN_LAYERS" \
    --scan-factors "$HEAD_SCAN_FACTORS" \
    --topks "$HEAD_TOPK" \
    --generation-kinds suppress,boost,mixed \
    --generation-apply-modes prefill \
    --generation-prompt-key base_rag \
    --prior-source dataset_orig \
    --split train \
    --scan-start 0 \
    --scan-max-rows "$HEAD_SCAN_ROWS" \
    --start 0 \
    --max-rows "$HEAD_REFINE_EVAL_ROWS" \
    --val-mod "$VAL_MOD" \
    --score-mode "$SCORE_MODE" \
    --score-apply-mode decision_tokens \
    --max-aliases-per-side 0 \
    --max-new-tokens 64 \
    --stop-strings "Q:" \
    --torch-dtype bfloat16 \
    --device cuda \
    --out-dir "$DISCOVERY_DIR/head_refine" \
    > "$DISCOVERY_DIR/head_refine.log" 2>&1

  HEAD_SCAN="$DISCOVERY_DIR/head_refine/head_scan.csv" \
  SELECT_OUT="$DISCOVERY_DIR/selected" \
  HEAD_TOPK="$HEAD_TOPK" \
  python - <<'PY'
import csv
import json
import os
from pathlib import Path

head_scan = Path(os.environ["HEAD_SCAN"])
out = Path(os.environ["SELECT_OUT"])
topk = int(os.environ["HEAD_TOPK"])
rows = list(csv.DictReader(head_scan.open("r", encoding="utf-8", newline="")))
if not rows:
    raise SystemExit(f"No head scan rows found: {head_scan}")

def f(row, key, default=0.0):
    try:
        value = str(row.get(key, "")).strip()
        return float(value) if value else default
    except Exception:
        return default

def select(role):
    role_rows = [r for r in rows if r.get("role") == role]
    positive = [r for r in role_rows if f(r, "mean_margin_gain") > 0]
    pool = positive or role_rows
    if not pool:
        raise SystemExit(f"No {role} head candidates found.")
    best_by_head = {}
    for row in pool:
        head = row["head_id"]
        prev = best_by_head.get(head)
        if prev is None or f(row, "mean_margin_gain") > f(prev, "mean_margin_gain"):
            best_by_head[head] = row
    return sorted(
        best_by_head.values(),
        key=lambda r: (f(r, "mean_margin_gain"), f(r, "gain_positive_rate")),
        reverse=True,
    )[:topk]

suppress = select("suppress")
boost = select("boost")
head_overlap = sorted({r["head_id"] for r in suppress} & {r["head_id"] for r in boost})
if head_overlap:
    raise SystemExit(
        "Ambiguous cross-role head selection: suppress and boost share "
        + ",".join(head_overlap)
        + ". Audit head_scan or lower HEAD_TOPK before training."
    )
suppress_heads = ",".join(r["head_id"] for r in suppress)
boost_heads = ",".join(r["head_id"] for r in boost)
if not suppress_heads or not boost_heads:
    raise SystemExit("Could not select both suppress and boost heads.")

with (out / "selected.env").open("a", encoding="utf-8") as fp:
    fp.write(f"SUPPRESS_HEADS={suppress_heads}\n")
    fp.write(f"BOOST_HEADS={boost_heads}\n")

payload = json.loads((out / "selected_components.json").read_text(encoding="utf-8"))
payload["head_selection_rule"] = {
    "suppress": "top heads by positive mean_margin_gain under suppress scaling; fallback to best available",
    "boost": "top heads by positive mean_margin_gain under boost scaling; fallback to best available",
}
payload["suppress_heads"] = [
    {k: r[k] for k in r.keys() if k in {"head_id", "layer_idx", "head_idx", "factor", "mean_margin_gain", "gain_positive_rate"}}
    for r in suppress
]
payload["boost_heads"] = [
    {k: r[k] for k in r.keys() if k in {"head_id", "layer_idx", "head_idx", "factor", "mean_margin_gain", "gain_positive_rate"}}
    for r in boost
]
(out / "selected_components.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print("SUPPRESS_HEADS", suppress_heads)
print("BOOST_HEADS", boost_heads)
PY

  # shellcheck disable=SC1090
  source "$DISCOVERY_DIR/selected/selected.env"
  echo "[$(date -Is)] discovered prior_mlp=$PRIOR_MLP_COMPONENTS attn_layers=$DISCOVERED_ATTN_LAYERS suppress=$SUPPRESS_HEADS boost=$BOOST_HEADS" | tee -a "$MASTER_LOG"
  echo -e "$(date -Is)\tdiscover_components\tdone" >> "$STATUS"
else
  echo "[$(date -Is)] using fixed components prior_mlp=$PRIOR_MLP_COMPONENTS suppress=$SUPPRESS_HEADS boost=$BOOST_HEADS" | tee -a "$MASTER_LOG"
fi

MLP_OUT="$ROOT/train_decision_tokens_prior_mlp"
HEAD_SUPPRESS_OUT="$ROOT/train_decision_tokens_head_suppress"
HEAD_BOOST_OUT="$ROOT/train_decision_tokens_head_boost"

if [[ "$SKIP_TRAIN" != "1" ]]; then
  mkdir -p "$MLP_OUT" "$HEAD_SUPPRESS_OUT" "$HEAD_BOOST_OUT"

  echo -e "$(date -Is)\ttrain_mlp\trunning" >> "$STATUS"
  CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.cecm_train_fixed_actuator \
    --model "$MODEL" \
    --pairs-csv "$PAIRS_DIR/pairs.csv" \
    --components-csv "$PRIOR_MLP_COMPONENTS" \
    --event source_context_over_prior \
    --train-split train \
    --val-split val \
    --max-train-rows "$MLP_TRAIN_ROWS" \
    --max-val-rows "$MLP_VAL_ROWS" \
    --epochs "$MLP_EPOCHS" \
    --lr "$LR" \
    --lambda-norm "$LAMBDA_NORM" \
    --alpha-train 1.0 \
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
  echo -e "$(date -Is)\ttrain_mlp\tdone" >> "$STATUS"

  train_head() {
    local name="$1"
    local heads="$2"
    local out="$3"
    echo -e "$(date -Is)\ttrain_head_${name}\trunning" >> "$STATUS"
    CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.cecm_train_attention_head_actuator \
      --model "$MODEL" \
      --pairs-csv "$PAIRS_DIR/pairs.csv" \
      --event source_context_over_prior \
      --heads "$heads" \
      --train-split train \
      --val-split val \
      --max-train-rows "$HEAD_TRAIN_ROWS" \
      --max-val-rows "$HEAD_VAL_ROWS" \
      --epochs "$HEAD_EPOCHS" \
      --lr "$LR" \
      --lambda-norm "$LAMBDA_NORM" \
      --alpha-train 1.0 \
      --state-margin-weight "$STATE_MARGIN_WEIGHT" \
      --gain-weight "$GAIN_WEIGHT" \
      --target-margin "$TARGET_MARGIN" \
      --target-gain "$TARGET_GAIN" \
      --apply-mode decision_tokens \
      --score-mode "$SCORE_MODE" \
      --empty-cache-every "$EMPTY_CACHE_EVERY" \
      --torch-dtype bfloat16 \
      --device cuda \
      --out-dir "$out" \
      > "$out/train.log" 2>&1
    echo -e "$(date -Is)\ttrain_head_${name}\tdone" >> "$STATUS"
  }

  train_head suppress "$SUPPRESS_HEADS" "$HEAD_SUPPRESS_OUT"
  train_head boost "$BOOST_HEADS" "$HEAD_BOOST_OUT"
else
  test -f "$MLP_OUT/fixed_actuator.pt"
  test -f "$HEAD_SUPPRESS_OUT/head_actuator.pt"
  test -f "$HEAD_BOOST_OUT/head_actuator.pt"
fi

run_generation() {
  local model="$1"
  local prompt_key="$2"
  local out="$3"
  local controls="$4"
  local comp_actuators="${5:-}"
  local head_actuators="${6:-}"
  mkdir -p "$out"
  echo "[$(date -Is)] generation model=$model prompt=$prompt_key out=$out controls=$controls" | tee -a "$MASTER_LOG"
  CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.cecm_run_joint_actuator_generation \
    --model "$model" \
    --eval-open-rows "$EVAL_OPEN_ROWS" \
    --component-actuators "$comp_actuators" \
    --head-actuators "$head_actuators" \
    --controls "$controls" \
    --generation-prompt-key "$prompt_key" \
    --prior-source dataset_orig \
    --split "$EVAL_SPLIT" \
    --start 0 \
    --max-rows "$EVAL_ROWS" \
    --generation-apply-mode prefill \
    --max-new-tokens 64 \
    --stop-strings "Q:" \
    --torch-dtype bfloat16 \
    --device cuda \
    --out-dir "$out" \
    > "$out/generation.log" 2>&1
}

echo -e "$(date -Is)\tgeneration_base\trunning" >> "$STATUS"
if [[ "$RUN_BASE_PROMPTS" == "1" ]]; then
  run_generation "$MODEL" base_rag "$ROOT/eval_base_base_rag" "base=;"
  run_generation "$MODEL" strong_rag "$ROOT/eval_base_strong_rag" "strong_prompt=;"
else
  echo "[$(date -Is)] skip base/strong prompt generation RUN_BASE_PROMPTS=$RUN_BASE_PROMPTS" | tee -a "$MASTER_LOG"
fi

COMP_ACTUATORS="prior_mlp=$MLP_OUT/fixed_actuator.pt"
HEAD_ACTUATORS="suppress=$HEAD_SUPPRESS_OUT/head_actuator.pt;boost=$HEAD_BOOST_OUT/head_actuator.pt"
if [[ "$AUTO_TUNE" == "1" && -z "${CAST_CONTROLS:-}" ]]; then
  echo -e "$(date -Is)\tauto_tune\trunning" >> "$STATUS"
  GPU="$GPU" \
    MODEL="$MODEL" \
    ROOT="$ROOT" \
    TUNE_OPEN_ROWS="${TUNE_OPEN_ROWS:-$TRAIN_OPEN_ROWS}" \
    VAL_MOD="$VAL_MOD" \
    EMPTY_CACHE_EVERY="$EMPTY_CACHE_EVERY" \
    bash scripts/run_cast_confiqa_auto_tune.sh | tee -a "$MASTER_LOG"
  # shellcheck disable=SC1090
  source "$ROOT/auto_tune/selection/best_config.env"
  echo "[$(date -Is)] auto_tune selected CAST_CONTROLS=$CAST_CONTROLS" | tee -a "$MASTER_LOG"
  echo -e "$(date -Is)\tauto_tune\tdone" >> "$STATUS"
fi
CAST_CONTROLS_DEFAULT="base=;cast_attn=head_act:suppress:${HEAD_ALPHA}:all+head_act:boost:${HEAD_ALPHA}:all;cast_full=head_act:suppress:${HEAD_ALPHA}:all+head_act:boost:${HEAD_ALPHA}:all+comp:prior_mlp:${MLP_ALPHA}:prefill"
CAST_CONTROLS="${CAST_CONTROLS:-$CAST_CONTROLS_DEFAULT}"
run_generation "$MODEL" base_rag "$ROOT/eval_cast_base_rag" "$CAST_CONTROLS" "$COMP_ACTUATORS" "$HEAD_ACTUATORS"
echo -e "$(date -Is)\tgeneration_base\tdone" >> "$STATUS"

if [[ "$RUN_CONTEXT_DPO" == "1" ]]; then
  if [[ -d "$CONTEXT_DPO_MODEL" ]]; then
    export SCRESCOMP_PEFT_BASE_MODEL="$MODEL"
    echo -e "$(date -Is)\tgeneration_context_dpo\trunning" >> "$STATUS"
    run_generation "$CONTEXT_DPO_MODEL" base_rag "$ROOT/eval_context_dpo_base_rag" "context_dpo=;"
    echo -e "$(date -Is)\tgeneration_context_dpo\tdone" >> "$STATUS"
  else
    echo "[$(date -Is)] skip Context-DPO: missing $CONTEXT_DPO_MODEL" | tee -a "$MASTER_LOG"
    echo -e "$(date -Is)\tgeneration_context_dpo\tskipped" >> "$STATUS"
  fi
fi

echo "[$(date -Is)] done" | tee -a "$MASTER_LOG"
echo -e "$(date -Is)\tall\tdone" >> "$STATUS"
find "$ROOT" -maxdepth 2 \( -name generation_summary.csv -o -name run_summary.csv -o -name pair_build_manifest.json -o -name control_plan.csv \) -print | sort
