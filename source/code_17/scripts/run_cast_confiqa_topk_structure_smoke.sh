#!/usr/bin/env bash
set -euo pipefail

# Structure-only CAST sweep for ConFiQA QA.
#
# This reruns the low-data closed loop while changing only the selected
# component/head pool sizes:
#   - DISCOVERY_TOPK_MLP: number of prior-resistance MLP components
#   - DISCOVERY_TOPK_ATTN_LAYERS: attention layers scanned for heads
#   - HEAD_TOPK: suppress/boost heads kept after head scan
#
# It is serial by design. Base prompts and Context-DPO are skipped by default;
# the output table contains the CAST rows from eval_cast_base_rag.

cd "$(dirname "${BASH_SOURCE[0]}")/.."

GPU="${GPU:-3}"
TASK="${TASK:-qa}"
MODEL="${MODEL:-LOCAL_HOME/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct}"
BASE_OUT="${BASE_OUT:-data_ckplug/cast_confiqa_topk_structure_smoke_${TASK}_v0}"

SPECS="${SPECS:-mlp4_head4:4:4:4 mlp2_head4:2:4:4 mlp4_head2:4:4:2 mlp4_head6:4:4:6 mlp2_head6:2:4:6}"

TRAIN_SOURCE_ROWS="${TRAIN_SOURCE_ROWS:-300}"
VAL_MOD="${VAL_MOD:-5}"
TRAIN_ROWS="${TRAIN_ROWS:-240}"
VAL_ROWS="${VAL_ROWS:-60}"
EVAL_SOURCE_START="${EVAL_SOURCE_START:-1000}"
EVAL_SOURCE_ROWS="${EVAL_SOURCE_ROWS:-}"
EVAL_ROWS="${EVAL_ROWS:-500}"
EVAL_SPLIT="${EVAL_SPLIT:-all}"

DISCOVERY_ROWS="${DISCOVERY_ROWS:-60}"
HEAD_SCAN_ROWS="${HEAD_SCAN_ROWS:-24}"
HEAD_SCAN_FACTORS="${HEAD_SCAN_FACTORS:-0.0,1.5}"
HEAD_REFINE_EVAL_ROWS="${HEAD_REFINE_EVAL_ROWS:-1}"

EPOCHS="${EPOCHS:-2}"
LR="${LR:-0.05}"
LAMBDA_NORM="${LAMBDA_NORM:-1e-4}"
STATE_MARGIN_WEIGHT="${STATE_MARGIN_WEIGHT:-0.0}"
GAIN_WEIGHT="${GAIN_WEIGHT:-1.0}"
TARGET_MARGIN="${TARGET_MARGIN:-0.0}"
TARGET_GAIN="${TARGET_GAIN:-0.0}"
EMPTY_CACHE_EVERY="${EMPTY_CACHE_EVERY:-25}"

MLP_ALPHA="${MLP_ALPHA:-0.05}"
HEAD_ALPHA="${HEAD_ALPHA:-0.5}"
AUTO_TUNE="${AUTO_TUNE:-0}"
RUN_JOINT_UNFREEZE="${RUN_JOINT_UNFREEZE:-0}"

JOINT_EPOCHS="${JOINT_EPOCHS:-2}"
JOINT_EVAL_ROWS="${JOINT_EVAL_ROWS:-$EVAL_ROWS}"
GATE_LR="${GATE_LR:-0.03}"
VECTOR_LR="${VECTOR_LR:-0.005}"

mkdir -p "$BASE_OUT"
MASTER_LOG="$BASE_OUT/topk_structure.log"
SUMMARY="$BASE_OUT/topk_structure_summary.tsv"

echo -e "config\tstage\tmlp_topk\tattn_layer_topk\thead_topk\tstate_margin_weight\tgain_weight\tcontrol_name\tn\tpc\tpo\tmr\tem\tcontext_only\tprior_only\tboth\tneither\tmean_chars\tout_dir" > "$SUMMARY"

append_summary() {
  local config="$1"
  local stage="$2"
  local mlp_topk="$3"
  local attn_layer_topk="$4"
  local head_topk="$5"
  local gen_summary="$6"
  local out_dir="$7"

  CONFIG="$config" \
  STAGE="$stage" \
  MLP_TOPK="$mlp_topk" \
  ATTN_LAYER_TOPK="$attn_layer_topk" \
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
            os.environ["ATTN_LAYER_TOPK"],
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
  IFS=: read -r name mlp_topk attn_layer_topk head_topk <<< "$spec"
  if [[ -z "${name:-}" || -z "${mlp_topk:-}" || -z "${attn_layer_topk:-}" || -z "${head_topk:-}" ]]; then
    echo "Bad spec '$spec'; expected name:mlp_topk:attn_layer_topk:head_topk" >&2
    exit 2
  fi

  root="$BASE_OUT/$name"
  mkdir -p "$root"
  {
    echo "[$(date -Is)] config=$name mlp_topk=$mlp_topk attn_layer_topk=$attn_layer_topk head_topk=$head_topk root=$root"
    echo "[$(date -Is)] loss state_margin_weight=$STATE_MARGIN_WEIGHT gain_weight=$GAIN_WEIGHT target_margin=$TARGET_MARGIN target_gain=$TARGET_GAIN"
  } | tee -a "$MASTER_LOG"

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
    DISCOVER_COMPONENTS=1 \
    REUSE_DISCOVERY=0 \
    DISCOVERY_ROWS="$DISCOVERY_ROWS" \
    DISCOVERY_TOPK_MLP="$mlp_topk" \
    DISCOVERY_TOPK_ATTN_LAYERS="$attn_layer_topk" \
    HEAD_SCAN_ROWS="$HEAD_SCAN_ROWS" \
    HEAD_SCAN_FACTORS="$HEAD_SCAN_FACTORS" \
    HEAD_TOPK="$head_topk" \
    HEAD_REFINE_EVAL_ROWS="$HEAD_REFINE_EVAL_ROWS" \
    MLP_ALPHA="$MLP_ALPHA" \
    HEAD_ALPHA="$HEAD_ALPHA" \
    RUN_BASE_PROMPTS=0 \
    RUN_CONTEXT_DPO=0 \
    AUTO_TUNE="$AUTO_TUNE" \
    EVAL_ROWS="$EVAL_ROWS" \
    EVAL_SPLIT="$EVAL_SPLIT" \
    EMPTY_CACHE_EVERY="$EMPTY_CACHE_EVERY" \
    bash scripts/run_cast_confiqa_min_loop.sh \
    > "$root/nohup.log" 2>&1

  append_summary "$name" fixed "$mlp_topk" "$attn_layer_topk" "$head_topk" "$root/eval_cast_base_rag/generation_summary.csv" "$root"

  if [[ "$RUN_JOINT_UNFREEZE" == "1" ]]; then
    joint_out="$BASE_OUT/${name}_joint"
    mkdir -p "$joint_out"
    echo "[$(date -Is)] joint_unfreeze config=$name out=$joint_out" | tee -a "$MASTER_LOG"
    GPU="$GPU" \
      MODEL="$MODEL" \
      ROOT="$root" \
      OUT="$joint_out" \
      TRAIN_ROWS="$TRAIN_ROWS" \
      VAL_ROWS="$VAL_ROWS" \
      EPOCHS="$JOINT_EPOCHS" \
      GATE_LR="$GATE_LR" \
      VECTOR_LR="$VECTOR_LR" \
      STATE_MARGIN_WEIGHT="$STATE_MARGIN_WEIGHT" \
      GAIN_WEIGHT="$GAIN_WEIGHT" \
      TARGET_MARGIN="$TARGET_MARGIN" \
      TARGET_GAIN="$TARGET_GAIN" \
      EVAL_ROWS="$JOINT_EVAL_ROWS" \
      EVAL_SPLIT="$EVAL_SPLIT" \
      EMPTY_CACHE_EVERY="$EMPTY_CACHE_EVERY" \
      bash scripts/run_cast_confiqa_joint_unfreeze_smoke.sh \
      > "$joint_out/nohup.log" 2>&1

    append_summary "$name" joint_unfreeze "$mlp_topk" "$attn_layer_topk" "$head_topk" "$joint_out/eval_joint_unfreeze_base_rag/generation_summary.csv" "$joint_out"
  fi
done

echo "[$(date -Is)] done" | tee -a "$MASTER_LOG"
cat "$SUMMARY"
