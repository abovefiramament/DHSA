#!/usr/bin/env bash
set -euo pipefail

# Fixed-budget CAST route for ConFiQA:
#   1) reuse the chosen component/head rankings to select one top-k window
#   2) train prior-MLP, suppress-head, and boost-head actuator groups separately
#   3) run one global joint refinement from those three role-wise initializations
#
# This is the main fixed-top-k training shape. Top-k sweeps and exclusion
# audits should use their own wrappers instead of growing this route into a
# search procedure.

cd "$(dirname "${BASH_SOURCE[0]}")/.."

GPU="${GPU:-3}"
TASK="${TASK:-qa}"
MODEL="${MODEL:-LOCAL_HOME/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct}"
BASE_OUT="${BASE_OUT:-data_ckplug/cast_confiqa_fixed_topk_joint_refine_${TASK}_v0}"

MLP_TOPK="${MLP_TOPK:-4}"
HEAD_TOPK="${HEAD_TOPK:-4}"
CONFIG_NAME="${CONFIG_NAME:-fixed_mlp${MLP_TOPK}_head${HEAD_TOPK}}"
SPECS="${SPECS:-${CONFIG_NAME}:${MLP_TOPK}:${HEAD_TOPK}}"

# Keep empty for the fixed recipe. Safety audits can pass "0" or a comma list.
EXCLUDE_HEAD_LAYERS="${EXCLUDE_HEAD_LAYERS:-}"

mkdir -p "$BASE_OUT"

echo "[$(date -Is)] fixed_topk_joint_refine task=$TASK config=$CONFIG_NAME mlp_topk=$MLP_TOPK head_topk=$HEAD_TOPK"
echo "[$(date -Is)] role-wise prior_mlp/suppress/boost init -> one global joint refinement"
if [[ -n "$EXCLUDE_HEAD_LAYERS" ]]; then
  echo "[$(date -Is)] exclude_head_layers=$EXCLUDE_HEAD_LAYERS"
fi

GPU="$GPU" \
TASK="$TASK" \
MODEL="$MODEL" \
BASE_OUT="$BASE_OUT" \
SPECS="$SPECS" \
EXCLUDE_HEAD_LAYERS="$EXCLUDE_HEAD_LAYERS" \
RUN_JOINT_UNFREEZE=1 \
bash scripts/run_cast_confiqa_topk_window_fast.sh
