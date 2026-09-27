#!/usr/bin/env bash
set -euo pipefail

# Attention-only joint refinement from an existing role-wise CAST run.
# Reuses trained suppress/boost head groups and refines them together without
# loading the prior-MLP actuator group.

cd "$(dirname "${BASH_SOURCE[0]}")/.."

GPU="${GPU:-3}"
ROOT="${ROOT:-data_ckplug/cast_confiqa_all3_smalltrain_heldout5000_v0/qa}"
OUT="${OUT:-data_ckplug/cast_confiqa_head_joint_refine_qa_v0}"
INIT_GATES="${INIT_GATES:-suppress=0.5;boost=0.5}"
STATE_MARGIN_WEIGHT="${STATE_MARGIN_WEIGHT:-0}"
GAIN_WEIGHT="${GAIN_WEIGHT:-1}"

GPU="$GPU" \
ROOT="$ROOT" \
OUT="$OUT" \
USE_MLP=0 \
INIT_GATES="$INIT_GATES" \
STATE_MARGIN_WEIGHT="$STATE_MARGIN_WEIGHT" \
GAIN_WEIGHT="$GAIN_WEIGHT" \
bash scripts/run_cast_confiqa_joint_unfreeze_smoke.sh
