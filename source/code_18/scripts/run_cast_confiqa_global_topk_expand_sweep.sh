#!/usr/bin/env bash
set -euo pipefail

# Global top-k expansion sweep for CAST / ConFiQA.
#
# Reuse the cached global selector instead of the layer-quota selector.
# Start from the current useful anchor, MLP top4 + head top6, then expand
# both axes without spending the screen on already-covered smaller windows.
# Layer 0 is excluded by default: the first raw global windows that admit
# L0 suppress heads show unstable repetitive generation.
#
# The cached runner retrains each unique MLP/head top-k once and composes
# those actuators for the requested evaluation windows.

cd "$(dirname "${BASH_SOURCE[0]}")/.."

BASE_OUT="${BASE_OUT:-data_ckplug/cast_confiqa_global_topk_expand_no_l0_gain_v0}"
EVAL_ROWS="${EVAL_ROWS:-200}"
SPECS="${SPECS:-mlp4_head6:4:6 mlp4_head8:4:8 mlp4_head10:4:10 mlp6_head6:6:6 mlp6_head8:6:8 mlp6_head10:6:10 mlp8_head6:8:6 mlp8_head8:8:8 mlp8_head10:8:10}"
EXCLUDE_HEAD_LAYERS="${EXCLUDE_HEAD_LAYERS:-0}"

BASE_OUT="$BASE_OUT" \
EVAL_ROWS="$EVAL_ROWS" \
SPECS="$SPECS" \
EXCLUDE_HEAD_LAYERS="$EXCLUDE_HEAD_LAYERS" \
bash scripts/run_cast_confiqa_topk_window_cached.sh
