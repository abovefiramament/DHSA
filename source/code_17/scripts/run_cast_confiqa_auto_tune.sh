#!/usr/bin/env bash
set -euo pipefail

# Lightweight CAST calibration sweep.
#
# This does not rediscover components and does not retrain actuators. It only
# chooses inference-time actuator weights on a calibration slice, usually the
# held-out val rows inside the small calibration pool.

cd "$(dirname "${BASH_SOURCE[0]}")/.."

GPU="${GPU:-3}"
MODEL="${MODEL:-LOCAL_HOME/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct}"
ROOT="${ROOT:-data_ckplug/cast_confiqa_all3_smalltrain_heldout5000_v0/qa}"
TUNE_OUT="${TUNE_OUT:-$ROOT/auto_tune}"

TUNE_OPEN_ROWS="${TUNE_OPEN_ROWS:-$ROOT/train_open_rows.jsonl}"
TUNE_SPLIT="${TUNE_SPLIT:-val}"
TUNE_START="${TUNE_START:-0}"
TUNE_ROWS="${TUNE_ROWS:-0}"
VAL_MOD="${VAL_MOD:-5}"
GENERATION_PROMPT_KEY="${GENERATION_PROMPT_KEY:-base_rag}"
EMPTY_CACHE_EVERY="${EMPTY_CACHE_EVERY:-25}"

HEAD_ALPHAS="${HEAD_ALPHAS:-0.35 0.5 0.65 0.8}"
MLP_ALPHAS="${MLP_ALPHAS:-0 0.025 0.05 0.075 0.1}"
TUNE_FAMILIES="${TUNE_FAMILIES:-attn full}"

SELECT_MIN_PC="${SELECT_MIN_PC:-}"
SELECT_MAX_PO="${SELECT_MAX_PO:-}"
SELECT_MAX_MR="${SELECT_MAX_MR:-}"
SELECT_MIN_EM="${SELECT_MIN_EM:-}"
SELECT_MIN_CONTEXT_ONLY="${SELECT_MIN_CONTEXT_ONLY:-}"
SELECT_MAX_PC_DROP="${SELECT_MAX_PC_DROP:-0.04}"
SELECT_MAX_CONTEXT_ONLY_DROP="${SELECT_MAX_CONTEXT_ONLY_DROP:-0.04}"

MLP_OUT="${MLP_OUT:-$ROOT/train_decision_tokens_prior_mlp}"
HEAD_SUPPRESS_OUT="${HEAD_SUPPRESS_OUT:-$ROOT/train_decision_tokens_head_suppress}"
HEAD_BOOST_OUT="${HEAD_BOOST_OUT:-$ROOT/train_decision_tokens_head_boost}"

mkdir -p "$TUNE_OUT"
MASTER_LOG="$TUNE_OUT/auto_tune.log"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  deactivate || true
fi
source LOCAL_HOME/anaconda3/etc/profile.d/conda.sh
conda activate screscomp
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

test -f "$TUNE_OPEN_ROWS"
test -f "$MLP_OUT/fixed_actuator.pt"
test -f "$HEAD_SUPPRESS_OUT/head_actuator.pt"
test -f "$HEAD_BOOST_OUT/head_actuator.pt"

CONTROLS_FILE="$TUNE_OUT/controls.txt"
HEAD_ALPHAS="$HEAD_ALPHAS" \
MLP_ALPHAS="$MLP_ALPHAS" \
TUNE_FAMILIES="$TUNE_FAMILIES" \
python - <<'PY' > "$CONTROLS_FILE"
import os


def parse_values(raw: str) -> list[float]:
    values: list[float] = []
    for item in raw.replace(",", " ").split():
        item = item.strip()
        if item:
            values.append(float(item))
    return values


def slug(value: float) -> str:
    return f"{value:.4g}".replace("-", "m").replace(".", "p")


head_alphas = parse_values(os.environ["HEAD_ALPHAS"])
mlp_alphas = parse_values(os.environ["MLP_ALPHAS"])
families = {item.strip() for item in os.environ["TUNE_FAMILIES"].replace(",", " ").split() if item.strip()}

controls: list[str] = ["base=;"]
for h in head_alphas:
    head = f"head_act:suppress:{h:.6g}:all+head_act:boost:{h:.6g}:all"
    if "attn" in families:
        controls.append(f"attn_h{slug(h)}={head};")
    if "full" in families:
        for m in mlp_alphas:
            controls.append(f"full_h{slug(h)}_m{slug(m)}={head}+comp:prior_mlp:{m:.6g}:prefill;")

print("".join(controls))
PY
CONTROLS="$(cat "$CONTROLS_FILE")"

COMP_ACTUATORS="prior_mlp=$MLP_OUT/fixed_actuator.pt"
HEAD_ACTUATORS="suppress=$HEAD_SUPPRESS_OUT/head_actuator.pt;boost=$HEAD_BOOST_OUT/head_actuator.pt"
GEN_OUT="$TUNE_OUT/generation"
SELECT_OUT="$TUNE_OUT/selection"
mkdir -p "$GEN_OUT" "$SELECT_OUT"

{
  echo "[$(date -Is)] root=$ROOT model=$MODEL tune_open_rows=$TUNE_OPEN_ROWS split=$TUNE_SPLIT rows=$TUNE_ROWS"
  echo "[$(date -Is)] head_alphas=$HEAD_ALPHAS mlp_alphas=$MLP_ALPHAS families=$TUNE_FAMILIES"
  echo "[$(date -Is)] controls=$CONTROLS"
} | tee -a "$MASTER_LOG"

CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.cecm_run_joint_actuator_generation \
  --model "$MODEL" \
  --eval-open-rows "$TUNE_OPEN_ROWS" \
  --component-actuators "$COMP_ACTUATORS" \
  --head-actuators "$HEAD_ACTUATORS" \
  --controls "$CONTROLS" \
  --generation-prompt-key "$GENERATION_PROMPT_KEY" \
  --prior-source dataset_orig \
  --split "$TUNE_SPLIT" \
  --start "$TUNE_START" \
  --max-rows "$TUNE_ROWS" \
  --generation-apply-mode prefill \
  --max-new-tokens 64 \
  --stop-strings "Q:" \
  --empty-cache-every "$EMPTY_CACHE_EVERY" \
  --torch-dtype bfloat16 \
  --device cuda \
  --out-dir "$GEN_OUT" \
  > "$GEN_OUT/generation.log" 2>&1

selector_args=(
  python -m screscomp.cli.cecm_select_cast_tune_config
  --summary-csv "$GEN_OUT/generation_summary.csv"
  --control-plan-csv "$GEN_OUT/control_plan.csv"
  --out-dir "$SELECT_OUT"
  --max-pc-drop-from-best "$SELECT_MAX_PC_DROP"
  --max-context-only-drop-from-best "$SELECT_MAX_CONTEXT_ONLY_DROP"
)
if [[ -n "$SELECT_MIN_PC" ]]; then selector_args+=(--min-pc "$SELECT_MIN_PC"); fi
if [[ -n "$SELECT_MAX_PO" ]]; then selector_args+=(--max-po "$SELECT_MAX_PO"); fi
if [[ -n "$SELECT_MAX_MR" ]]; then selector_args+=(--max-mr "$SELECT_MAX_MR"); fi
if [[ -n "$SELECT_MIN_EM" ]]; then selector_args+=(--min-em "$SELECT_MIN_EM"); fi
if [[ -n "$SELECT_MIN_CONTEXT_ONLY" ]]; then selector_args+=(--min-context-only "$SELECT_MIN_CONTEXT_ONLY"); fi

"${selector_args[@]}" | tee -a "$MASTER_LOG"

echo "[$(date -Is)] best config:"
cat "$SELECT_OUT/best_config.env"
find "$TUNE_OUT" -maxdepth 3 \( -name generation_summary.csv -o -name control_plan.csv -o -name best_config.env -o -name selection_ranked.csv \) -print | sort
