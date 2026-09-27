#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ROUND_DIR="${ROUND_DIR:?set ROUND_DIR to a round directory such as data_gsm8k/rounds/.../round_00}"

SOURCE_EVAL_NAME="${SOURCE_EVAL_NAME:-gsm8k_val_500}"
SOURCE_EVAL_DIR="${SOURCE_EVAL_DIR:-$ROUND_DIR/eval/$SOURCE_EVAL_NAME}"
SOURCE_RUN_CONFIG="${SOURCE_RUN_CONFIG:-$SOURCE_EVAL_DIR/run_config.json}"
PAIR_MANIFEST="${PAIR_MANIFEST:-$ROUND_DIR/pairs/pair_build_manifest.json}"
PAIRS_CSV="${PAIRS_CSV:-$ROUND_DIR/pairs/pairs.csv}"

MLP_PREFILL_DIR="${MLP_PREFILL_DIR:-$ROUND_DIR/train/train_mlp_positive_prefill}"
MLP_FIRST_DECODE_DIR="${MLP_FIRST_DECODE_DIR:-$ROUND_DIR/train/train_mlp_positive_first_decode}"
HEAD_SUPPRESS_DIR="${HEAD_SUPPRESS_DIR:-$ROUND_DIR/train/train_head_suppress}"
HEAD_BOOST_DIR="${HEAD_BOOST_DIR:-$ROUND_DIR/train/train_head_boost}"

MLP_PREFILL_PATH="${MLP_PREFILL_PATH:-$MLP_PREFILL_DIR/fixed_actuator.pt}"
MLP_FIRST_DECODE_PATH="${MLP_FIRST_DECODE_PATH:-$MLP_FIRST_DECODE_DIR/fixed_actuator.pt}"
HEAD_SUPPRESS_PATH="${HEAD_SUPPRESS_PATH:-$HEAD_SUPPRESS_DIR/head_actuator.pt}"
HEAD_BOOST_PATH="${HEAD_BOOST_PATH:-$HEAD_BOOST_DIR/head_actuator.pt}"

TRAIN_OUT_NAME="${TRAIN_OUT_NAME:-train_conditional_all_groups}"
TRAIN_OUT_DIR="${TRAIN_OUT_DIR:-$ROUND_DIR/conditional/$TRAIN_OUT_NAME}"
OUT_NAME="${OUT_NAME:-${SOURCE_EVAL_NAME}_conditional_all_groups}"
OUT_DIR="${OUT_DIR:-$ROUND_DIR/eval/$OUT_NAME}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-1000}"
MAX_VAL_ROWS="${MAX_VAL_ROWS:-500}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-0.03}"
LAMBDA_ALPHA="${LAMBDA_ALPHA:-1e-3}"
STATE_MARGIN_WEIGHT="${STATE_MARGIN_WEIGHT:-0.0}"
GAIN_WEIGHT="${GAIN_WEIGHT:-1.0}"
TARGET_MARGIN="${TARGET_MARGIN:-0.0}"
TARGET_GAIN="${TARGET_GAIN:-0.0}"
COMPONENT_ALPHA_MAX="${COMPONENT_ALPHA_MAX:-0.02}"
HEAD_ALPHA_MAX="${HEAD_ALPHA_MAX:-0.04}"
INIT_ALPHAS="${INIT_ALPHAS:-mlp_prefill=0.005;mlp_first_decode=0.0025;head_suppress=0.01;head_boost=0.01}"
INIT_DIRECTION_THRESHOLDS="${INIT_DIRECTION_THRESHOLDS:-}"
DIRECTION_TEMPERATURE="${DIRECTION_TEMPERATURE:-8.0}"
EMPTY_CACHE_EVERY="${EMPTY_CACHE_EVERY:-25}"
REUSE_TRAIN_IF_PRESENT="${REUSE_TRAIN_IF_PRESENT:-1}"
FREEZE_ALPHA_MAX="${FREEZE_ALPHA_MAX:-1}"
SAVE_GATE_MODE="${SAVE_GATE_MODE:-hard}"

SPLIT="${SPLIT:-all}"
START="${START:-0}"
MAX_ROWS="${MAX_ROWS:-500}"
GENERATION_APPLY_MODE="${GENERATION_APPLY_MODE:-prefill}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
STOP_STRINGS="${STOP_STRINGS:-$'\nUser:'}"
FLUSH_EVERY="${FLUSH_EVERY:-1}"
SUMMARY_EVERY="${SUMMARY_EVERY:-50}"
DO_SAMPLE="${DO_SAMPLE:-0}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:-100}"
OVERWRITE="${OVERWRITE:-1}"
BASE_SOURCE_CONTROL_NAME="${BASE_SOURCE_CONTROL_NAME:-base}"

cd "$ROOT"
export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export SOURCE_RUN_CONFIG PAIR_MANIFEST

for required in "$SOURCE_RUN_CONFIG" "$PAIR_MANIFEST" "$PAIRS_CSV" "$MLP_PREFILL_PATH" "$MLP_FIRST_DECODE_PATH" "$HEAD_SUPPRESS_PATH" "$HEAD_BOOST_PATH"; do
  if [[ ! -f "$required" ]]; then
    echo "[cond-all] missing required file: $required" >&2
    exit 1
  fi
done

readarray -t CONFIG_LINES < <(
  "$PYTHON_BIN" - <<'PY'
import json
import os
import shlex
from pathlib import Path

source_run_config = Path(os.environ["SOURCE_RUN_CONFIG"])
pair_manifest = Path(os.environ["PAIR_MANIFEST"])

cfg = json.loads(source_run_config.read_text(encoding="utf-8"))
manifest = json.loads(pair_manifest.read_text(encoding="utf-8"))

def emit(name: str, value: str) -> None:
    print(f"{name}={shlex.quote(str(value))}")

emit("MODEL", cfg["model"])
emit("EVAL_OPEN_ROWS", cfg["eval_open_rows"])
emit("GENERATION_PROMPT_KEY", cfg.get("generation_prompt_key", "deepseek_math"))
emit("PRIOR_SOURCE", cfg.get("prior_source", "model_prior"))
emit("SCORING_KIND", cfg.get("scoring_kind", "gsm8k"))
emit("EVENT", manifest.get("event", ""))
PY
)

for line in "${CONFIG_LINES[@]}"; do
  eval "$line"
done

mkdir -p "$TRAIN_OUT_DIR" "$OUT_DIR"

CAN_REUSE_TRAIN=0
if [[ "$REUSE_TRAIN_IF_PRESENT" == "1" && -f "$TRAIN_OUT_DIR/best_config.env" && -f "$TRAIN_OUT_DIR/conditional_actuator.pt" && -f "$TRAIN_OUT_DIR/run_config.json" ]]; then
  export TRAIN_OUT_DIR FREEZE_ALPHA_MAX SAVE_GATE_MODE INIT_ALPHAS
  if "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path
import torch

run_cfg = json.loads((Path(os.environ["TRAIN_OUT_DIR"]) / "run_config.json").read_text(encoding="utf-8"))
payload = torch.load(Path(os.environ["TRAIN_OUT_DIR"]) / "conditional_actuator.pt", map_location="cpu")
freeze_alpha_max = str(run_cfg.get("freeze_alpha_max", False)).lower() in {"1", "true", "yes"}
expected_freeze = str(os.environ["FREEZE_ALPHA_MAX"]).strip() == "1"
save_gate_mode = str(run_cfg.get("save_gate_mode", "soft")).strip().lower()
expected_gate_mode = str(os.environ["SAVE_GATE_MODE"]).strip().lower()
init_alphas = str(run_cfg.get("init_alphas", ""))
expected_init_alphas = str(os.environ["INIT_ALPHAS"])
payload_version = int(payload.get("payload_format_version", 0) or 0)
ok = (
    freeze_alpha_max == expected_freeze
    and save_gate_mode == expected_gate_mode
    and init_alphas == expected_init_alphas
    and payload_version >= 5
)
raise SystemExit(0 if ok else 1)
PY
  then
    CAN_REUSE_TRAIN=1
  fi
fi

if [[ "$CAN_REUSE_TRAIN" == "1" ]]; then
  echo "[cond-all] reuse existing conditional training payload: $TRAIN_OUT_DIR/conditional_actuator.pt"
else
  train_cmd=(
    "$PYTHON_BIN" -m screscomp.cli.cecm_train_conditional_actuator
    --model "$MODEL"
    --pairs-csv "$PAIRS_CSV"
    --event "$EVENT"
    --train-split train
    --val-split val
    --max-train-rows "$MAX_TRAIN_ROWS"
    --max-val-rows "$MAX_VAL_ROWS"
    --epochs "$EPOCHS"
    --lr "$LR"
    --lambda-alpha "$LAMBDA_ALPHA"
    --state-margin-weight "$STATE_MARGIN_WEIGHT"
    --gain-weight "$GAIN_WEIGHT"
    --target-margin "$TARGET_MARGIN"
    --target-gain "$TARGET_GAIN"
    --score-mode answer_rest_margin
    --component-actuators "mlp_prefill=$MLP_PREFILL_PATH;mlp_first_decode=$MLP_FIRST_DECODE_PATH"
    --head-actuators "head_suppress=$HEAD_SUPPRESS_PATH;head_boost=$HEAD_BOOST_PATH"
    --component-train-apply-modes "mlp_prefill=prompt_last;mlp_first_decode=decision_tokens"
    --head-train-apply-modes "head_suppress=all;head_boost=all"
    --component-alpha-max "$COMPONENT_ALPHA_MAX"
    --head-alpha-max "$HEAD_ALPHA_MAX"
    --init-alphas "$INIT_ALPHAS"
    --direction-temperature "$DIRECTION_TEMPERATURE"
    --save-gate-mode "$SAVE_GATE_MODE"
    --empty-cache-every "$EMPTY_CACHE_EVERY"
    --torch-dtype bfloat16
    --device cuda
    --out-dir "$TRAIN_OUT_DIR"
  )
  if [[ "$FREEZE_ALPHA_MAX" == "1" ]]; then
    train_cmd+=(--freeze-alpha-max)
  fi
  if [[ -n "$INIT_DIRECTION_THRESHOLDS" ]]; then
    train_cmd+=(--init-direction-thresholds "$INIT_DIRECTION_THRESHOLDS")
  fi

  echo "[cond-all] training conditional controller"
  echo "[cond-all] train_out=$TRAIN_OUT_DIR"
  if [[ "$REUSE_TRAIN_IF_PRESENT" == "1" ]]; then
    echo "[cond-all] existing training payload is missing the new hard-gate/fixed-alpha settings; retraining"
  fi
  "${train_cmd[@]}"
fi

# shellcheck disable=SC1090
source "$TRAIN_OUT_DIR/best_config.env"

EVAL_OVERWRITE="$OVERWRITE"
if [[ "$CAN_REUSE_TRAIN" != "1" && -f "$OUT_DIR/generation_rows.jsonl" && "$EVAL_OVERWRITE" != "1" ]]; then
  echo "[cond-all] forcing eval overwrite because the conditional payload was retrained"
  EVAL_OVERWRITE=1
fi

eval_cmd=(
  "$PYTHON_BIN" -m screscomp.cli.cecm_run_joint_actuator_generation
  --model "$MODEL"
  --eval-open-rows "$EVAL_OPEN_ROWS"
  --conditional-actuators "$CAST_CONDITIONAL_ACTUATORS"
  --controls "$CAST_CONTROLS"
  --generation-prompt-key "$GENERATION_PROMPT_KEY"
  --prior-source "$PRIOR_SOURCE"
  --scoring-kind "$SCORING_KIND"
  --split "$SPLIT"
  --start "$START"
  --max-rows "$MAX_ROWS"
  --generation-apply-mode "$GENERATION_APPLY_MODE"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --stop-strings "$STOP_STRINGS"
  --flush-every "$FLUSH_EVERY"
  --summary-every "$SUMMARY_EVERY"
  --empty-cache-every "$EMPTY_CACHE_EVERY"
  --top-p "$TOP_P"
  --top-k "$TOP_K"
  --torch-dtype bfloat16
  --device cuda
  --out-dir "$OUT_DIR"
)

if [[ "$DO_SAMPLE" == "1" ]]; then
  eval_cmd+=(--do-sample --temperature "$TEMPERATURE")
fi
if [[ "$EVAL_OVERWRITE" == "1" ]]; then
  eval_cmd+=(--overwrite)
fi

echo "[cond-all] eval_out=$OUT_DIR"
echo "[cond-all] controls=$CAST_CONTROLS"
"${eval_cmd[@]}"

SOURCE_SUMMARY="$SOURCE_EVAL_DIR/generation_summary.csv"
TARGET_SUMMARY="$OUT_DIR/generation_summary.csv"
COMPARE_SUMMARY="$OUT_DIR/comparison_generation_summary.csv"
export SOURCE_SUMMARY TARGET_SUMMARY COMPARE_SUMMARY BASE_SOURCE_CONTROL_NAME
if [[ -f "$SOURCE_SUMMARY" && -f "$TARGET_SUMMARY" ]]; then
  "$PYTHON_BIN" - <<'PY'
import csv
import os
from pathlib import Path

source_summary = Path(os.environ["SOURCE_SUMMARY"])
target_summary = Path(os.environ["TARGET_SUMMARY"])
compare_summary = Path(os.environ["COMPARE_SUMMARY"])
base_name = os.environ["BASE_SOURCE_CONTROL_NAME"]

with source_summary.open("r", encoding="utf-8", newline="") as f:
    source_rows = list(csv.DictReader(f))
with target_summary.open("r", encoding="utf-8", newline="") as f:
    target_rows = list(csv.DictReader(f))

base_rows = [row for row in source_rows if row.get("control_name") == base_name]
if not base_rows:
    raise SystemExit(0)
rows = []
rows.extend(base_rows)
rows.extend(row for row in target_rows if row.get("control_name") not in {base_name})
fieldnames = []
for row in rows:
    for key in row.keys():
        if key not in fieldnames:
            fieldnames.append(key)

with compare_summary.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f"[cond-all] wrote comparison summary: {compare_summary}")
PY
fi
