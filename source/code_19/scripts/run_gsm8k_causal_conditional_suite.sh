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
MLP_COMPONENTS_CSV="${MLP_COMPONENTS_CSV:-$ROUND_DIR/discovery/selected/mlp_positive_components.csv}"
SUPPRESS_HEADS_FILE="${SUPPRESS_HEADS_FILE:-$ROUND_DIR/discovery/head_scan/selected_suppress_heads.txt}"
BOOST_HEADS_FILE="${BOOST_HEADS_FILE:-$ROUND_DIR/discovery/head_scan/selected_boost_heads.txt}"

FIXED_MLP_PREFILL_DIR="${FIXED_MLP_PREFILL_DIR:-$ROUND_DIR/train/train_mlp_positive_prefill_causal}"
FIXED_MLP_FIRST_DIR="${FIXED_MLP_FIRST_DIR:-$ROUND_DIR/train/train_mlp_positive_first_decode_causal}"
FIXED_HEAD_SUPPRESS_DIR="${FIXED_HEAD_SUPPRESS_DIR:-$ROUND_DIR/train/train_head_suppress_causal}"
FIXED_HEAD_BOOST_DIR="${FIXED_HEAD_BOOST_DIR:-$ROUND_DIR/train/train_head_boost_causal}"

FIXED_SWEEP_NAME="${FIXED_SWEEP_NAME:-gsm8k_val_500_fixed_causal_sweep}"
FIXED_SWEEP_DIR="${FIXED_SWEEP_DIR:-$ROUND_DIR/eval/$FIXED_SWEEP_NAME}"

COND_MLP_NAME="${COND_MLP_NAME:-train_conditional_mlp_causal}"
COND_ATT_NAME="${COND_ATT_NAME:-train_conditional_att_causal}"
COND_FULL_NAME="${COND_FULL_NAME:-train_conditional_full_causal}"
COND_MLP_DIR="${COND_MLP_DIR:-$ROUND_DIR/conditional/$COND_MLP_NAME}"
COND_ATT_DIR="${COND_ATT_DIR:-$ROUND_DIR/conditional/$COND_ATT_NAME}"
COND_FULL_DIR="${COND_FULL_DIR:-$ROUND_DIR/conditional/$COND_FULL_NAME}"

FINAL_OUT_NAME="${FINAL_OUT_NAME:-gsm8k_test_full_conditional_suite_causal}"
FINAL_OUT_DIR="${FINAL_OUT_DIR:-$ROUND_DIR/eval/$FINAL_OUT_NAME}"
ALPHA_ENV_PATH="${ALPHA_ENV_PATH:-$ROUND_DIR/conditional/causal_conditional_suite_alphas.env}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-1000}"
MAX_VAL_ROWS="${MAX_VAL_ROWS:-500}"
PAIR_SOURCE_ROW_LIMIT="${PAIR_SOURCE_ROW_LIMIT:-1500}"
if (( PAIR_SOURCE_ROW_LIMIT <= 0 )); then
  echo "[suite] PAIR_SOURCE_ROW_LIMIT must be positive; got $PAIR_SOURCE_ROW_LIMIT" >&2
  exit 1
fi
MAX_SOURCE_ROW_INDEX="$((PAIR_SOURCE_ROW_LIMIT - 1))"
RAW_PAIRS_CSV="$PAIRS_CSV"
FILTERED_PAIRS_CSV="${FILTERED_PAIRS_CSV:-$ROUND_DIR/pairs/pairs_source_rows_0_${MAX_SOURCE_ROW_INDEX}.csv}"

FIXED_EPOCHS="${FIXED_EPOCHS:-3}"
FIXED_LR="${FIXED_LR:-0.05}"
FIXED_LAMBDA_NORM="${FIXED_LAMBDA_NORM:-1e-4}"
FIXED_ALPHA_TRAIN="${FIXED_ALPHA_TRAIN:-1.0}"
FIXED_STATE_MARGIN_WEIGHT="${FIXED_STATE_MARGIN_WEIGHT:-1.0}"
FIXED_GAIN_WEIGHT="${FIXED_GAIN_WEIGHT:-0.0}"
FIXED_TARGET_MARGIN="${FIXED_TARGET_MARGIN:-0.0}"
FIXED_TARGET_GAIN="${FIXED_TARGET_GAIN:-0.0}"
FIXED_EMPTY_CACHE_EVERY="${FIXED_EMPTY_CACHE_EVERY:-25}"
MAX_ALIASES_PER_SIDE="${MAX_ALIASES_PER_SIDE:-3}"
QUESTION_STATE_WEIGHTS="${QUESTION_STATE_WEIGHTS:-}"

GATE_EPOCHS="${GATE_EPOCHS:-3}"
GATE_LR="${GATE_LR:-0.03}"
GATE_LAMBDA_ALPHA="${GATE_LAMBDA_ALPHA:-1e-3}"
GATE_STATE_MARGIN_WEIGHT="${GATE_STATE_MARGIN_WEIGHT:-0.0}"
GATE_GAIN_WEIGHT="${GATE_GAIN_WEIGHT:-1.0}"
GATE_TARGET_MARGIN="${GATE_TARGET_MARGIN:-0.0}"
GATE_TARGET_GAIN="${GATE_TARGET_GAIN:-0.0}"
GATE_EMPTY_CACHE_EVERY="${GATE_EMPTY_CACHE_EVERY:-25}"
SIMILARITY_TEMPERATURE="${SIMILARITY_TEMPERATURE:-8.0}"
INIT_DIRECTION_THRESHOLDS="${INIT_DIRECTION_THRESHOLDS:-mlp_prefill=-1;mlp_first_decode=-1;head_suppress=-1;head_boost=-1}"
INIT_DIRECTION_UPPER_THRESHOLDS="${INIT_DIRECTION_UPPER_THRESHOLDS:-mlp_prefill=0.9;mlp_first_decode=0.9;head_suppress=0.9;head_boost=0.9}"
PROJECTION_TEMPERATURE="${PROJECTION_TEMPERATURE:-1.0}"
INIT_PROJECTION_THRESHOLDS="${INIT_PROJECTION_THRESHOLDS:-mlp_prefill=-100;mlp_first_decode=-100;head_suppress=-100;head_boost=-100}"
INIT_PROJECTION_UPPER_THRESHOLDS="${INIT_PROJECTION_UPPER_THRESHOLDS:-mlp_prefill=4;mlp_first_decode=4;head_suppress=4;head_boost=4}"
COMPONENT_ALPHA_MAX="${COMPONENT_ALPHA_MAX:-0.02}"
HEAD_ALPHA_MAX="${HEAD_ALPHA_MAX:-0.04}"
SAVE_GATE_MODE="${SAVE_GATE_MODE:-hard}"
FREEZE_ALPHA_MAX="${FREEZE_ALPHA_MAX:-1}"

MLP_ALPHA_GRID="${MLP_ALPHA_GRID:-0.1,0.2,0.3,0.4,0.5}"
ATT_ALPHA_GRID="${ATT_ALPHA_GRID:-0.1,0.2,0.3,0.4,0.5}"
ALPHA_SELECT_TIE_TOL="${ALPHA_SELECT_TIE_TOL:-0.002}"
ALPHA_FLIP_REGRESSION_WEIGHT="${ALPHA_FLIP_REGRESSION_WEIGHT:-1.0}"

REUSE_FIXED_IF_PRESENT="${REUSE_FIXED_IF_PRESENT:-0}"
REUSE_SWEEP_IF_PRESENT="${REUSE_SWEEP_IF_PRESENT:-0}"
REUSE_GATE_IF_PRESENT="${REUSE_GATE_IF_PRESENT:-0}"
OVERWRITE_FIXED_SWEEP="${OVERWRITE_FIXED_SWEEP:-0}"
OVERWRITE_FINAL="${OVERWRITE_FINAL:-1}"
FINAL_INCLUDE_BASE="${FINAL_INCLUDE_BASE:-1}"
RUN_FINAL_EVAL="${RUN_FINAL_EVAL:-1}"

FINAL_SPLIT="${FINAL_SPLIT:-all}"
FINAL_START="${FINAL_START:-0}"
FINAL_MAX_ROWS="${FINAL_MAX_ROWS:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
STOP_STRINGS="${STOP_STRINGS:-$'\nUser:'}"
FLUSH_EVERY="${FLUSH_EVERY:-1}"
SUMMARY_EVERY="${SUMMARY_EVERY:-100}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:-100}"
DO_SAMPLE="${DO_SAMPLE:-0}"
TEMPERATURE="${TEMPERATURE:-1.0}"

cd "$ROOT"
export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export SOURCE_RUN_CONFIG PAIR_MANIFEST SUPPRESS_HEADS_FILE BOOST_HEADS_FILE

PAIR_SOURCE_FILTER_ARGS=(--max-source-row-index "$MAX_SOURCE_ROW_INDEX")
QUESTION_STATE_WEIGHT_ARGS=()
if [[ -n "$QUESTION_STATE_WEIGHTS" ]]; then
  QUESTION_STATE_WEIGHT_ARGS=(--question-state-weights "$QUESTION_STATE_WEIGHTS")
fi
INIT_DIRECTION_THRESHOLD_ARGS=()
if [[ -n "$INIT_DIRECTION_THRESHOLDS" ]]; then
  INIT_DIRECTION_THRESHOLD_ARGS+=(--init-direction-thresholds "$INIT_DIRECTION_THRESHOLDS")
fi
if [[ -n "$INIT_DIRECTION_UPPER_THRESHOLDS" ]]; then
  INIT_DIRECTION_THRESHOLD_ARGS+=(--init-direction-upper-thresholds "$INIT_DIRECTION_UPPER_THRESHOLDS")
fi
if [[ -n "$INIT_PROJECTION_THRESHOLDS" ]]; then
  INIT_DIRECTION_THRESHOLD_ARGS+=(--init-projection-thresholds "$INIT_PROJECTION_THRESHOLDS")
fi
if [[ -n "$INIT_PROJECTION_UPPER_THRESHOLDS" ]]; then
  INIT_DIRECTION_THRESHOLD_ARGS+=(--init-projection-upper-thresholds "$INIT_PROJECTION_UPPER_THRESHOLDS")
fi

for required in "$SOURCE_RUN_CONFIG" "$PAIR_MANIFEST" "$RAW_PAIRS_CSV" "$MLP_COMPONENTS_CSV" "$SUPPRESS_HEADS_FILE" "$BOOST_HEADS_FILE"; do
  if [[ ! -f "$required" ]]; then
    echo "[suite] missing required file: $required" >&2
    exit 1
  fi
done

export RAW_PAIRS_CSV FILTERED_PAIRS_CSV MAX_SOURCE_ROW_INDEX
"$PYTHON_BIN" - <<'PY'
import csv
import os
from pathlib import Path

raw_path = Path(os.environ["RAW_PAIRS_CSV"])
out_path = Path(os.environ["FILTERED_PAIRS_CSV"])
max_row_index = int(os.environ["MAX_SOURCE_ROW_INDEX"])
if raw_path.resolve() == out_path.resolve():
    raise SystemExit("[suite] FILTERED_PAIRS_CSV must differ from the raw PAIRS_CSV")

total = 0
kept = 0
out_path.parent.mkdir(parents=True, exist_ok=True)
with raw_path.open("r", encoding="utf-8", newline="") as fin, out_path.open("w", encoding="utf-8", newline="") as fout:
    reader = csv.DictReader(fin)
    if reader.fieldnames is None:
        raise SystemExit(f"[suite] empty pair csv: {raw_path}")
    writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
    writer.writeheader()
    for row in reader:
        total += 1
        try:
            row_index = int(str(row.get("row_index", "")).strip())
        except ValueError:
            continue
        if row_index > max_row_index:
            continue
        writer.writerow(row)
        kept += 1
if kept == 0:
    raise SystemExit(f"[suite] no pairs remain after row_index <= {max_row_index} filter")
print(f"[suite] pair source filter row_index<= {max_row_index}: kept={kept} excluded={total-kept} out={out_path}", flush=True)
PY
PAIRS_CSV="$FILTERED_PAIRS_CSV"

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
emit("GENERATION_APPLY_MODE", cfg.get("generation_apply_mode", "prefill"))
emit("EVENT", manifest.get("event", ""))
PY
)

for line in "${CONFIG_LINES[@]}"; do
  eval "$line"
done

readarray -t HEAD_LINES < <(
  "$PYTHON_BIN" - <<'PY'
import os
import shlex
from pathlib import Path

def read_heads(env_name: str) -> str:
    path = Path(os.environ[env_name])
    heads = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return ",".join(heads)

print(f"SUPPRESS_HEADS={shlex.quote(read_heads('SUPPRESS_HEADS_FILE'))}")
print(f"BOOST_HEADS={shlex.quote(read_heads('BOOST_HEADS_FILE'))}")
PY
)

for line in "${HEAD_LINES[@]}"; do
  eval "$line"
done
if [[ -z "$SUPPRESS_HEADS" || -z "$BOOST_HEADS" ]]; then
  echo "[suite] selected head files must not be empty" >&2
  exit 1
fi

mkdir -p \
  "$FIXED_MLP_PREFILL_DIR" \
  "$FIXED_MLP_FIRST_DIR" \
  "$FIXED_HEAD_SUPPRESS_DIR" \
  "$FIXED_HEAD_BOOST_DIR" \
  "$FIXED_SWEEP_DIR" \
  "$COND_MLP_DIR" \
  "$COND_ATT_DIR" \
  "$COND_FULL_DIR" \
  "$FINAL_OUT_DIR" \
  "$(dirname "$ALPHA_ENV_PATH")"

can_reuse_fixed_component() {
  local out_dir="$1"
  local expected_mode="$2"
  local payload="$out_dir/fixed_actuator.pt"
  local run_cfg="$out_dir/run_config.json"
  [[ -f "$payload" && -f "$run_cfg" ]] || return 1
  REUSE_RUN_CFG="$run_cfg" EXPECTED_MODE="$expected_mode" EXPECTED_MAX_TRAIN_ROWS="$MAX_TRAIN_ROWS" EXPECTED_MAX_VAL_ROWS="$MAX_VAL_ROWS" EXPECTED_MAX_SOURCE_ROW_INDEX="$MAX_SOURCE_ROW_INDEX" EXPECTED_EPOCHS="$FIXED_EPOCHS" EXPECTED_QUESTION_STATE_WEIGHTS="$QUESTION_STATE_WEIGHTS" \
    "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

def parse_map(raw: str) -> dict[str, float]:
    out = {}
    for part in str(raw or "").split(";"):
        part = part.strip()
        if not part:
            continue
        name, value = part.split("=", 1)
        out[name.strip()] = float(value.strip())
    return out

cfg = json.loads(Path(os.environ["REUSE_RUN_CFG"]).read_text(encoding="utf-8"))
ok = (
    bool(cfg.get("causal_train_mask", False))
    and str(cfg.get("apply_mode", "")) == os.environ["EXPECTED_MODE"]
    and int(cfg.get("max_train_rows")) == int(os.environ["EXPECTED_MAX_TRAIN_ROWS"])
    and int(cfg.get("max_val_rows")) == int(os.environ["EXPECTED_MAX_VAL_ROWS"])
    and int(cfg.get("max_source_row_index")) == int(os.environ["EXPECTED_MAX_SOURCE_ROW_INDEX"])
    and int(cfg.get("epochs")) == int(os.environ["EXPECTED_EPOCHS"])
    and str(cfg.get("score_mode", "")) == "answer_rest_margin"
    and {str(k): float(v) for k, v in dict(cfg.get("question_state_weights", {}) or {}).items()}
        == parse_map(os.environ["EXPECTED_QUESTION_STATE_WEIGHTS"])
)
raise SystemExit(0 if ok else 1)
PY
}

can_reuse_fixed_head() {
  local out_dir="$1"
  local expected_mode="$2"
  local payload="$out_dir/head_actuator.pt"
  local run_cfg="$out_dir/run_config.json"
  [[ -f "$payload" && -f "$run_cfg" ]] || return 1
  REUSE_RUN_CFG="$run_cfg" EXPECTED_MODE="$expected_mode" EXPECTED_MAX_TRAIN_ROWS="$MAX_TRAIN_ROWS" EXPECTED_MAX_VAL_ROWS="$MAX_VAL_ROWS" EXPECTED_MAX_SOURCE_ROW_INDEX="$MAX_SOURCE_ROW_INDEX" EXPECTED_EPOCHS="$FIXED_EPOCHS" EXPECTED_QUESTION_STATE_WEIGHTS="$QUESTION_STATE_WEIGHTS" \
    "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

def parse_map(raw: str) -> dict[str, float]:
    out = {}
    for part in str(raw or "").split(";"):
        part = part.strip()
        if not part:
            continue
        name, value = part.split("=", 1)
        out[name.strip()] = float(value.strip())
    return out

cfg = json.loads(Path(os.environ["REUSE_RUN_CFG"]).read_text(encoding="utf-8"))
ok = (
    bool(cfg.get("causal_train_mask", False))
    and str(cfg.get("apply_mode", "")) == os.environ["EXPECTED_MODE"]
    and int(cfg.get("max_train_rows")) == int(os.environ["EXPECTED_MAX_TRAIN_ROWS"])
    and int(cfg.get("max_val_rows")) == int(os.environ["EXPECTED_MAX_VAL_ROWS"])
    and int(cfg.get("max_source_row_index")) == int(os.environ["EXPECTED_MAX_SOURCE_ROW_INDEX"])
    and int(cfg.get("epochs")) == int(os.environ["EXPECTED_EPOCHS"])
    and str(cfg.get("score_mode", "")) == "answer_rest_margin"
    and {str(k): float(v) for k, v in dict(cfg.get("question_state_weights", {}) or {}).items()}
        == parse_map(os.environ["EXPECTED_QUESTION_STATE_WEIGHTS"])
)
raise SystemExit(0 if ok else 1)
PY
}

RETRAINED_FIXED=0

if [[ "$REUSE_FIXED_IF_PRESENT" == "1" ]] && can_reuse_fixed_component "$FIXED_MLP_PREFILL_DIR" "prompt_last"; then
  echo "[suite] reuse fixed mlp prefill: $FIXED_MLP_PREFILL_DIR/fixed_actuator.pt"
else
  "$PYTHON_BIN" -m screscomp.cli.cecm_train_fixed_actuator \
    --model "$MODEL" \
    --pairs-csv "$PAIRS_CSV" \
    --components-csv "$MLP_COMPONENTS_CSV" \
    --event "$EVENT" \
    --train-split train \
    --val-split val \
    --max-train-rows "$MAX_TRAIN_ROWS" \
    --max-val-rows "$MAX_VAL_ROWS" \
    "${PAIR_SOURCE_FILTER_ARGS[@]}" \
    --epochs "$FIXED_EPOCHS" \
    --lr "$FIXED_LR" \
    --lambda-norm "$FIXED_LAMBDA_NORM" \
    --alpha-train "$FIXED_ALPHA_TRAIN" \
    --state-margin-weight "$FIXED_STATE_MARGIN_WEIGHT" \
    --gain-weight "$FIXED_GAIN_WEIGHT" \
    --target-margin "$FIXED_TARGET_MARGIN" \
    --target-gain "$FIXED_TARGET_GAIN" \
    --apply-mode prompt_last \
    --causal-train-mask \
    --score-mode answer_rest_margin \
    --option-selection-mode model_max \
    --max-aliases-per-side "$MAX_ALIASES_PER_SIDE" \
    "${QUESTION_STATE_WEIGHT_ARGS[@]}" \
    --alpha-sweep "0,${MLP_ALPHA_GRID}" \
    --empty-cache-every "$FIXED_EMPTY_CACHE_EVERY" \
    --torch-dtype bfloat16 \
    --device cuda \
    --out-dir "$FIXED_MLP_PREFILL_DIR"
  RETRAINED_FIXED=1
fi

if [[ "$REUSE_FIXED_IF_PRESENT" == "1" ]] && can_reuse_fixed_component "$FIXED_MLP_FIRST_DIR" "first_decode"; then
  echo "[suite] reuse fixed mlp first_decode: $FIXED_MLP_FIRST_DIR/fixed_actuator.pt"
else
  "$PYTHON_BIN" -m screscomp.cli.cecm_train_fixed_actuator \
    --model "$MODEL" \
    --pairs-csv "$PAIRS_CSV" \
    --components-csv "$MLP_COMPONENTS_CSV" \
    --event "$EVENT" \
    --train-split train \
    --val-split val \
    --max-train-rows "$MAX_TRAIN_ROWS" \
    --max-val-rows "$MAX_VAL_ROWS" \
    "${PAIR_SOURCE_FILTER_ARGS[@]}" \
    --epochs "$FIXED_EPOCHS" \
    --lr "$FIXED_LR" \
    --lambda-norm "$FIXED_LAMBDA_NORM" \
    --alpha-train "$FIXED_ALPHA_TRAIN" \
    --state-margin-weight "$FIXED_STATE_MARGIN_WEIGHT" \
    --gain-weight "$FIXED_GAIN_WEIGHT" \
    --target-margin "$FIXED_TARGET_MARGIN" \
    --target-gain "$FIXED_TARGET_GAIN" \
    --apply-mode first_decode \
    --causal-train-mask \
    --score-mode answer_rest_margin \
    --option-selection-mode model_max \
    --max-aliases-per-side "$MAX_ALIASES_PER_SIDE" \
    "${QUESTION_STATE_WEIGHT_ARGS[@]}" \
    --alpha-sweep "0,${MLP_ALPHA_GRID}" \
    --empty-cache-every "$FIXED_EMPTY_CACHE_EVERY" \
    --torch-dtype bfloat16 \
    --device cuda \
    --out-dir "$FIXED_MLP_FIRST_DIR"
  RETRAINED_FIXED=1
fi

if [[ "$REUSE_FIXED_IF_PRESENT" == "1" ]] && can_reuse_fixed_head "$FIXED_HEAD_SUPPRESS_DIR" "all"; then
  echo "[suite] reuse fixed head suppress: $FIXED_HEAD_SUPPRESS_DIR/head_actuator.pt"
else
  "$PYTHON_BIN" -m screscomp.cli.cecm_train_attention_head_actuator \
    --model "$MODEL" \
    --pairs-csv "$PAIRS_CSV" \
    --event "$EVENT" \
    --heads "$SUPPRESS_HEADS" \
    --train-split train \
    --val-split val \
    --max-train-rows "$MAX_TRAIN_ROWS" \
    --max-val-rows "$MAX_VAL_ROWS" \
    "${PAIR_SOURCE_FILTER_ARGS[@]}" \
    --epochs "$FIXED_EPOCHS" \
    --lr "$FIXED_LR" \
    --lambda-norm "$FIXED_LAMBDA_NORM" \
    --alpha-train "$FIXED_ALPHA_TRAIN" \
    --state-margin-weight "$FIXED_STATE_MARGIN_WEIGHT" \
    --gain-weight "$FIXED_GAIN_WEIGHT" \
    --target-margin "$FIXED_TARGET_MARGIN" \
    --target-gain "$FIXED_TARGET_GAIN" \
    --apply-mode all \
    --causal-train-mask \
    --score-mode answer_rest_margin \
    --option-selection-mode model_max \
    --max-aliases-per-side "$MAX_ALIASES_PER_SIDE" \
    "${QUESTION_STATE_WEIGHT_ARGS[@]}" \
    --alpha-sweep "0,${ATT_ALPHA_GRID}" \
    --empty-cache-every "$FIXED_EMPTY_CACHE_EVERY" \
    --torch-dtype bfloat16 \
    --device cuda \
    --out-dir "$FIXED_HEAD_SUPPRESS_DIR"
  RETRAINED_FIXED=1
fi

if [[ "$REUSE_FIXED_IF_PRESENT" == "1" ]] && can_reuse_fixed_head "$FIXED_HEAD_BOOST_DIR" "all"; then
  echo "[suite] reuse fixed head boost: $FIXED_HEAD_BOOST_DIR/head_actuator.pt"
else
  "$PYTHON_BIN" -m screscomp.cli.cecm_train_attention_head_actuator \
    --model "$MODEL" \
    --pairs-csv "$PAIRS_CSV" \
    --event "$EVENT" \
    --heads "$BOOST_HEADS" \
    --train-split train \
    --val-split val \
    --max-train-rows "$MAX_TRAIN_ROWS" \
    --max-val-rows "$MAX_VAL_ROWS" \
    "${PAIR_SOURCE_FILTER_ARGS[@]}" \
    --epochs "$FIXED_EPOCHS" \
    --lr "$FIXED_LR" \
    --lambda-norm "$FIXED_LAMBDA_NORM" \
    --alpha-train "$FIXED_ALPHA_TRAIN" \
    --state-margin-weight "$FIXED_STATE_MARGIN_WEIGHT" \
    --gain-weight "$FIXED_GAIN_WEIGHT" \
    --target-margin "$FIXED_TARGET_MARGIN" \
    --target-gain "$FIXED_TARGET_GAIN" \
    --apply-mode all \
    --causal-train-mask \
    --score-mode answer_rest_margin \
    --option-selection-mode model_max \
    --max-aliases-per-side "$MAX_ALIASES_PER_SIDE" \
    "${QUESTION_STATE_WEIGHT_ARGS[@]}" \
    --alpha-sweep "0,${ATT_ALPHA_GRID}" \
    --empty-cache-every "$FIXED_EMPTY_CACHE_EVERY" \
    --torch-dtype bfloat16 \
    --device cuda \
    --out-dir "$FIXED_HEAD_BOOST_DIR"
  RETRAINED_FIXED=1
fi

export MLP_ALPHA_GRID ATT_ALPHA_GRID
FIXED_CONTROLS="$("$PYTHON_BIN" - <<'PY'
import os

def values(key: str) -> list[str]:
    return [item.strip() for item in os.environ[key].split(",") if item.strip()]

def alpha_name(value: str) -> str:
    return str(value).replace(".", "p")

controls = ["base="]
for alpha in values("MLP_ALPHA_GRID"):
    controls.append(
        f"mlp_a{alpha_name(alpha)}="
        f"comp:mlp_prefill:{alpha}:prefill+comp:mlp_first_decode:{alpha}:first_decode"
    )
for alpha in values("ATT_ALPHA_GRID"):
    controls.append(
        f"att_a{alpha_name(alpha)}="
        f"head_act:head_boost:{alpha}:all+head_act:head_suppress:{alpha}:all"
    )
print(";".join(controls))
PY
)"

FIXED_SWEEP_SUMMARY="$FIXED_SWEEP_DIR/generation_summary.csv"
can_reuse_fixed_sweep() {
  local summary="$FIXED_SWEEP_DIR/generation_summary.csv"
  local run_cfg="$FIXED_SWEEP_DIR/run_config.json"
  [[ -f "$summary" && -f "$run_cfg" ]] || return 1
  REUSE_SWEEP_SUMMARY="$summary" REUSE_SWEEP_RUN_CFG="$run_cfg" "$PYTHON_BIN" - <<'PY'
import csv
import json
import os
from pathlib import Path

rows = list(csv.DictReader(Path(os.environ["REUSE_SWEEP_SUMMARY"]).open("r", encoding="utf-8", newline="")))
names = {str(row.get("control_name", "")) for row in rows}
cfg = json.loads(Path(os.environ["REUSE_SWEEP_RUN_CFG"]).read_text(encoding="utf-8"))
ok = (
    "base" in names
    and any(name.startswith("mlp_a") for name in names)
    and any(name.startswith("att_a") for name in names)
    and bool(cfg.get("force_stepwise", False))
)
raise SystemExit(0 if ok else 1)
PY
}

CAN_REUSE_SWEEP=0
if [[ "$RETRAINED_FIXED" != "1" && "$REUSE_SWEEP_IF_PRESENT" == "1" ]] && can_reuse_fixed_sweep; then
  CAN_REUSE_SWEEP=1
fi

if [[ "$CAN_REUSE_SWEEP" != "1" ]]; then
  FIXED_SWEEP_ROWS_PATH="$FIXED_SWEEP_DIR/generation_rows.jsonl"
  FIXED_SWEEP_OVERWRITE="$OVERWRITE_FIXED_SWEEP"
  if [[ "$RETRAINED_FIXED" == "1" && -f "$FIXED_SWEEP_ROWS_PATH" && "$FIXED_SWEEP_OVERWRITE" != "1" ]]; then
    FIXED_SWEEP_OVERWRITE=1
  fi
  sweep_cmd=(
    "$PYTHON_BIN" -m screscomp.cli.cecm_run_joint_actuator_generation
    --model "$MODEL"
    --eval-open-rows "$EVAL_OPEN_ROWS"
    --component-actuators "mlp_prefill=$FIXED_MLP_PREFILL_DIR/fixed_actuator.pt;mlp_first_decode=$FIXED_MLP_FIRST_DIR/fixed_actuator.pt"
    --head-actuators "head_suppress=$FIXED_HEAD_SUPPRESS_DIR/head_actuator.pt;head_boost=$FIXED_HEAD_BOOST_DIR/head_actuator.pt"
    --controls "$FIXED_CONTROLS"
    --generation-prompt-key "$GENERATION_PROMPT_KEY"
    --prior-source "$PRIOR_SOURCE"
    --scoring-kind "$SCORING_KIND"
    --split all
    --start 0
    --max-rows 500
    --generation-apply-mode "$GENERATION_APPLY_MODE"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --stop-strings "$STOP_STRINGS"
    --flush-every "$FLUSH_EVERY"
    --summary-every 50
    --empty-cache-every "$FIXED_EMPTY_CACHE_EVERY"
    --top-p "$TOP_P"
    --top-k "$TOP_K"
    --force-stepwise
    --torch-dtype bfloat16
    --device cuda
    --out-dir "$FIXED_SWEEP_DIR"
  )
  if [[ "$FIXED_SWEEP_OVERWRITE" == "1" ]]; then
    sweep_cmd+=(--overwrite)
  fi
  "${sweep_cmd[@]}"
else
  echo "[suite] reuse fixed sweep: $FIXED_SWEEP_SUMMARY"
fi

: "${ALPHA_MLP:=}"
: "${ALPHA_ATT:=}"

if [[ -z "$ALPHA_MLP" || -z "$ALPHA_ATT" ]]; then
  export FIXED_SWEEP_SUMMARY ALPHA_SELECT_TIE_TOL ALPHA_FLIP_REGRESSION_WEIGHT
  readarray -t ALPHA_LINES < <(
    "$PYTHON_BIN" - <<'PY'
import csv
import json
import os
import re
import shlex
from pathlib import Path

summary_path = Path(os.environ["FIXED_SWEEP_SUMMARY"])
rows_path = summary_path.with_name("generation_rows.jsonl")
rows = list(csv.DictReader(summary_path.open("r", encoding="utf-8", newline="")))
base_row = next((row for row in rows if row.get("control_name") == "base"), None)
if base_row is None:
    raise SystemExit("missing base row in fixed sweep summary")
base_strict = float(base_row.get("strict_final_exact", 0.0) or 0.0)
tie_tol = float(os.environ["ALPHA_SELECT_TIE_TOL"])
regression_weight = float(os.environ["ALPHA_FLIP_REGRESSION_WEIGHT"])

per_control = {}
if rows_path.exists():
    for line in rows_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        per_control.setdefault(str(row.get("control_name", "")), {})[str(row.get("sample_id", ""))] = row
base_items = per_control.get("base", {})

def parse_alpha(name: str) -> float:
    match = re.search(r"_a([0-9]+p[0-9]+)$", name)
    if not match:
        raise ValueError(f"cannot parse alpha from {name!r}")
    return float(match.group(1).replace("p", "."))

def choose(prefix: str) -> float:
    group_rows = []
    for row in rows:
      name = str(row.get("control_name", ""))
      if not name.startswith(prefix + "_a"):
        continue
      strict = float(row.get("strict_final_exact", 0.0) or 0.0)
      alpha = parse_alpha(name)
      control_items = per_control.get(name, {})
      wrong_to_right = 0
      right_to_wrong = 0
      for sample_id, base_item in base_items.items():
          control_item = control_items.get(sample_id)
          if control_item is None:
              continue
          base_ok = bool(base_item.get("strict_final_exact", False))
          control_ok = bool(control_item.get("strict_final_exact", False))
          if not base_ok and control_ok:
              wrong_to_right += 1
          elif base_ok and not control_ok:
              right_to_wrong += 1
      flip_score = wrong_to_right - regression_weight * right_to_wrong
      group_rows.append((flip_score, strict, alpha, name, wrong_to_right, right_to_wrong))
    if not group_rows:
      raise ValueError(f"no rows for {prefix}")
    positive = [item for item in group_rows if item[1] >= base_strict]
    pool = positive if positive else group_rows
    best_flip = max(item[0] for item in pool)
    candidates = [item for item in pool if item[0] == best_flip]
    best_strict = max(item[1] for item in candidates)
    candidates = [item for item in candidates if item[1] >= best_strict - tie_tol]
    chosen = min(candidates, key=lambda item: (item[2], -item[1], item[3]))
    return chosen[2]

selected = {
    "ALPHA_MLP": choose("mlp"),
    "ALPHA_ATT": choose("att"),
}
for key, value in selected.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
  )
  for line in "${ALPHA_LINES[@]}"; do
    eval "$line"
  done
fi

cat > "$ALPHA_ENV_PATH" <<EOF
ALPHA_MLP=$ALPHA_MLP
ALPHA_ATT=$ALPHA_ATT
EOF
echo "[suite] chosen alphas saved to $ALPHA_ENV_PATH"

can_reuse_gate() {
  local out_dir="$1"
  local expected_component_specs="$2"
  local expected_head_specs="$3"
  local expected_component_modes="$4"
  local expected_head_modes="$5"
  local expected_init_alphas="$6"
  local payload="$out_dir/conditional_actuator.pt"
  local run_cfg="$out_dir/run_config.json"
  [[ -f "$payload" && -f "$run_cfg" ]] || return 1
  REUSE_RUN_CFG="$run_cfg" REUSE_PAYLOAD="$payload" EXPECTED_COMPONENT_SPECS="$expected_component_specs" EXPECTED_HEAD_SPECS="$expected_head_specs" EXPECTED_COMPONENT_MODES="$expected_component_modes" EXPECTED_HEAD_MODES="$expected_head_modes" EXPECTED_INIT_ALPHAS="$expected_init_alphas" EXPECTED_MAX_SOURCE_ROW_INDEX="$MAX_SOURCE_ROW_INDEX" EXPECTED_QUESTION_STATE_WEIGHTS="$QUESTION_STATE_WEIGHTS" EXPECTED_INIT_DIRECTION_THRESHOLDS="$INIT_DIRECTION_THRESHOLDS" EXPECTED_INIT_DIRECTION_UPPER_THRESHOLDS="$INIT_DIRECTION_UPPER_THRESHOLDS" EXPECTED_INIT_PROJECTION_THRESHOLDS="$INIT_PROJECTION_THRESHOLDS" EXPECTED_INIT_PROJECTION_UPPER_THRESHOLDS="$INIT_PROJECTION_UPPER_THRESHOLDS" EXPECTED_PROJECTION_TEMPERATURE="$PROJECTION_TEMPERATURE" \
    "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path
import torch

def parse_map(raw: str) -> dict[str, str]:
    out = {}
    for part in str(raw or "").split(";"):
        part = part.strip()
        if not part:
            continue
        name, value = part.split("=", 1)
        out[name.strip()] = value.strip()
    return out

def parse_float_map(raw: str) -> dict[str, float]:
    return {str(key): float(value) for key, value in parse_map(raw).items()}

cfg = json.loads(Path(os.environ["REUSE_RUN_CFG"]).read_text(encoding="utf-8"))
payload = torch.load(Path(os.environ["REUSE_PAYLOAD"]), map_location="cpu")
expected_alphas = {key: float(value) for key, value in parse_map(os.environ["EXPECTED_INIT_ALPHAS"]).items()}
actual_alphas = {str(key): float(value) for key, value in dict(cfg.get("init_alphas", {}) or {}).items()}
actual_weights = {str(key): float(value) for key, value in dict(cfg.get("question_state_weights", {}) or {}).items()}
actual_init_direction_thresholds = {
    str(key): float(value)
    for key, value in dict(cfg.get("init_direction_thresholds", {}) or {}).items()
}
actual_init_direction_upper_thresholds = {
    str(key): float(value)
    for key, value in dict(cfg.get("init_direction_upper_thresholds", {}) or {}).items()
}
actual_init_projection_thresholds = {
    str(key): float(value)
    for key, value in dict(cfg.get("init_projection_thresholds", {}) or {}).items()
}
actual_init_projection_upper_thresholds = {
    str(key): float(value)
    for key, value in dict(cfg.get("init_projection_upper_thresholds", {}) or {}).items()
}
ok = (
    int(payload.get("payload_format_version", 0) or 0) >= 7
    and bool(cfg.get("freeze_alpha_max", False))
    and str(cfg.get("save_gate_mode", "")).lower() == "hard"
    and int(cfg.get("max_source_row_index")) == int(os.environ["EXPECTED_MAX_SOURCE_ROW_INDEX"])
    and actual_alphas == expected_alphas
    and actual_init_direction_thresholds == parse_float_map(os.environ["EXPECTED_INIT_DIRECTION_THRESHOLDS"])
    and actual_init_direction_upper_thresholds == parse_float_map(os.environ["EXPECTED_INIT_DIRECTION_UPPER_THRESHOLDS"])
    and actual_init_projection_thresholds == parse_float_map(os.environ["EXPECTED_INIT_PROJECTION_THRESHOLDS"])
    and actual_init_projection_upper_thresholds == parse_float_map(os.environ["EXPECTED_INIT_PROJECTION_UPPER_THRESHOLDS"])
    and float(cfg.get("projection_temperature", 1.0)) == float(os.environ["EXPECTED_PROJECTION_TEMPERATURE"])
    and dict(cfg.get("component_actuators", {}) or {}) == parse_map(os.environ["EXPECTED_COMPONENT_SPECS"])
    and dict(cfg.get("head_actuators", {}) or {}) == parse_map(os.environ["EXPECTED_HEAD_SPECS"])
    and dict(cfg.get("component_train_apply_modes", {}) or {}) == parse_map(os.environ["EXPECTED_COMPONENT_MODES"])
    and dict(cfg.get("head_train_apply_modes", {}) or {}) == parse_map(os.environ["EXPECTED_HEAD_MODES"])
    and actual_weights == parse_float_map(os.environ["EXPECTED_QUESTION_STATE_WEIGHTS"])
)
raise SystemExit(0 if ok else 1)
PY
}

RETRAINED_GATES=0
if [[ "$RETRAINED_FIXED" == "1" ]]; then
  REUSE_GATE_IF_PRESENT=0
fi

train_conditional_group() {
  local label="$1"
  local out_dir="$2"
  local component_specs="$3"
  local head_specs="$4"
  local component_modes="$5"
  local head_modes="$6"
  local init_alphas="$7"

  if [[ "$REUSE_GATE_IF_PRESENT" == "1" ]] && can_reuse_gate "$out_dir" "$component_specs" "$head_specs" "$component_modes" "$head_modes" "$init_alphas"; then
    echo "[suite] reuse conditional $label: $out_dir/conditional_actuator.pt"
    return 0
  fi

  "$PYTHON_BIN" -m screscomp.cli.cecm_train_conditional_actuator \
    --model "$MODEL" \
    --pairs-csv "$PAIRS_CSV" \
    --event "$EVENT" \
    --train-split train \
    --val-split val \
    --max-train-rows "$MAX_TRAIN_ROWS" \
    --max-val-rows "$MAX_VAL_ROWS" \
    "${PAIR_SOURCE_FILTER_ARGS[@]}" \
    --epochs "$GATE_EPOCHS" \
    --lr "$GATE_LR" \
    --lambda-alpha "$GATE_LAMBDA_ALPHA" \
    --state-margin-weight "$GATE_STATE_MARGIN_WEIGHT" \
    --gain-weight "$GATE_GAIN_WEIGHT" \
    --target-margin "$GATE_TARGET_MARGIN" \
    --target-gain "$GATE_TARGET_GAIN" \
    --score-mode answer_rest_margin \
    "${QUESTION_STATE_WEIGHT_ARGS[@]}" \
    --component-actuators "$component_specs" \
    --head-actuators "$head_specs" \
    --component-train-apply-modes "$component_modes" \
    --head-train-apply-modes "$head_modes" \
    --component-alpha-max "$COMPONENT_ALPHA_MAX" \
    --head-alpha-max "$HEAD_ALPHA_MAX" \
    --init-alphas "$init_alphas" \
    "${INIT_DIRECTION_THRESHOLD_ARGS[@]}" \
    --direction-temperature "$SIMILARITY_TEMPERATURE" \
    --projection-temperature "$PROJECTION_TEMPERATURE" \
    --save-gate-mode "$SAVE_GATE_MODE" \
    --empty-cache-every "$GATE_EMPTY_CACHE_EVERY" \
    --freeze-alpha-max \
    --torch-dtype bfloat16 \
    --device cuda \
    --out-dir "$out_dir"
  RETRAINED_GATES=1
}

MLP_COMPONENT_SPECS="mlp_prefill=$FIXED_MLP_PREFILL_DIR/fixed_actuator.pt;mlp_first_decode=$FIXED_MLP_FIRST_DIR/fixed_actuator.pt"
MLP_COMPONENT_MODES="mlp_prefill=prompt_last;mlp_first_decode=first_decode"
ATT_HEAD_SPECS="head_suppress=$FIXED_HEAD_SUPPRESS_DIR/head_actuator.pt;head_boost=$FIXED_HEAD_BOOST_DIR/head_actuator.pt"
ATT_HEAD_MODES="head_suppress=all;head_boost=all"

train_conditional_group \
  "mlp" \
  "$COND_MLP_DIR" \
  "$MLP_COMPONENT_SPECS" \
  "" \
  "$MLP_COMPONENT_MODES" \
  "" \
  "mlp_prefill=$ALPHA_MLP;mlp_first_decode=$ALPHA_MLP"

train_conditional_group \
  "att" \
  "$COND_ATT_DIR" \
  "" \
  "$ATT_HEAD_SPECS" \
  "" \
  "$ATT_HEAD_MODES" \
  "head_suppress=$ALPHA_ATT;head_boost=$ALPHA_ATT"

train_conditional_group \
  "full" \
  "$COND_FULL_DIR" \
  "$MLP_COMPONENT_SPECS" \
  "$ATT_HEAD_SPECS" \
  "$MLP_COMPONENT_MODES" \
  "$ATT_HEAD_MODES" \
  "mlp_prefill=$ALPHA_MLP;mlp_first_decode=$ALPHA_MLP;head_suppress=$ALPHA_ATT;head_boost=$ALPHA_ATT"

FINAL_ROWS_PATH="$FINAL_OUT_DIR/generation_rows.jsonl"
FINAL_EVAL_OVERWRITE="$OVERWRITE_FINAL"
if [[ "$RETRAINED_GATES" == "1" && -f "$FINAL_ROWS_PATH" && "$FINAL_EVAL_OVERWRITE" != "1" ]]; then
  FINAL_EVAL_OVERWRITE=1
fi

if [[ "$RUN_FINAL_EVAL" != "1" ]]; then
  echo "[suite] skipping final eval because RUN_FINAL_EVAL=$RUN_FINAL_EVAL"
  echo "[suite] gate payloads:"
  echo "[suite]   mlp=$COND_MLP_DIR/conditional_actuator.pt"
  echo "[suite]   att=$COND_ATT_DIR/conditional_actuator.pt"
  echo "[suite]   full=$COND_FULL_DIR/conditional_actuator.pt"
  exit 0
fi

FINAL_CONTROLS="cast_att=cond:att;cast_mlp=cond:mlp;cast_full=cond:full"
if [[ "$FINAL_INCLUDE_BASE" == "1" ]]; then
  FINAL_CONTROLS="base=;$FINAL_CONTROLS"
fi

eval_cmd=(
  "$PYTHON_BIN" -m screscomp.cli.cecm_run_joint_actuator_generation
  --model "$MODEL"
  --eval-open-rows "$EVAL_OPEN_ROWS"
  --conditional-actuators "mlp=$COND_MLP_DIR/conditional_actuator.pt;att=$COND_ATT_DIR/conditional_actuator.pt;full=$COND_FULL_DIR/conditional_actuator.pt"
  --controls "$FINAL_CONTROLS"
  --generation-prompt-key "$GENERATION_PROMPT_KEY"
  --prior-source "$PRIOR_SOURCE"
  --scoring-kind "$SCORING_KIND"
  --split "$FINAL_SPLIT"
  --start "$FINAL_START"
  --max-rows "$FINAL_MAX_ROWS"
  --generation-apply-mode "$GENERATION_APPLY_MODE"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --stop-strings "$STOP_STRINGS"
  --flush-every "$FLUSH_EVERY"
  --summary-every "$SUMMARY_EVERY"
  --empty-cache-every "$GATE_EMPTY_CACHE_EVERY"
  --top-p "$TOP_P"
  --top-k "$TOP_K"
  --force-stepwise
  --torch-dtype bfloat16
  --device cuda
  --out-dir "$FINAL_OUT_DIR"
)

if [[ "$DO_SAMPLE" == "1" ]]; then
  eval_cmd+=(--do-sample --temperature "$TEMPERATURE")
fi
if [[ "$FINAL_EVAL_OVERWRITE" == "1" ]]; then
  eval_cmd+=(--overwrite)
fi

echo "[suite] final eval out=$FINAL_OUT_DIR"
echo "[suite] final controls=$FINAL_CONTROLS"
"${eval_cmd[@]}"

echo "[suite] done"
echo "[suite] fixed_sweep_summary=$FIXED_SWEEP_DIR/generation_summary.csv"
echo "[suite] final_summary=$FINAL_OUT_DIR/generation_summary.csv"
