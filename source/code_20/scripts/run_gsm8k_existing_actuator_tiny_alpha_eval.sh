#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ROUND_DIR="${ROUND_DIR:?set ROUND_DIR to a round directory such as data_gsm8k/rounds/.../round_00}"
SOURCE_EVAL_NAME="${SOURCE_EVAL_NAME:-gsm8k_val_500}"
SOURCE_EVAL_DIR="${SOURCE_EVAL_DIR:-$ROUND_DIR/eval/$SOURCE_EVAL_NAME}"
SOURCE_RUN_CONFIG="${SOURCE_RUN_CONFIG:-$SOURCE_EVAL_DIR/run_config.json}"
OUT_NAME="${OUT_NAME:-${SOURCE_EVAL_NAME}_tiny_alpha}"
OUT_DIR="${OUT_DIR:-$ROUND_DIR/eval/$OUT_NAME}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
SPLIT="${SPLIT:-all}"
START="${START:-0}"
MAX_ROWS="${MAX_ROWS:-500}"
GENERATION_APPLY_MODE="${GENERATION_APPLY_MODE:-prefill}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
STOP_STRINGS="${STOP_STRINGS:-$'\nUser:'}"
FLUSH_EVERY="${FLUSH_EVERY:-1}"
SUMMARY_EVERY="${SUMMARY_EVERY:-50}"
EMPTY_CACHE_EVERY="${EMPTY_CACHE_EVERY:-25}"
DO_SAMPLE="${DO_SAMPLE:-0}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:-100}"
OVERWRITE="${OVERWRITE:-0}"

INCLUDE_BASE="${INCLUDE_BASE:-0}"
BASE_SOURCE_CONTROL_NAME="${BASE_SOURCE_CONTROL_NAME:-base}"
INCLUDE_MLP_PREFILL="${INCLUDE_MLP_PREFILL:-1}"
INCLUDE_MLP_FIRST_DECODE="${INCLUDE_MLP_FIRST_DECODE:-0}"
INCLUDE_ATT="${INCLUDE_ATT:-1}"
INCLUDE_FULL_PREFILL="${INCLUDE_FULL_PREFILL:-0}"
INCLUDE_FULL_FIRST_DECODE="${INCLUDE_FULL_FIRST_DECODE:-0}"

MLP_PREFILL_ALPHAS="${MLP_PREFILL_ALPHAS:-0.0025,0.005,0.0075,0.01,0.0125,0.015,0.02}"
MLP_FIRST_DECODE_ALPHAS="${MLP_FIRST_DECODE_ALPHAS:-0.0025,0.005,0.0075,0.01,0.0125,0.015,0.02}"
ATT_ALPHAS="${ATT_ALPHAS:-0.0025,0.005,0.0075,0.01,0.015,0.02,0.04,0.0625}"
FULL_PREFILL_PAIRS="${FULL_PREFILL_PAIRS:-0.0025:0.0025,0.005:0.005,0.0075:0.01,0.01:0.015,0.015:0.02,0.02:0.04}"
FULL_FIRST_DECODE_PAIRS="${FULL_FIRST_DECODE_PAIRS:-0.0025:0.0025,0.005:0.005,0.0075:0.01,0.01:0.015,0.015:0.02,0.02:0.04}"

cd "$ROOT"
export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export SOURCE_RUN_CONFIG OUT_DIR
export INCLUDE_BASE BASE_SOURCE_CONTROL_NAME
export INCLUDE_MLP_PREFILL INCLUDE_MLP_FIRST_DECODE INCLUDE_ATT INCLUDE_FULL_PREFILL INCLUDE_FULL_FIRST_DECODE
export MLP_PREFILL_ALPHAS MLP_FIRST_DECODE_ALPHAS ATT_ALPHAS FULL_PREFILL_PAIRS FULL_FIRST_DECODE_PAIRS

if [[ ! -f "$SOURCE_RUN_CONFIG" ]]; then
  echo "[tiny-alpha] missing source run_config: $SOURCE_RUN_CONFIG" >&2
  exit 1
fi

readarray -t CONFIG_LINES < <(
  "$PYTHON_BIN" - <<'PY'
import json
import os
import shlex
from pathlib import Path

source_run_config = Path(os.environ["SOURCE_RUN_CONFIG"])
cfg = json.loads(source_run_config.read_text(encoding="utf-8"))

component_actuators = {str(k): str(v) for k, v in dict(cfg.get("component_actuators", {})).items()}
head_actuators = {str(k): str(v) for k, v in dict(cfg.get("head_actuators", {})).items()}
controls = dict(cfg.get("controls", {}))
background_parts = list(controls.get("prev", []))

def parse_float_list(raw: str):
    return [float(item.strip()) for item in raw.split(",") if item.strip()]

def parse_pairs(raw: str):
    pairs = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        left, right = item.split(":", 1)
        pairs.append((float(left), float(right)))
    return pairs

def truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "1" if default else "0").strip().lower()
    return raw not in {"", "0", "false", "no", "off"}

def alpha_name(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")

def join_parts(extra_parts):
    parts = [*background_parts, *extra_parts]
    return "+".join(parts)

prefill_names = sorted(name for name in component_actuators if name.endswith("__prefill"))
first_decode_names = sorted(name for name in component_actuators if name.endswith("__first_decode"))
head_names = sorted(head_actuators)

tiny_controls: dict[str, str] = {}
if truthy("INCLUDE_BASE"):
    tiny_controls["base"] = "+".join(background_parts)

if truthy("INCLUDE_MLP_PREFILL"):
    for alpha in parse_float_list(os.environ["MLP_PREFILL_ALPHAS"]):
        parts = [f"comp:{name}:{alpha}:prefill" for name in prefill_names]
        if parts:
            tiny_controls[f"cast_mlp_prefill_a{alpha_name(alpha)}"] = join_parts(parts)

if truthy("INCLUDE_MLP_FIRST_DECODE"):
    for alpha in parse_float_list(os.environ["MLP_FIRST_DECODE_ALPHAS"]):
        parts = [f"comp:{name}:{alpha}:first_decode" for name in first_decode_names]
        if parts:
            tiny_controls[f"cast_mlp_first_decode_a{alpha_name(alpha)}"] = join_parts(parts)

if truthy("INCLUDE_ATT"):
    for alpha in parse_float_list(os.environ["ATT_ALPHAS"]):
        parts = [f"head_act:{name}:{alpha}:all" for name in head_names]
        if parts:
            tiny_controls[f"cast_att_a{alpha_name(alpha)}"] = join_parts(parts)

if truthy("INCLUDE_FULL_PREFILL"):
    for mlp_alpha, head_alpha in parse_pairs(os.environ["FULL_PREFILL_PAIRS"]):
        parts = [f"comp:{name}:{mlp_alpha}:prefill" for name in prefill_names]
        parts.extend(f"head_act:{name}:{head_alpha}:all" for name in head_names)
        if parts:
            tiny_controls[f"cast_full_prefill_m{alpha_name(mlp_alpha)}_h{alpha_name(head_alpha)}"] = join_parts(parts)

if truthy("INCLUDE_FULL_FIRST_DECODE"):
    for mlp_alpha, head_alpha in parse_pairs(os.environ["FULL_FIRST_DECODE_PAIRS"]):
        parts = [f"comp:{name}:{mlp_alpha}:first_decode" for name in first_decode_names]
        parts.extend(f"head_act:{name}:{head_alpha}:all" for name in head_names)
        if parts:
            tiny_controls[f"cast_full_first_decode_m{alpha_name(mlp_alpha)}_h{alpha_name(head_alpha)}"] = join_parts(parts)

controls_text = ";".join(f"{name}={parts}" for name, parts in tiny_controls.items()) + ";"
component_text = ";".join(f"{name}={path}" for name, path in sorted(component_actuators.items()))
head_text = ";".join(f"{name}={path}" for name, path in sorted(head_actuators.items()))

def emit(name: str, value: str) -> None:
    print(f"{name}={shlex.quote(str(value))}")

emit("MODEL", cfg["model"])
emit("EVAL_OPEN_ROWS", cfg["eval_open_rows"])
emit("GENERATION_PROMPT_KEY", cfg.get("generation_prompt_key", "deepseek_math"))
emit("PRIOR_SOURCE", cfg.get("prior_source", "model_prior"))
emit("SCORING_KIND", cfg.get("scoring_kind", "gsm8k"))
emit("COMPONENT_ACTUATORS", component_text)
emit("HEAD_ACTUATORS", head_text)
emit("CONTROLS", controls_text)
PY
)

for line in "${CONFIG_LINES[@]}"; do
  eval "$line"
done

mkdir -p "$OUT_DIR"

cmd=(
  "$PYTHON_BIN" -m screscomp.cli.cecm_run_joint_actuator_generation
  --model "$MODEL"
  --eval-open-rows "$EVAL_OPEN_ROWS"
  --component-actuators "$COMPONENT_ACTUATORS"
  --head-actuators "$HEAD_ACTUATORS"
  --controls "$CONTROLS"
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
  --out-dir "$OUT_DIR"
)

if [[ "$DO_SAMPLE" == "1" ]]; then
  cmd+=(--do-sample --temperature "$TEMPERATURE")
fi
if [[ "$OVERWRITE" == "1" ]]; then
  cmd+=(--overwrite)
fi

echo "[tiny-alpha] source=$SOURCE_RUN_CONFIG out=$OUT_DIR"
echo "[tiny-alpha] controls=$CONTROLS"
"${cmd[@]}"

SOURCE_SUMMARY="$SOURCE_EVAL_DIR/generation_summary.csv"
TARGET_SUMMARY="$OUT_DIR/generation_summary.csv"
COMPARE_SUMMARY="$OUT_DIR/comparison_generation_summary.csv"
export SOURCE_SUMMARY TARGET_SUMMARY COMPARE_SUMMARY
if [[ -f "$SOURCE_SUMMARY" && -f "$TARGET_SUMMARY" && "$INCLUDE_BASE" != "1" ]]; then
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
fieldnames = list(base_rows[0].keys())
rows = []
rows.extend(base_rows)
rows.extend(row for row in target_rows if row.get("control_name") not in {base_name})

with compare_summary.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f"[tiny-alpha] wrote comparison summary: {compare_summary}")
PY
fi
