#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ROUND_DIR="${ROUND_DIR:?set ROUND_DIR to a round directory such as data_gsm8k/rounds/.../round_00}"

SOURCE_EVAL_NAME="${SOURCE_EVAL_NAME:-gsm8k_val_500}"
SOURCE_EVAL_DIR="${SOURCE_EVAL_DIR:-$ROUND_DIR/eval/$SOURCE_EVAL_NAME}"
SOURCE_RUN_CONFIG="${SOURCE_RUN_CONFIG:-$SOURCE_EVAL_DIR/run_config.json}"

FIXED_MLP_PREFILL_DIR="${FIXED_MLP_PREFILL_DIR:-$ROUND_DIR/train/train_mlp_positive_prefill_causal}"
FIXED_MLP_FIRST_DIR="${FIXED_MLP_FIRST_DIR:-$ROUND_DIR/train/train_mlp_positive_first_decode_causal}"
FIXED_HEAD_SUPPRESS_DIR="${FIXED_HEAD_SUPPRESS_DIR:-$ROUND_DIR/train/train_head_suppress_causal}"
FIXED_HEAD_BOOST_DIR="${FIXED_HEAD_BOOST_DIR:-$ROUND_DIR/train/train_head_boost_causal}"

COND_MLP_NAME="${COND_MLP_NAME:-train_conditional_mlp_causal}"
COND_ATT_NAME="${COND_ATT_NAME:-train_conditional_att_causal}"
COND_FULL_NAME="${COND_FULL_NAME:-train_conditional_full_causal}"
COND_MLP_DIR="${COND_MLP_DIR:-$ROUND_DIR/conditional/$COND_MLP_NAME}"
COND_ATT_DIR="${COND_ATT_DIR:-$ROUND_DIR/conditional/$COND_ATT_NAME}"
COND_FULL_DIR="${COND_FULL_DIR:-$ROUND_DIR/conditional/$COND_FULL_NAME}"

ALPHA_ENV_PATH="${ALPHA_ENV_PATH:-$ROUND_DIR/conditional/causal_conditional_suite_alphas.env}"

SMOKE_OUT_NAME="${SMOKE_OUT_NAME:-gsm8k_matched_gate_smoke}"
SMOKE_OUT_DIR="${SMOKE_OUT_DIR:-$ROUND_DIR/eval/$SMOKE_OUT_NAME}"

SMOKE_SPLIT="${SMOKE_SPLIT:-all}"
SMOKE_START="${SMOKE_START:-0}"
SMOKE_MAX_ROWS="${SMOKE_MAX_ROWS:-200}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
STOP_STRINGS="${STOP_STRINGS:-$'\nUser:'}"
FLUSH_EVERY="${FLUSH_EVERY:-1}"
SUMMARY_EVERY="${SUMMARY_EVERY:-25}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:-100}"
DO_SAMPLE="${DO_SAMPLE:-0}"
TEMPERATURE="${TEMPERATURE:-1.0}"
SMOKE_OVERWRITE="${SMOKE_OVERWRITE:-0}"
SMOKE_RESET_CONTROLS="${SMOKE_RESET_CONTROLS:-}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

cd "$ROOT"
export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export SOURCE_RUN_CONFIG

for required in \
  "$SOURCE_RUN_CONFIG" \
  "$FIXED_MLP_PREFILL_DIR/fixed_actuator.pt" \
  "$FIXED_MLP_FIRST_DIR/fixed_actuator.pt" \
  "$FIXED_HEAD_SUPPRESS_DIR/head_actuator.pt" \
  "$FIXED_HEAD_BOOST_DIR/head_actuator.pt" \
  "$COND_MLP_DIR/conditional_actuator.pt" \
  "$COND_ATT_DIR/conditional_actuator.pt" \
  "$COND_FULL_DIR/conditional_actuator.pt" \
  "$ALPHA_ENV_PATH"
do
  if [[ ! -f "$required" ]]; then
    echo "[smoke] missing required file: $required" >&2
    exit 1
  fi
done

source "$ALPHA_ENV_PATH"
: "${ALPHA_MLP:?missing ALPHA_MLP in $ALPHA_ENV_PATH}"
: "${ALPHA_ATT:?missing ALPHA_ATT in $ALPHA_ENV_PATH}"

readarray -t CONFIG_LINES < <(
  "$PYTHON_BIN" - <<'PY'
import json
import os
import shlex
from pathlib import Path

cfg = json.loads(Path(os.environ["SOURCE_RUN_CONFIG"]).read_text(encoding="utf-8"))

def emit(name: str, value: str) -> None:
    print(f"{name}={shlex.quote(str(value))}")

emit("MODEL", cfg["model"])
emit("EVAL_OPEN_ROWS", cfg["eval_open_rows"])
emit("GENERATION_PROMPT_KEY", cfg.get("generation_prompt_key", "deepseek_math"))
emit("PRIOR_SOURCE", cfg.get("prior_source", "model_prior"))
emit("SCORING_KIND", cfg.get("scoring_kind", "gsm8k"))
emit("GENERATION_APPLY_MODE", cfg.get("generation_apply_mode", "prefill"))
PY
)

for line in "${CONFIG_LINES[@]}"; do
  eval "$line"
done

mkdir -p "$SMOKE_OUT_DIR"

if [[ -n "$SMOKE_RESET_CONTROLS" ]]; then
  export SMOKE_OUT_DIR SMOKE_RESET_CONTROLS
  "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

out_dir = Path(os.environ["SMOKE_OUT_DIR"])
rows_path = out_dir / "generation_rows.jsonl"
summary_path = out_dir / "generation_summary.csv"
compare_path = out_dir / "comparison_summary.csv"
reset_controls = {
    item.strip()
    for item in str(os.environ["SMOKE_RESET_CONTROLS"]).split(",")
    if item.strip()
}
if reset_controls and rows_path.exists():
    kept = []
    removed = 0
    for line in rows_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("control_name", "")) in reset_controls:
            removed += 1
            continue
        kept.append(line)
    rows_path.write_text(("".join(f"{line}\n" for line in kept)), encoding="utf-8")
    if summary_path.exists():
        summary_path.unlink()
    if compare_path.exists():
        compare_path.unlink()
    print(f"[smoke] reset controls={sorted(reset_controls)} removed_rows={removed}", flush=True)
PY
fi

CONTROLS="base=;\
fixed_att=head_act:head_boost:${ALPHA_ATT}:all+head_act:head_suppress:${ALPHA_ATT}:all;\
gate_att=cond:att;\
fixed_mlp=comp:mlp_prefill:${ALPHA_MLP}:prefill+comp:mlp_first_decode:${ALPHA_MLP}:first_decode;\
gate_mlp=cond:mlp;\
fixed_full=comp:mlp_prefill:${ALPHA_MLP}:prefill+comp:mlp_first_decode:${ALPHA_MLP}:first_decode+head_act:head_boost:${ALPHA_ATT}:all+head_act:head_suppress:${ALPHA_ATT}:all;\
gate_full=cond:full"

cmd=(
  "$PYTHON_BIN" -m screscomp.cli.cecm_run_joint_actuator_generation
  --model "$MODEL"
  --eval-open-rows "$EVAL_OPEN_ROWS"
  --component-actuators "mlp_prefill=$FIXED_MLP_PREFILL_DIR/fixed_actuator.pt;mlp_first_decode=$FIXED_MLP_FIRST_DIR/fixed_actuator.pt"
  --head-actuators "head_suppress=$FIXED_HEAD_SUPPRESS_DIR/head_actuator.pt;head_boost=$FIXED_HEAD_BOOST_DIR/head_actuator.pt"
  --conditional-actuators "mlp=$COND_MLP_DIR/conditional_actuator.pt;att=$COND_ATT_DIR/conditional_actuator.pt;full=$COND_FULL_DIR/conditional_actuator.pt"
  --controls "$CONTROLS"
  --generation-prompt-key "$GENERATION_PROMPT_KEY"
  --prior-source "$PRIOR_SOURCE"
  --scoring-kind "$SCORING_KIND"
  --split "$SMOKE_SPLIT"
  --start "$SMOKE_START"
  --max-rows "$SMOKE_MAX_ROWS"
  --generation-apply-mode "$GENERATION_APPLY_MODE"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --stop-strings "$STOP_STRINGS"
  --flush-every "$FLUSH_EVERY"
  --summary-every "$SUMMARY_EVERY"
  --top-p "$TOP_P"
  --top-k "$TOP_K"
  --force-stepwise
  --torch-dtype bfloat16
  --device cuda
  --out-dir "$SMOKE_OUT_DIR"
)

if [[ "$DO_SAMPLE" == "1" ]]; then
  cmd+=(--do-sample --temperature "$TEMPERATURE")
fi
if [[ "$SMOKE_OVERWRITE" == "1" ]]; then
  cmd+=(--overwrite)
fi

echo "[smoke] out=$SMOKE_OUT_DIR"
echo "[smoke] split=$SMOKE_SPLIT start=$SMOKE_START max_rows=$SMOKE_MAX_ROWS"
echo "[smoke] alpha_mlp=$ALPHA_MLP alpha_att=$ALPHA_ATT"
"${cmd[@]}"

SMOKE_ROWS_PATH="$SMOKE_OUT_DIR/generation_rows.jsonl"
SMOKE_COMPARE_PATH="$SMOKE_OUT_DIR/comparison_summary.csv"
export SMOKE_ROWS_PATH SMOKE_COMPARE_PATH

"$PYTHON_BIN" - <<'PY'
import csv
import json
import os
from pathlib import Path

rows_path = Path(os.environ["SMOKE_ROWS_PATH"])
out_path = Path(os.environ["SMOKE_COMPARE_PATH"])

per_control: dict[str, dict[str, dict[str, object]]] = {}
with rows_path.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        row = json.loads(line)
        per_control.setdefault(str(row["control_name"]), {})[str(row["sample_id"])] = row

comparisons = [
    ("base", "fixed_att"),
    ("base", "gate_att"),
    ("base", "fixed_mlp"),
    ("base", "gate_mlp"),
    ("base", "fixed_full"),
    ("base", "gate_full"),
    ("fixed_att", "gate_att"),
    ("fixed_mlp", "gate_mlp"),
    ("fixed_full", "gate_full"),
]

fieldnames = [
    "anchor",
    "candidate",
    "n_common",
    "anchor_acc",
    "candidate_acc",
    "delta",
    "wrong_to_right",
    "right_to_wrong",
    "right_to_right",
    "wrong_to_wrong",
    "net",
]

with out_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for anchor, candidate in comparisons:
        anchor_rows = per_control.get(anchor, {})
        candidate_rows = per_control.get(candidate, {})
        common = sorted(set(anchor_rows) & set(candidate_rows))
        n = len(common)
        if n == 0:
            writer.writerow(
                {
                    "anchor": anchor,
                    "candidate": candidate,
                    "n_common": 0,
                    "anchor_acc": "",
                    "candidate_acc": "",
                    "delta": "",
                    "wrong_to_right": "",
                    "right_to_wrong": "",
                    "right_to_right": "",
                    "wrong_to_wrong": "",
                    "net": "",
                }
            )
            continue
        wr = rw = rr = ww = anchor_ok = candidate_ok = 0
        for sample_id in common:
            a = bool(anchor_rows[sample_id]["strict_final_exact"])
            b = bool(candidate_rows[sample_id]["strict_final_exact"])
            anchor_ok += int(a)
            candidate_ok += int(b)
            if (not a) and b:
                wr += 1
            elif a and (not b):
                rw += 1
            elif a and b:
                rr += 1
            else:
                ww += 1
        writer.writerow(
            {
                "anchor": anchor,
                "candidate": candidate,
                "n_common": n,
                "anchor_acc": anchor_ok / n,
                "candidate_acc": candidate_ok / n,
                "delta": (candidate_ok - anchor_ok) / n,
                "wrong_to_right": wr,
                "right_to_wrong": rw,
                "right_to_right": rr,
                "wrong_to_wrong": ww,
                "net": wr - rw,
            }
        )

print(f"[smoke] comparison summary -> {out_path}", flush=True)
PY

echo "[smoke] done"
echo "[smoke] summary=$SMOKE_OUT_DIR/generation_summary.csv"
echo "[smoke] comparisons=$SMOKE_COMPARE_PATH"
