#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_large_v1_model_loop.sh --model PATH_OR_HF_ID --out-dir PATH [options]

Options:
  --rendered-jsonl PATH       Input rendered samples. Default: data_large_v1/02_prompts/09_rendered_samples.jsonl
  --layers SPEC               Layer selection for component scans. Default: all
  --components CSV            Component types. Default: attn,mlp
  --top-k-components CSV      Steering prefix sizes. Default: 1,2,4,8
  --alphas CSV                Steering alpha grid. Default: 0.25,0.5,1.0,2.0
  --device DEVICE             cpu|cuda|auto. Default: auto
  --torch-dtype DTYPE         auto|float16|bfloat16|float32. Default: auto
  --use-chat-template         Apply tokenizer chat template before scoring.
  --max-behavior-rows N       Optional behavior row limit.
  --max-component-rows N      Optional component row limit.
  --max-train-rows N          Optional steering train row limit.
  --max-validation-rows N     Optional steering validation row limit.
  --max-eval-rows N           Optional steering eval row limit.
  --transformers-src PATH     Optional local transformers checkout, usually vendor/transformers/src.
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL=""
OUT_DIR=""
RENDERED_JSONL="data_large_v1/02_prompts/09_rendered_samples.jsonl"
LAYERS="all"
COMPONENTS="attn,mlp"
TOP_K_COMPONENTS="1,2,4,8"
ALPHAS="0.25,0.5,1.0,2.0"
DEVICE="auto"
TORCH_DTYPE="auto"
USE_CHAT_TEMPLATE=0
MAX_BEHAVIOR_ROWS=""
MAX_COMPONENT_ROWS=""
MAX_TRAIN_ROWS=""
MAX_VALIDATION_ROWS=""
MAX_EVAL_ROWS=""
TRANSFORMERS_SRC=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --rendered-jsonl) RENDERED_JSONL="$2"; shift 2 ;;
    --layers) LAYERS="$2"; shift 2 ;;
    --components) COMPONENTS="$2"; shift 2 ;;
    --top-k-components) TOP_K_COMPONENTS="$2"; shift 2 ;;
    --alphas) ALPHAS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --torch-dtype) TORCH_DTYPE="$2"; shift 2 ;;
    --use-chat-template) USE_CHAT_TEMPLATE=1; shift ;;
    --max-behavior-rows) MAX_BEHAVIOR_ROWS="$2"; shift 2 ;;
    --max-component-rows) MAX_COMPONENT_ROWS="$2"; shift 2 ;;
    --max-train-rows) MAX_TRAIN_ROWS="$2"; shift 2 ;;
    --max-validation-rows) MAX_VALIDATION_ROWS="$2"; shift 2 ;;
    --max-eval-rows) MAX_EVAL_ROWS="$2"; shift 2 ;;
    --transformers-src) TRANSFORMERS_SRC="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$MODEL" || -z "$OUT_DIR" ]]; then
  usage >&2
  exit 2
fi

if [[ -z "${CONDA_PREFIX:-}" && -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

PYTHONPATH_PARTS=("src")
if [[ -z "$TRANSFORMERS_SRC" && -d "vendor/transformers/src" ]]; then
  TRANSFORMERS_SRC="vendor/transformers/src"
fi
if [[ -n "$TRANSFORMERS_SRC" ]]; then
  PYTHONPATH_PARTS=("$TRANSFORMERS_SRC" "${PYTHONPATH_PARTS[@]}")
fi
export PYTHONPATH="$(IFS=:; echo "${PYTHONPATH_PARTS[*]}")${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUT_DIR"

COMMON_ARGS=(
  --rendered_jsonl "$RENDERED_JSONL"
  --model "$MODEL"
  --device "$DEVICE"
  --torch_dtype "$TORCH_DTYPE"
)
if [[ "$USE_CHAT_TEMPLATE" -eq 1 ]]; then
  COMMON_ARGS+=(--use_chat_template)
fi

echo "[large-v1-model-loop] behavior"
BEHAVIOR_ARGS=(
  -m screscomp.cli.score_behavior
  "${COMMON_ARGS[@]}"
  --out_scores_jsonl "$OUT_DIR/behavior_scores.jsonl"
  --out_summary_csv "$OUT_DIR/behavior_summary.csv"
)
if [[ -n "$MAX_BEHAVIOR_ROWS" ]]; then
  BEHAVIOR_ARGS+=(--max_rows "$MAX_BEHAVIOR_ROWS")
fi
python "${BEHAVIOR_ARGS[@]}"

echo "[large-v1-model-loop] component E1"
COMP_E1_ARGS=(
  -m screscomp.cli.score_component_competition
  "${COMMON_ARGS[@]}"
  --out_rows_jsonl "$OUT_DIR/component_competition_e1.jsonl"
  --out_summary_csv "$OUT_DIR/component_competition_e1_summary.csv"
  --layers "$LAYERS"
  --components "$COMPONENTS"
  --pairs "e1_content,e1_content_swapped"
  --splits "discovery,validation,test"
)
if [[ -n "$MAX_COMPONENT_ROWS" ]]; then
  COMP_E1_ARGS+=(--max_rows "$MAX_COMPONENT_ROWS")
fi
python "${COMP_E1_ARGS[@]}"

echo "[large-v1-model-loop] steering"
STEER_ARGS=(
  -m screscomp.cli.score_steering_utility
  "${COMMON_ARGS[@]}"
  --component_summary_csv "$OUT_DIR/component_competition_e1_summary.csv"
  --out_rows_jsonl "$OUT_DIR/steering_utility.jsonl"
  --out_summary_csv "$OUT_DIR/steering_utility_summary.csv"
  --out_manifest_csv "$OUT_DIR/steering_manifest.csv"
  --selection_pairs "e1_content,e1_content_swapped"
  --direction_mode "label_conditional"
  --top_k_components "$TOP_K_COMPONENTS"
  --alphas "$ALPHAS"
)
if [[ -n "$MAX_TRAIN_ROWS" ]]; then
  STEER_ARGS+=(--max_train_rows "$MAX_TRAIN_ROWS")
fi
if [[ -n "$MAX_VALIDATION_ROWS" ]]; then
  STEER_ARGS+=(--max_validation_rows "$MAX_VALIDATION_ROWS")
fi
if [[ -n "$MAX_EVAL_ROWS" ]]; then
  STEER_ARGS+=(--max_eval_rows "$MAX_EVAL_ROWS")
fi
python "${STEER_ARGS[@]}"

echo "[large-v1-model-loop] component E2"
COMP_E2_ARGS=(
  -m screscomp.cli.score_component_competition
  "${COMMON_ARGS[@]}"
  --out_rows_jsonl "$OUT_DIR/component_competition_e2.jsonl"
  --out_summary_csv "$OUT_DIR/component_competition_e2_summary.csv"
  --layers "$LAYERS"
  --components "$COMPONENTS"
  --pairs "e2_task,e2_task_swapped,e2_source,e2_source_swapped"
  --splits "discovery,validation,test"
)
if [[ -n "$MAX_COMPONENT_ROWS" ]]; then
  COMP_E2_ARGS+=(--max_rows "$MAX_COMPONENT_ROWS")
fi
python "${COMP_E2_ARGS[@]}"

echo "[large-v1-model-loop] done"
echo "Output dir: $OUT_DIR"
