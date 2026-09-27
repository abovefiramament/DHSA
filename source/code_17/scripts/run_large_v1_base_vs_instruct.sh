#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_large_v1_base_vs_instruct.sh --instruct-model PATH --base-model PATH [options]

Options:
  --rendered-jsonl PATH       Input rendered samples. Default: data_large_v1/02_prompts/09_rendered_samples.jsonl
  --instruct-out-dir PATH     Output dir for instruct run. Default: data_large_v1_runs/instruct
  --base-out-dir PATH         Output dir for base run. Default: data_large_v1_runs/base
  --compare-out-csv PATH      Comparison CSV. Default: data_large_v1_runs/base_vs_instruct_compare.csv
  --label-instruct NAME       Comparison label. Default: instruct
  --label-base NAME           Comparison label. Default: base
  --instruct-use-chat-template
  --base-use-chat-template
  --layers SPEC               Layer selection. Default: all
  --components CSV            Component types. Default: attn,mlp
  --top-k-components CSV      Steering prefix sizes. Default: 1,2,4,8
  --alphas CSV                Steering alpha grid. Default: 0.25,0.5,1.0,2.0
  --compare-top-k N           Selected-component comparison prefix size. Default: 4
  --device DEVICE             cpu|cuda|auto. Default: auto
  --torch-dtype DTYPE         auto|float16|bfloat16|float32. Default: auto
  --transformers-src PATH     Optional local transformers checkout, usually vendor/transformers/src.
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

INSTRUCT_MODEL=""
BASE_MODEL=""
RENDERED_JSONL="data_large_v1/02_prompts/09_rendered_samples.jsonl"
INSTRUCT_OUT_DIR="data_large_v1_runs/instruct"
BASE_OUT_DIR="data_large_v1_runs/base"
COMPARE_OUT_CSV="data_large_v1_runs/base_vs_instruct_compare.csv"
LABEL_INSTRUCT="instruct"
LABEL_BASE="base"
INSTRUCT_USE_CHAT_TEMPLATE=0
BASE_USE_CHAT_TEMPLATE=0
LAYERS="all"
COMPONENTS="attn,mlp"
TOP_K_COMPONENTS="1,2,4,8"
ALPHAS="0.25,0.5,1.0,2.0"
COMPARE_TOP_K="4"
DEVICE="auto"
TORCH_DTYPE="auto"
TRANSFORMERS_SRC=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --instruct-model) INSTRUCT_MODEL="$2"; shift 2 ;;
    --base-model) BASE_MODEL="$2"; shift 2 ;;
    --rendered-jsonl) RENDERED_JSONL="$2"; shift 2 ;;
    --instruct-out-dir) INSTRUCT_OUT_DIR="$2"; shift 2 ;;
    --base-out-dir) BASE_OUT_DIR="$2"; shift 2 ;;
    --compare-out-csv) COMPARE_OUT_CSV="$2"; shift 2 ;;
    --label-instruct) LABEL_INSTRUCT="$2"; shift 2 ;;
    --label-base) LABEL_BASE="$2"; shift 2 ;;
    --instruct-use-chat-template) INSTRUCT_USE_CHAT_TEMPLATE=1; shift ;;
    --base-use-chat-template) BASE_USE_CHAT_TEMPLATE=1; shift ;;
    --layers) LAYERS="$2"; shift 2 ;;
    --components) COMPONENTS="$2"; shift 2 ;;
    --top-k-components) TOP_K_COMPONENTS="$2"; shift 2 ;;
    --alphas) ALPHAS="$2"; shift 2 ;;
    --compare-top-k) COMPARE_TOP_K="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --torch-dtype) TORCH_DTYPE="$2"; shift 2 ;;
    --transformers-src) TRANSFORMERS_SRC="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$INSTRUCT_MODEL" || -z "$BASE_MODEL" ]]; then
  usage >&2
  exit 2
fi

mkdir -p "$(dirname "$COMPARE_OUT_CSV")"

INSTRUCT_ARGS=(
  --model "$INSTRUCT_MODEL"
  --out-dir "$INSTRUCT_OUT_DIR"
  --rendered-jsonl "$RENDERED_JSONL"
  --layers "$LAYERS"
  --components "$COMPONENTS"
  --top-k-components "$TOP_K_COMPONENTS"
  --alphas "$ALPHAS"
  --device "$DEVICE"
  --torch-dtype "$TORCH_DTYPE"
)
BASE_ARGS=(
  --model "$BASE_MODEL"
  --out-dir "$BASE_OUT_DIR"
  --rendered-jsonl "$RENDERED_JSONL"
  --layers "$LAYERS"
  --components "$COMPONENTS"
  --top-k-components "$TOP_K_COMPONENTS"
  --alphas "$ALPHAS"
  --device "$DEVICE"
  --torch-dtype "$TORCH_DTYPE"
)
if [[ "$INSTRUCT_USE_CHAT_TEMPLATE" -eq 1 ]]; then
  INSTRUCT_ARGS+=(--use-chat-template)
fi
if [[ "$BASE_USE_CHAT_TEMPLATE" -eq 1 ]]; then
  BASE_ARGS+=(--use-chat-template)
fi
if [[ -n "$TRANSFORMERS_SRC" ]]; then
  INSTRUCT_ARGS+=(--transformers-src "$TRANSFORMERS_SRC")
  BASE_ARGS+=(--transformers-src "$TRANSFORMERS_SRC")
fi

echo "[base-vs-instruct] instruct loop"
bash scripts/run_large_v1_model_loop.sh "${INSTRUCT_ARGS[@]}"

echo "[base-vs-instruct] base loop"
bash scripts/run_large_v1_model_loop.sh "${BASE_ARGS[@]}"

PYTHONPATH_PARTS=("src")
if [[ -z "$TRANSFORMERS_SRC" && -d "vendor/transformers/src" ]]; then
  TRANSFORMERS_SRC="vendor/transformers/src"
fi
if [[ -n "$TRANSFORMERS_SRC" ]]; then
  PYTHONPATH_PARTS=("$TRANSFORMERS_SRC" "${PYTHONPATH_PARTS[@]}")
fi
export PYTHONPATH="$(IFS=:; echo "${PYTHONPATH_PARTS[*]}")${PYTHONPATH:+:$PYTHONPATH}"

python -m screscomp.cli.compare_model_runs \
  --run_a_dir "$BASE_OUT_DIR" \
  --run_b_dir "$INSTRUCT_OUT_DIR" \
  --label_a "$LABEL_BASE" \
  --label_b "$LABEL_INSTRUCT" \
  --compare_top_k "$COMPARE_TOP_K" \
  --out_csv "$COMPARE_OUT_CSV"

echo "Instruct dir:    $INSTRUCT_OUT_DIR"
echo "Base dir:        $BASE_OUT_DIR"
echo "Comparison CSV:  $COMPARE_OUT_CSV"
