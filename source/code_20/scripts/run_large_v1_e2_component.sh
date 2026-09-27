#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_large_v1_e2_component.sh --model PATH_OR_HF_ID [options]

Options:
  --rendered-jsonl PATH       Input rendered samples. Default: data_large_v1/02_prompts/09_rendered_samples.jsonl
  --out-rows-jsonl PATH       Output per-component rows. Default: data_large_v1/04_reports/component_competition_e2.jsonl
  --out-summary-csv PATH      Output component summary CSV. Default: data_large_v1/04_reports/component_competition_e2_summary.csv
  --layers SPEC               Layer selection, e.g. all or 12-24. Default: all
  --components CSV            Component types. Default: attn,mlp
  --splits CSV                Split filter. Default: discovery,validation,test
  --templates CSV             Optional template-family filter.
  --max-rows N                Optional row limit for smoke tests.
  --device DEVICE             cpu|cuda|auto. Default: auto
  --torch-dtype DTYPE         auto|float16|bfloat16|float32. Default: auto
  --use-chat-template         Apply tokenizer chat template before scoring.
  --transformers-src PATH     Optional local transformers checkout, usually vendor/transformers/src.
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL=""
RENDERED_JSONL="data_large_v1/02_prompts/09_rendered_samples.jsonl"
OUT_ROWS_JSONL="data_large_v1/04_reports/component_competition_e2.jsonl"
OUT_SUMMARY_CSV="data_large_v1/04_reports/component_competition_e2_summary.csv"
LAYERS="all"
COMPONENTS="attn,mlp"
SPLITS="discovery,validation,test"
TEMPLATES=""
MAX_ROWS=""
DEVICE="auto"
TORCH_DTYPE="auto"
USE_CHAT_TEMPLATE=0
TRANSFORMERS_SRC=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --rendered-jsonl) RENDERED_JSONL="$2"; shift 2 ;;
    --out-rows-jsonl) OUT_ROWS_JSONL="$2"; shift 2 ;;
    --out-summary-csv) OUT_SUMMARY_CSV="$2"; shift 2 ;;
    --layers) LAYERS="$2"; shift 2 ;;
    --components) COMPONENTS="$2"; shift 2 ;;
    --splits) SPLITS="$2"; shift 2 ;;
    --templates) TEMPLATES="$2"; shift 2 ;;
    --max-rows) MAX_ROWS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --torch-dtype) TORCH_DTYPE="$2"; shift 2 ;;
    --use-chat-template) USE_CHAT_TEMPLATE=1; shift ;;
    --transformers-src) TRANSFORMERS_SRC="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$MODEL" ]]; then
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

mkdir -p "$(dirname "$OUT_ROWS_JSONL")" "$(dirname "$OUT_SUMMARY_CSV")"

ARGS=(
  -m screscomp.cli.score_component_competition
  --rendered_jsonl "$RENDERED_JSONL"
  --model "$MODEL"
  --out_rows_jsonl "$OUT_ROWS_JSONL"
  --out_summary_csv "$OUT_SUMMARY_CSV"
  --device "$DEVICE"
  --torch_dtype "$TORCH_DTYPE"
  --layers "$LAYERS"
  --components "$COMPONENTS"
  --pairs "e2_task,e2_task_swapped,e2_source,e2_source_swapped"
  --splits "$SPLITS"
)
if [[ "$USE_CHAT_TEMPLATE" -eq 1 ]]; then
  ARGS+=(--use_chat_template)
fi
if [[ -n "$TEMPLATES" ]]; then
  ARGS+=(--templates "$TEMPLATES")
fi
if [[ -n "$MAX_ROWS" ]]; then
  ARGS+=(--max_rows "$MAX_ROWS")
fi

python "${ARGS[@]}"
echo "E2 component rows:    $OUT_ROWS_JSONL"
echo "E2 component summary: $OUT_SUMMARY_CSV"
