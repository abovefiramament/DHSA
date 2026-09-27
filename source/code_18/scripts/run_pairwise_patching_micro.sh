#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_pairwise_patching_micro.sh --model PATH_OR_HF_ID [options]

Options:
  --rendered-jsonl PATH       Input rendered samples. Default: data/02_prompts/09_rendered_samples.jsonl
  --out-patches-jsonl PATH    Output patch rows. Default: data/04_reports/pairwise_patches_micro.jsonl
  --out-summary-csv PATH      Output summary CSV. Default: data/04_reports/pairwise_patching_micro_summary.csv
  --device DEVICE             cpu|cuda|auto. Default: auto
  --torch-dtype DTYPE         auto|float16|bfloat16|float32. Default: auto
  --use-chat-template         Apply tokenizer chat template before A/B scoring.
  --layers LAYERS             Default: 12,16,20,24,28
  --splits SPLITS             Default: discovery,test
  --pairs PAIRS               Default: e1_content,e1_content_swapped,e2_task,e2_source,format_label_control
  --templates TEMPLATES       Optional comma-separated template families.
  --max-rows N                Optional limit after filtering.
  --seed N                    Random seed for random-source control. Default: 42
  --transformers-src PATH     Optional local transformers checkout, usually vendor/transformers/src.
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL=""
RENDERED_JSONL="data/02_prompts/09_rendered_samples.jsonl"
OUT_PATCHES_JSONL="data/04_reports/pairwise_patches_micro.jsonl"
OUT_SUMMARY_CSV="data/04_reports/pairwise_patching_micro_summary.csv"
DEVICE="auto"
TORCH_DTYPE="auto"
USE_CHAT_TEMPLATE=0
LAYERS="12,16,20,24,28"
SPLITS="discovery,test"
PAIRS="e1_content,e1_content_swapped,e2_task,e2_source,format_label_control"
TEMPLATES=""
MAX_ROWS=""
SEED=42
TRANSFORMERS_SRC=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --rendered-jsonl) RENDERED_JSONL="$2"; shift 2 ;;
    --out-patches-jsonl) OUT_PATCHES_JSONL="$2"; shift 2 ;;
    --out-summary-csv) OUT_SUMMARY_CSV="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --torch-dtype) TORCH_DTYPE="$2"; shift 2 ;;
    --use-chat-template) USE_CHAT_TEMPLATE=1; shift ;;
    --layers) LAYERS="$2"; shift 2 ;;
    --splits) SPLITS="$2"; shift 2 ;;
    --pairs) PAIRS="$2"; shift 2 ;;
    --templates) TEMPLATES="$2"; shift 2 ;;
    --max-rows) MAX_ROWS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
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

mkdir -p "$(dirname "$OUT_PATCHES_JSONL")" "$(dirname "$OUT_SUMMARY_CSV")"

ARGS=(
  -m screscomp.cli.score_pairwise_patching
  --rendered_jsonl "$RENDERED_JSONL"
  --model "$MODEL"
  --out_patches_jsonl "$OUT_PATCHES_JSONL"
  --out_summary_csv "$OUT_SUMMARY_CSV"
  --device "$DEVICE"
  --torch_dtype "$TORCH_DTYPE"
  --layers "$LAYERS"
  --splits "$SPLITS"
  --pairs "$PAIRS"
  --include_random_control
  --seed "$SEED"
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
echo "Pairwise patch rows:    $OUT_PATCHES_JSONL"
echo "Pairwise patch summary: $OUT_SUMMARY_CSV"
