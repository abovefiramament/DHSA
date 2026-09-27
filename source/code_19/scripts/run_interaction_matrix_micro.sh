#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_interaction_matrix_micro.sh --model PATH_OR_HF_ID [options]

Options:
  --rendered-jsonl PATH       Input rendered samples. Default: data/02_prompts/09_rendered_samples.jsonl
  --out-rows-jsonl PATH       Default: data/04_reports/interaction_matrix_micro.jsonl
  --out-summary-csv PATH      Default: data/04_reports/interaction_matrix_micro_summary.csv
  --device DEVICE             cpu|cuda|auto. Default: auto
  --torch-dtype DTYPE         auto|float16|bfloat16|float32. Default: auto
  --use-chat-template         Apply tokenizer chat template before A/B scoring.
  --layers LAYERS             Default: 16
  --train-splits SPLITS       Default: discovery
  --eval-splits SPLITS        Default: test
  --alphas ALPHAS             Default: 0.5,1.0,2.0
  --max-train-rows N          Optional train row limit.
  --max-eval-rows N           Optional eval row limit.
  --raw-diffs                 Use raw residual differences instead of unit-normalized per-row differences.
  --no-orthogonalize          Do not project away the A/B format-label direction.
  --uniform-cell              Add the same direction to all four cells instead of signed DID-cell intervention.
  --seed N                    Random seed. Default: 42
  --transformers-src PATH     Optional local transformers checkout, usually vendor/transformers/src.
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL=""
RENDERED_JSONL="data/02_prompts/09_rendered_samples.jsonl"
OUT_ROWS_JSONL="data/04_reports/interaction_matrix_micro.jsonl"
OUT_SUMMARY_CSV="data/04_reports/interaction_matrix_micro_summary.csv"
DEVICE="auto"
TORCH_DTYPE="auto"
USE_CHAT_TEMPLATE=0
LAYERS="16"
TRAIN_SPLITS="discovery"
EVAL_SPLITS="test"
ALPHAS="0.5,1.0,2.0"
MAX_TRAIN_ROWS=""
MAX_EVAL_ROWS=""
NORMALIZE_DIFFS=1
ORTHOGONALIZE_TO_FORMAT_LABEL=1
SIGNED_CELL_INTERVENTION=1
SEED=42
TRANSFORMERS_SRC=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --rendered-jsonl) RENDERED_JSONL="$2"; shift 2 ;;
    --out-rows-jsonl) OUT_ROWS_JSONL="$2"; shift 2 ;;
    --out-summary-csv) OUT_SUMMARY_CSV="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --torch-dtype) TORCH_DTYPE="$2"; shift 2 ;;
    --use-chat-template) USE_CHAT_TEMPLATE=1; shift ;;
    --layers) LAYERS="$2"; shift 2 ;;
    --train-splits) TRAIN_SPLITS="$2"; shift 2 ;;
    --eval-splits) EVAL_SPLITS="$2"; shift 2 ;;
    --alphas) ALPHAS="$2"; shift 2 ;;
    --max-train-rows) MAX_TRAIN_ROWS="$2"; shift 2 ;;
    --max-eval-rows) MAX_EVAL_ROWS="$2"; shift 2 ;;
    --raw-diffs) NORMALIZE_DIFFS=0; shift ;;
    --no-orthogonalize) ORTHOGONALIZE_TO_FORMAT_LABEL=0; shift ;;
    --uniform-cell) SIGNED_CELL_INTERVENTION=0; shift ;;
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

mkdir -p "$(dirname "$OUT_ROWS_JSONL")" "$(dirname "$OUT_SUMMARY_CSV")"

ARGS=(
  -m screscomp.cli.score_interaction_matrix
  --rendered_jsonl "$RENDERED_JSONL"
  --model "$MODEL"
  --out_rows_jsonl "$OUT_ROWS_JSONL"
  --out_summary_csv "$OUT_SUMMARY_CSV"
  --device "$DEVICE"
  --torch_dtype "$TORCH_DTYPE"
  --layers "$LAYERS"
  --train_splits "$TRAIN_SPLITS"
  --eval_splits "$EVAL_SPLITS"
  --directions "ix_interaction_balanced,ix_content_according_balanced,ix_content_actual_balanced,format_label_unbalanced"
  --matrices "normal,swapped"
  --alphas "$ALPHAS"
  --seed "$SEED"
)

if [[ "$USE_CHAT_TEMPLATE" -eq 1 ]]; then
  ARGS+=(--use_chat_template)
fi
if [[ -n "$MAX_TRAIN_ROWS" ]]; then
  ARGS+=(--max_train_rows "$MAX_TRAIN_ROWS")
fi
if [[ -n "$MAX_EVAL_ROWS" ]]; then
  ARGS+=(--max_eval_rows "$MAX_EVAL_ROWS")
fi
if [[ "$NORMALIZE_DIFFS" -eq 1 ]]; then
  ARGS+=(--normalize_diffs)
fi
if [[ "$ORTHOGONALIZE_TO_FORMAT_LABEL" -eq 1 ]]; then
  ARGS+=(--orthogonalize_to_format_label)
fi
if [[ "$SIGNED_CELL_INTERVENTION" -eq 1 ]]; then
  ARGS+=(--signed_cell_intervention)
fi

python "${ARGS[@]}"
echo "Interaction matrix rows:    $OUT_ROWS_JSONL"
echo "Interaction matrix summary: $OUT_SUMMARY_CSV"
