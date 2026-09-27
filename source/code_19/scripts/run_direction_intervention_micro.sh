#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_direction_intervention_micro.sh --model PATH_OR_HF_ID [options]

Options:
  --rendered-jsonl PATH       Input rendered samples. Default: data/02_prompts/09_rendered_samples.jsonl
  --out-rows-jsonl PATH       Output intervention rows. Default: data/04_reports/direction_intervention_micro.jsonl
  --out-summary-csv PATH      Output summary CSV. Default: data/04_reports/direction_intervention_micro_summary.csv
  --device DEVICE             cpu|cuda|auto. Default: auto
  --torch-dtype DTYPE         auto|float16|bfloat16|float32. Default: auto
  --use-chat-template         Apply tokenizer chat template before A/B scoring.
  --layers LAYERS             Default: 16,20,24,28
  --train-splits SPLITS       Default: discovery
  --eval-splits SPLITS        Default: test
  --directions DIRS           Default includes balanced task/source/content and format-label controls
  --eval-pairs PAIRS          Default includes normal/swapped task/source/content and format-label controls
  --alphas ALPHAS             Default: 1.0
  --max-train-rows N          Optional train row limit.
  --max-eval-rows N           Optional eval row limit.
  --normalize-diffs           Average unit per-row differences instead of raw differences.
  --orthogonalize-to-format-label
                              Project non-format directions away from the unbalanced A/B format-label direction.
  --seed N                    Random seed. Default: 42
  --transformers-src PATH     Optional local transformers checkout, usually vendor/transformers/src.
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL=""
RENDERED_JSONL="data/02_prompts/09_rendered_samples.jsonl"
OUT_ROWS_JSONL="data/04_reports/direction_intervention_micro.jsonl"
OUT_SUMMARY_CSV="data/04_reports/direction_intervention_micro_summary.csv"
DEVICE="auto"
TORCH_DTYPE="auto"
USE_CHAT_TEMPLATE=0
LAYERS="16,20,24,28"
TRAIN_SPLITS="discovery"
EVAL_SPLITS="test"
DIRECTIONS="e2_task_balanced,e2_source_balanced,e1_content_balanced,format_label_unbalanced,format_label_balanced"
EVAL_PAIRS="e2_task,e2_task_swapped,e2_source,e2_source_swapped,e1_content,e1_content_swapped,format_label_control,format_label_control_swapped"
ALPHAS="1.0"
MAX_TRAIN_ROWS=""
MAX_EVAL_ROWS=""
NORMALIZE_DIFFS=0
ORTHOGONALIZE_TO_FORMAT_LABEL=0
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
    --directions) DIRECTIONS="$2"; shift 2 ;;
    --eval-pairs) EVAL_PAIRS="$2"; shift 2 ;;
    --alphas) ALPHAS="$2"; shift 2 ;;
    --max-train-rows) MAX_TRAIN_ROWS="$2"; shift 2 ;;
    --max-eval-rows) MAX_EVAL_ROWS="$2"; shift 2 ;;
    --normalize-diffs) NORMALIZE_DIFFS=1; shift ;;
    --orthogonalize-to-format-label) ORTHOGONALIZE_TO_FORMAT_LABEL=1; shift ;;
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
  -m screscomp.cli.score_direction_intervention
  --rendered_jsonl "$RENDERED_JSONL"
  --model "$MODEL"
  --out_rows_jsonl "$OUT_ROWS_JSONL"
  --out_summary_csv "$OUT_SUMMARY_CSV"
  --device "$DEVICE"
  --torch_dtype "$TORCH_DTYPE"
  --layers "$LAYERS"
  --train_splits "$TRAIN_SPLITS"
  --eval_splits "$EVAL_SPLITS"
  --directions "$DIRECTIONS"
  --eval_pairs "$EVAL_PAIRS"
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

python "${ARGS[@]}"
echo "Direction rows:    $OUT_ROWS_JSONL"
echo "Direction summary: $OUT_SUMMARY_CSV"
