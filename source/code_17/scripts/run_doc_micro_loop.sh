#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_doc_micro_loop.sh --model PATH_OR_HF_ID [options]

Runs the document-locked small loop:
  render/audit -> behavior -> component C/I -> discovery-selected E5-lite steering

Options:
  --retained-jsonl PATH         Default: data/01_intermediate/05_retained_pairs.jsonl
  --split-manifest PATH         Default: data/03_splits/split_manifest.json
  --rendered-jsonl PATH         Default: data/02_prompts/09_rendered_samples.jsonl
  --device DEVICE               cpu|cuda|auto. Default: auto
  --torch-dtype DTYPE           auto|float16|bfloat16|float32. Default: auto
  --use-chat-template           Apply tokenizer chat template before A/B scoring.
  --layers LAYERS               Component layers for C/I. Default: 12-24; use all for full scan
  --components LIST             Default: attn,mlp
  --component-pairs LIST        Default: e1_content,e1_content_swapped
  --top-k-components N          Default: 4
  --direction-mode MODE         shared|label_conditional. Default: label_conditional
  --alphas LIST                 Default: 0.25,0.5,1.0,2.0
  --max-component-rows N        Optional row limit for component C/I smoke tests.
  --max-train-rows N            Optional steering discovery row limit.
  --max-validation-rows N       Optional steering validation row limit.
  --max-eval-rows N             Optional steering held-out row limit.
  --skip-render                 Reuse existing rendered/audited prompts.
  --normalize-diffs             Unit-normalize per-row component deltas before averaging.
  --transformers-src PATH       Optional local transformers checkout, usually vendor/transformers/src.
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL=""
RETAINED_JSONL="data/01_intermediate/05_retained_pairs.jsonl"
SPLIT_MANIFEST="data/03_splits/split_manifest.json"
RENDERED_JSONL="data/02_prompts/09_rendered_samples.jsonl"
DEVICE="auto"
TORCH_DTYPE="auto"
USE_CHAT_TEMPLATE=0
LAYERS="12-24"
COMPONENTS="attn,mlp"
COMPONENT_PAIRS="e1_content,e1_content_swapped"
TOP_K_COMPONENTS=4
DIRECTION_MODE="label_conditional"
ALPHAS="0.25,0.5,1.0,2.0"
MAX_COMPONENT_ROWS=""
MAX_TRAIN_ROWS=""
MAX_VALIDATION_ROWS=""
MAX_EVAL_ROWS=""
SKIP_RENDER=0
NORMALIZE_DIFFS=0
TRANSFORMERS_SRC=""

BEHAVIOR_ROWS="data/04_reports/doc_loop_behavior_scores.jsonl"
BEHAVIOR_SUMMARY="data/04_reports/doc_loop_behavior_summary.csv"
COMP_ROWS="data/04_reports/doc_loop_component_competition.jsonl"
COMP_SUMMARY="data/04_reports/doc_loop_component_competition_summary.csv"
STEER_ROWS="data/04_reports/doc_loop_steering_utility.jsonl"
STEER_SUMMARY="data/04_reports/doc_loop_steering_utility_summary.csv"
STEER_MANIFEST="data/04_reports/doc_loop_steering_manifest.csv"
AUDIT_JSON="data/04_reports/prompt_audit.json"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --retained-jsonl) RETAINED_JSONL="$2"; shift 2 ;;
    --split-manifest) SPLIT_MANIFEST="$2"; shift 2 ;;
    --rendered-jsonl) RENDERED_JSONL="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --torch-dtype) TORCH_DTYPE="$2"; shift 2 ;;
    --use-chat-template) USE_CHAT_TEMPLATE=1; shift ;;
    --layers) LAYERS="$2"; shift 2 ;;
    --components) COMPONENTS="$2"; shift 2 ;;
    --component-pairs) COMPONENT_PAIRS="$2"; shift 2 ;;
    --top-k-components) TOP_K_COMPONENTS="$2"; shift 2 ;;
    --direction-mode) DIRECTION_MODE="$2"; shift 2 ;;
    --alphas) ALPHAS="$2"; shift 2 ;;
    --max-component-rows) MAX_COMPONENT_ROWS="$2"; shift 2 ;;
    --max-train-rows) MAX_TRAIN_ROWS="$2"; shift 2 ;;
    --max-validation-rows) MAX_VALIDATION_ROWS="$2"; shift 2 ;;
    --max-eval-rows) MAX_EVAL_ROWS="$2"; shift 2 ;;
    --skip-render) SKIP_RENDER=1; shift ;;
    --normalize-diffs) NORMALIZE_DIFFS=1; shift ;;
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

mkdir -p data/02_prompts data/04_reports

CHAT_ARGS=()
if [[ "$USE_CHAT_TEMPLATE" -eq 1 ]]; then
  CHAT_ARGS+=(--use_chat_template)
fi

if [[ "$SKIP_RENDER" -eq 0 ]]; then
  echo "[doc-loop] render prompts"
  python -m screscomp.cli.render_prompts \
    --retained_jsonl "$RETAINED_JSONL" \
    --split_manifest "$SPLIT_MANIFEST" \
    --out_dir data/02_prompts \
    --template_mode all_by_split

  echo "[doc-loop] audit prompts"
  python -m screscomp.cli.audit_prompts \
    --rendered_jsonl "$RENDERED_JSONL" \
    --split_manifest "$SPLIT_MANIFEST" \
    --out_json "$AUDIT_JSON" \
    --strict
fi

echo "[doc-loop] behavior"
python -m screscomp.cli.score_behavior \
  --rendered_jsonl "$RENDERED_JSONL" \
  --model "$MODEL" \
  --out_scores_jsonl "$BEHAVIOR_ROWS" \
  --out_summary_csv "$BEHAVIOR_SUMMARY" \
  --device "$DEVICE" \
  --torch_dtype "$TORCH_DTYPE" \
  "${CHAT_ARGS[@]}"

COMP_ARGS=(
  -m screscomp.cli.score_component_competition
  --rendered_jsonl "$RENDERED_JSONL"
  --model "$MODEL"
  --out_rows_jsonl "$COMP_ROWS"
  --out_summary_csv "$COMP_SUMMARY"
  --device "$DEVICE"
  --torch_dtype "$TORCH_DTYPE"
  --layers "$LAYERS"
  --components "$COMPONENTS"
  --pairs "$COMPONENT_PAIRS"
  --splits "discovery,validation,test"
)
if [[ "$USE_CHAT_TEMPLATE" -eq 1 ]]; then
  COMP_ARGS+=(--use_chat_template)
fi
if [[ -n "$MAX_COMPONENT_ROWS" ]]; then
  COMP_ARGS+=(--max_rows "$MAX_COMPONENT_ROWS")
fi

echo "[doc-loop] component C/I"
python "${COMP_ARGS[@]}"

STEER_ARGS=(
  -m screscomp.cli.score_steering_utility
  --rendered_jsonl "$RENDERED_JSONL"
  --component_summary_csv "$COMP_SUMMARY"
  --model "$MODEL"
  --out_rows_jsonl "$STEER_ROWS"
  --out_summary_csv "$STEER_SUMMARY"
  --out_manifest_csv "$STEER_MANIFEST"
  --device "$DEVICE"
  --torch_dtype "$TORCH_DTYPE"
  --top_k_components "$TOP_K_COMPONENTS"
  --direction_mode "$DIRECTION_MODE"
  --alphas "$ALPHAS"
)
if [[ "$USE_CHAT_TEMPLATE" -eq 1 ]]; then
  STEER_ARGS+=(--use_chat_template)
fi
if [[ "$NORMALIZE_DIFFS" -eq 1 ]]; then
  STEER_ARGS+=(--normalize_diffs)
fi
if [[ -n "$MAX_TRAIN_ROWS" ]]; then
  STEER_ARGS+=(--max_train_rows "$MAX_TRAIN_ROWS")
fi
if [[ -n "$MAX_VALIDATION_ROWS" ]]; then
  STEER_ARGS+=(--max_validation_rows "$MAX_VALIDATION_ROWS")
fi
if [[ -n "$MAX_EVAL_ROWS" ]]; then
  STEER_ARGS+=(--max_eval_rows "$MAX_EVAL_ROWS")
fi

echo "[doc-loop] steering utility"
python "${STEER_ARGS[@]}"

echo "[doc-loop] done"
echo "Behavior summary:      $BEHAVIOR_SUMMARY"
echo "Component C/I summary: $COMP_SUMMARY"
echo "Steering summary:      $STEER_SUMMARY"
echo "Steering manifest:     $STEER_MANIFEST"
