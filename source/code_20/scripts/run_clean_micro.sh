#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_clean_micro.sh --seed-csv PATH --model PATH_OR_HF_ID [options]

Options:
  --out-dir PATH             Output data root. Default: data
  --device DEVICE            cpu|cuda|auto. Default: auto
  --torch-dtype DTYPE        auto|float16|bfloat16|float32. Default: auto
  --use-chat-template        Apply tokenizer chat template before A/B scoring.
  --num-counters N           Counter candidates per fact. Default: 1
  --seed N                   Random seed. Default: 42
  --group-key KEY            subject_id|fact_id. Default: subject_id
  --transformers-src PATH    Optional local transformers checkout, usually vendor/transformers/src.
  --hf-endpoint URL          Optional HF endpoint, e.g. https://hf-mirror.com.
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SEED_CSV=""
MODEL=""
OUT_DIR="data"
DEVICE="auto"
TORCH_DTYPE="auto"
USE_CHAT_TEMPLATE=0
NUM_COUNTERS=1
SEED=42
GROUP_KEY="subject_id"
TRANSFORMERS_SRC=""
HF_ENDPOINT_VALUE=""
DISCOVERY_TEMPLATES="main_v1,discovery_v2,discovery_v3,discovery_v4"
HELDOUT_TEMPLATES="heldout_v1,heldout_v2,heldout_v3,heldout_v4"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed-csv) SEED_CSV="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --torch-dtype) TORCH_DTYPE="$2"; shift 2 ;;
    --use-chat-template) USE_CHAT_TEMPLATE=1; shift ;;
    --num-counters) NUM_COUNTERS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --group-key) GROUP_KEY="$2"; shift 2 ;;
    --transformers-src) TRANSFORMERS_SRC="$2"; shift 2 ;;
    --hf-endpoint) HF_ENDPOINT_VALUE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$SEED_CSV" || -z "$MODEL" ]]; then
  usage >&2
  exit 2
fi

if [[ -n "$HF_ENDPOINT_VALUE" ]]; then
  export HF_ENDPOINT="$HF_ENDPOINT_VALUE"
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

INTERMEDIATE_DIR="$OUT_DIR/01_intermediate"
PROMPT_DIR="$OUT_DIR/02_prompts"
SPLIT_DIR="$OUT_DIR/03_splits"
REPORT_DIR="$OUT_DIR/04_reports"
mkdir -p "$INTERMEDIATE_DIR" "$PROMPT_DIR" "$SPLIT_DIR" "$REPORT_DIR"

echo "[clean-micro] build"
python -m screscomp.cli.build_dataset \
  --seed_csv "$SEED_CSV" \
  --out_dir "$OUT_DIR" \
  --num_counters "$NUM_COUNTERS" \
  --seed "$SEED" \
  --template_family main_v1 \
  --out_funnel_csv "$REPORT_DIR/sample_funnel_build.csv"

echo "[clean-micro] probe with discovery-template stability"
PROBE_ARGS=(
  -m screscomp.cli.run_prior_probe
  --in_jsonl "$INTERMEDIATE_DIR/04_prompt_ready_pairs.jsonl"
  --model "$MODEL"
  --device "$DEVICE"
  --torch_dtype "$TORCH_DTYPE"
  --out_retained "$INTERMEDIATE_DIR/05_retained_pairs.jsonl"
  --out_excluded "$INTERMEDIATE_DIR/05_excluded_pairs.jsonl"
  --out_funnel_csv "$REPORT_DIR/sample_funnel_probe.csv"
  --out_exclusion_stats_csv "$REPORT_DIR/exclusion_stats.csv"
  --stability_template_families "$DISCOVERY_TEMPLATES"
  --require_template_stability
  --out_template_scores_jsonl "$REPORT_DIR/template_prior_scores.jsonl"
)
if [[ "$USE_CHAT_TEMPLATE" -eq 1 ]]; then
  PROBE_ARGS+=(--use_chat_template)
fi
python "${PROBE_ARGS[@]}"

echo "[clean-micro] split"
python -m screscomp.cli.make_splits \
  --retained_jsonl "$INTERMEDIATE_DIR/05_retained_pairs.jsonl" \
  --out_manifest "$SPLIT_DIR/split_manifest.json" \
  --group_key "$GROUP_KEY" \
  --seed "$SEED" \
  --discovery_template_families "$DISCOVERY_TEMPLATES" \
  --heldout_template_families "$HELDOUT_TEMPLATES" \
  --validation_template_mode discovery_only

echo "[clean-micro] render all templates allowed by split"
python -m screscomp.cli.render_prompts \
  --retained_jsonl "$INTERMEDIATE_DIR/05_retained_pairs.jsonl" \
  --split_manifest "$SPLIT_DIR/split_manifest.json" \
  --out_dir "$PROMPT_DIR" \
  --template_mode all_by_split

echo "[clean-micro] audit"
python -m screscomp.cli.audit_prompts \
  --rendered_jsonl "$PROMPT_DIR/09_rendered_samples.jsonl" \
  --split_manifest "$SPLIT_DIR/split_manifest.json" \
  --out_json "$REPORT_DIR/prompt_audit.json" \
  --strict

echo "[clean-micro] done"
echo "Rendered samples: $PROMPT_DIR/09_rendered_samples.jsonl"
echo "Audit report:     $REPORT_DIR/prompt_audit.json"
