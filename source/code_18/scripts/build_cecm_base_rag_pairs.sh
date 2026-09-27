#!/usr/bin/env bash
set -euo pipefail

# Build the clean source-conflict preference pairs used by the main CECM setting.
# The prompt is minimal context exposure:
#   {context}
#   Q:{question}
#   A:
# No explicit instruction says that the model must use the context.

cd "$(dirname "${BASH_SOURCE[0]}")/.."

DATA="${DATA:-data_ckplug/confiqa_large_ck_default_test5000/02_prompts/confiqa_open_rows.jsonl}"
OUT="${OUT:-data_ckplug/cecm_source_context_over_prior_base_step1}"
MAX_ROWS="${MAX_ROWS:-}"
PRIOR_SOURCE="${PRIOR_SOURCE:-dataset_orig}"

test -f "$DATA"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  deactivate || true
fi
source LOCAL_HOME/anaconda3/etc/profile.d/conda.sh
conda activate screscomp
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

mkdir -p "$OUT"

args=(
  python -m screscomp.cli.cecm_build_preference_pairs
  --input-jsonl "$DATA"
  --out-dir "$OUT"
  --event source_context_over_prior
  --prompt-key base_rag
  --prior-source "$PRIOR_SOURCE"
)

if [[ -n "$MAX_ROWS" ]]; then
  args+=(--max-rows "$MAX_ROWS")
fi

"${args[@]}"
echo "built pairs: $OUT/pairs.csv"
