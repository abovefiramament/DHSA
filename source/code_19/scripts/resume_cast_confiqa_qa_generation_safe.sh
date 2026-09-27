#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-3}"
ROOT="${ROOT:-data_ckplug/cast_confiqa_all3_smalltrain_heldout5000_v0/qa}"
MODEL="${MODEL:-LOCAL_HOME/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct}"
CONTEXT_DPO_MODEL="${CONTEXT_DPO_MODEL:-LOCAL_HOME/.cache/huggingface/hub/models--Bibaolong--Context-Faithful-LLaMA-3-8b-instruct}"
RUN_CONTEXT_DPO="${RUN_CONTEXT_DPO:-1}"

MLP_ALPHA="${MLP_ALPHA:-0.05}"
HEAD_ALPHA="${HEAD_ALPHA:-0.5}"
EVAL_ROWS="${EVAL_ROWS:-0}"

EVAL_OPEN_ROWS="$ROOT/eval_open_rows.jsonl"
MLP_OUT="$ROOT/train_decision_tokens_prior_mlp"
HEAD_SUPPRESS_OUT="$ROOT/train_decision_tokens_head_suppress"
HEAD_BOOST_OUT="$ROOT/train_decision_tokens_head_boost"

COMP_ACTUATORS="prior_mlp=$MLP_OUT/fixed_actuator.pt"
HEAD_ACTUATORS="suppress=$HEAD_SUPPRESS_OUT/head_actuator.pt;boost=$HEAD_BOOST_OUT/head_actuator.pt"

run_generation() {
  local model="$1"
  local prompt_key="$2"
  local out="$3"
  local controls="$4"
  local comp_actuators="${5:-}"
  local head_actuators="${6:-}"
  mkdir -p "$out"
  if [[ -f "$out/run_summary.csv" ]] && awk -F, '
    NR > 1 && $1 == "eval_rows" { eval_rows = $2 + 0 }
    NR > 1 && $1 == "generation_rows" { generation_rows = $2 + 0 }
    END { exit !(eval_rows > 0 && generation_rows >= eval_rows) }
  ' "$out/run_summary.csv"; then
    echo "[$(date -Is)] skip complete out=$out"
    return 0
  fi
  echo "[$(date -Is)] generation model=$model prompt=$prompt_key out=$out controls=$controls"
  CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.cecm_run_joint_actuator_generation \
    --model "$model" \
    --eval-open-rows "$EVAL_OPEN_ROWS" \
    --component-actuators "$comp_actuators" \
    --head-actuators "$head_actuators" \
    --controls "$controls" \
    --generation-prompt-key "$prompt_key" \
    --prior-source dataset_orig \
    --split all \
    --start 0 \
    --max-rows "$EVAL_ROWS" \
    --generation-apply-mode prefill \
    --max-new-tokens 64 \
    --stop-strings "Q:" \
    --torch-dtype bfloat16 \
    --device cuda \
    --out-dir "$out" \
    > "$out/generation.log" 2>&1
}

run_generation \
  "$MODEL" \
  base_rag \
  "$ROOT/eval_cast_attn_base_rag" \
  "cast_attn=head_act:suppress:${HEAD_ALPHA}:all+head_act:boost:${HEAD_ALPHA}:all" \
  "$COMP_ACTUATORS" \
  "$HEAD_ACTUATORS"

run_generation \
  "$MODEL" \
  base_rag \
  "$ROOT/eval_cast_full_base_rag" \
  "cast_full=head_act:suppress:${HEAD_ALPHA}:all+head_act:boost:${HEAD_ALPHA}:all+comp:prior_mlp:${MLP_ALPHA}:prefill" \
  "$COMP_ACTUATORS" \
  "$HEAD_ACTUATORS"

if [[ "$RUN_CONTEXT_DPO" == "1" ]]; then
  export SCRESCOMP_PEFT_BASE_MODEL="$MODEL"
  run_generation \
    "$CONTEXT_DPO_MODEL" \
    base_rag \
    "$ROOT/eval_context_dpo_base_rag" \
    "context_dpo=;"
fi

echo "[$(date -Is)] done"
find "$ROOT" -maxdepth 2 \( -name generation_summary.csv -o -name run_summary.csv \) -print | sort
