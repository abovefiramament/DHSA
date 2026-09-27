#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-1}"
BASE_ROOT="${BASE_ROOT:-data_ckplug/cast_confiqa_all3_smalltrain_heldout5000_v0/qa}"
OUT="${OUT:-data_ckplug/cast_confiqa_qa_calib60_similarity_v0}"
MODEL="${MODEL:-LOCAL_HOME/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct}"

PAIRS="$BASE_ROOT/pairs/pairs.csv"
COMPONENTS="$BASE_ROOT/discovery/selected/prior_mlp_components.csv"
SUPPRESS_HEADS="${SUPPRESS_HEADS:-L13.attn.h17,L14.attn.h7,L14.attn.h30,L14.attn.h12}"
BOOST_HEADS="${BOOST_HEADS:-L13.attn.h18,L31.attn.h14,L14.attn.h20,L31.attn.h3}"

# Strict 60-source-row calibration under VAL_MOD=5: 48 train rows + 12 val rows.
TRAIN_ROWS="${TRAIN_ROWS:-48}"
VAL_ROWS="${VAL_ROWS:-12}"
EPOCHS="${EPOCHS:-2}"

mkdir -p "$OUT"
cat > "$OUT/config.env" <<EOF
GPU=$GPU
BASE_ROOT=$BASE_ROOT
MODEL=$MODEL
TRAIN_ROWS=$TRAIN_ROWS
VAL_ROWS=$VAL_ROWS
EPOCHS=$EPOCHS
SUPPRESS_HEADS=$SUPPRESS_HEADS
BOOST_HEADS=$BOOST_HEADS
EOF

echo "[$(date -Is)] train calib60 MLP"
CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.cecm_train_fixed_actuator \
  --model "$MODEL" \
  --pairs-csv "$PAIRS" \
  --components-csv "$COMPONENTS" \
  --event source_context_over_prior \
  --train-split train \
  --val-split val \
  --max-train-rows "$TRAIN_ROWS" \
  --max-val-rows "$VAL_ROWS" \
  --epochs "$EPOCHS" \
  --lr 0.05 \
  --lambda-norm 0.0001 \
  --alpha-train 1.0 \
  --apply-mode decision_tokens \
  --score-mode answer_rest_margin \
  --alpha-sweep 0,0.25,0.5,0.75,1.0,1.5,2.0 \
  --torch-dtype bfloat16 \
  --device cuda \
  --empty-cache-every 25 \
  --out-dir "$OUT/train_calib60_prior_mlp" \
  > "$OUT/train_calib60_prior_mlp.log" 2>&1

train_head() {
  local name="$1"
  local heads="$2"
  local out_dir="$OUT/train_calib60_head_${name}"
  echo "[$(date -Is)] train calib60 head $name"
  CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.cecm_train_attention_head_actuator \
    --model "$MODEL" \
    --pairs-csv "$PAIRS" \
    --event source_context_over_prior \
    --heads "$heads" \
    --train-split train \
    --val-split val \
    --max-train-rows "$TRAIN_ROWS" \
    --max-val-rows "$VAL_ROWS" \
    --epochs "$EPOCHS" \
    --lr 0.05 \
    --lambda-norm 0.0001 \
    --alpha-train 1.0 \
    --apply-mode decision_tokens \
    --score-mode answer_rest_margin \
    --alpha-sweep 0,0.25,0.5,1.0,1.5 \
    --torch-dtype bfloat16 \
    --device cuda \
    --out-dir "$out_dir" \
    > "$out_dir.log" 2>&1
}

train_head suppress "$SUPPRESS_HEADS"
train_head boost "$BOOST_HEADS"

echo "[$(date -Is)] compare vectors"
python -m screscomp.cli.cecm_compare_actuator_similarity \
  --name prior_mlp_300src_vs_60src \
  --reference "$BASE_ROOT/train_decision_tokens_prior_mlp/fixed_actuator.pt" \
  --candidate "$OUT/train_calib60_prior_mlp/fixed_actuator.pt" \
  --out-csv "$OUT/similarity_prior_mlp.csv"

python -m screscomp.cli.cecm_compare_actuator_similarity \
  --name head_suppress_300src_vs_60src \
  --reference "$BASE_ROOT/train_decision_tokens_head_suppress/head_actuator.pt" \
  --candidate "$OUT/train_calib60_head_suppress/head_actuator.pt" \
  --out-csv "$OUT/similarity_head_suppress.csv"

python -m screscomp.cli.cecm_compare_actuator_similarity \
  --name head_boost_300src_vs_60src \
  --reference "$BASE_ROOT/train_decision_tokens_head_boost/head_actuator.pt" \
  --candidate "$OUT/train_calib60_head_boost/head_actuator.pt" \
  --out-csv "$OUT/similarity_head_boost.csv"

echo "[$(date -Is)] done"
find "$OUT" -maxdepth 2 \( -name "similarity_*.csv" -o -name "alpha_summary.csv" -o -name "train_history.csv" \) -print | sort
