#!/usr/bin/env bash
set -euo pipefail

# Formal IMDb post-O attention-component rerun.
# Attention controls are trained as residual-dimension fixed actuators on Lx.attn
# module outputs, i.e. after the attention output projection has written into the
# residual stream. MLP controls keep the locked prefill timing.

export LC_ALL=C.UTF-8
cd LOCAL_HOME/RPEC/projects/screscomp
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

PY="${PY:-LOCAL_HOME/anaconda3/envs/screscomp/bin/python}"
ROOT="${ROOT:-/dev/shm/screscomp_runs/imdb_ma921_expanded1024_sampled_paperkl_20260626_055222}"
MODEL="${MODEL:-LOCAL_HOME/.cache/huggingface/hub/models--ma921--gpt2-large-sft-imdb/snapshots/f480190690d5abfc0e003ccb4f7e650626019bb9}"
OUT="${OUT:-/dev/shm/screscomp_runs/imdb_post_o_attention_components_$(date -u +%Y%m%d_%H%M%S)}"
GPU="${GPU:-3}"

TRAIN_ROWS="${TRAIN_ROWS:-512}"
VAL_ROWS="${VAL_ROWS:-256}"
ALPHA_SWEEP="${ALPHA_SWEEP:-0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}"
NONZERO_SWEEP="${NONZERO_SWEEP:-0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}"
GEN_BATCH="${GEN_BATCH:-32}"
SCORE_BATCH="${SCORE_BATCH:-32}"
KL_BATCH="${KL_BATCH:-32}"
EVAL_SPLIT="${EVAL_SPLIT:-eval}"
EVAL_ROWS="${EVAL_ROWS:-}"

POS_MLP_COMPONENTS="${POS_MLP_COMPONENTS:-$ROOT/components/mlp_positive_components.csv}"
NEG_MLP_COMPONENTS="${NEG_MLP_COMPONENTS:-$ROOT/components/mlp_negative_components.csv}"
POS_ATTN_COMPONENTS="${POS_ATTN_COMPONENTS:-$ROOT/components/attn_positive_components.csv}"
NEG_ATTN_COMPONENTS="${NEG_ATTN_COMPONENTS:-$ROOT/components/attn_negative_components.csv}"

MLP_ROOT="${MLP_ROOT:-/dev/shm/screscomp_runs/imdb_zero512_mlp_then_head_20260627_091106}"
MLP_POS_PAYLOAD="${MLP_POS_PAYLOAD:-$MLP_ROOT/train/mlp_positive_zero_train_512/fixed_actuator.pt}"
MLP_NEG_PAYLOAD="${MLP_NEG_PAYLOAD:-$MLP_ROOT/train/mlp_negative_zero_train_512/fixed_actuator.pt}"
TRAIN_MLP_IF_MISSING="${TRAIN_MLP_IF_MISSING:-1}"

export CUDA_VISIBLE_DEVICES="$GPU"
STATUS="$OUT/status.tsv"
mkdir -p "$OUT/train" "$OUT/evaluation/val_select"
printf 'time\tstage\tstatus\n' > "$STATUS"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$OUT/master.log"
}

stage() {
  printf '%s\t%s\t%s\n' "$(date -Is)" "$1" "$2" >> "$STATUS"
  log "stage=$1 $2"
}

component_ids() {
  "$PY" - "$1" <<'PY'
import csv, sys
with open(sys.argv[1], newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
print(",".join(row["component_id"] for row in rows if row.get("component_id")))
PY
}

score_and_kl() {
  local out_dir="$1"
  "$PY" -m screscomp.cli.score_imdb_sentiment_generations \
    --input "$out_dir/generations.jsonl" \
    --out-jsonl "$out_dir/scored_generations.jsonl" \
    --summary-csv "$out_dir/score_pre_kl_summary.csv" \
    --batch-size "$SCORE_BATCH" --device 0 --overwrite >"$out_dir/score.log" 2>&1
  "$PY" -m screscomp.cli.compute_imdb_generation_kl \
    --input-jsonl "$out_dir/scored_generations.jsonl" \
    --out-jsonl "$out_dir/kl_scored_generations.jsonl" \
    --summary-csv "$out_dir/score_summary.csv" \
    --model "$MODEL" --batch-size "$KL_BATCH" --device cuda --torch-dtype bfloat16 \
    --sampled-only --overwrite >"$out_dir/kl.log" 2>&1
}

materialize_group_generations() {
  local base_dir="$1"
  local nonzero_path="$2"
  local group="$3"
  local out_path="$4"
  "$PY" - "$base_dir/generations.jsonl" "$nonzero_path" "$group" "$out_path" <<'PY'
import json, sys
from pathlib import Path

base, nonzero, group, out = map(Path, sys.argv[1:5])
rows = []
for line in base.open(encoding="utf-8-sig"):
    if not line.strip():
        continue
    row = json.loads(line)
    row["control_name"] = str(group)
    row["alpha"] = 0.0
    rows.append(row)
for line in nonzero.open(encoding="utf-8-sig"):
    if line.strip():
        rows.append(json.loads(line))
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"[materialize] group={group} rows={len(rows)} out={out}", flush=True)
PY
}

eval_row_args() {
  if [[ -n "$EVAL_ROWS" ]]; then
    printf '%s\n' --max-rows "$EVAL_ROWS"
  fi
}

run_train_component() {
  local name="$1"
  local components_csv="$2"
  local apply_mode="$3"
  local zero_flag="$4"
  local out_dir="$OUT/train/$name"
  local extra=()
  if [[ "$zero_flag" == "1" ]]; then
    extra=(--zero-components train_components --zero-component-apply-mode "$apply_mode")
  fi
  if [[ -f "$out_dir/fixed_actuator.pt" ]]; then
    stage "train_$name" reused
    return
  fi
  stage "train_$name" running
  mkdir -p "$out_dir"
  "$PY" -m screscomp.cli.cecm_train_fixed_actuator \
    --model "$MODEL" \
    --pairs-csv "$ROOT/pairs/pairs.csv" \
    --components-csv "$components_csv" \
    --event imdb_positive_sentiment \
    --endpoint-objective pair_margin \
    --train-split train --val-split val \
    --max-train-rows "$TRAIN_ROWS" --max-val-rows "$VAL_ROWS" \
    --epochs 2 --train-batch-size 32 \
    --lr 0.05 --lambda-norm 0.0001 --alpha-train 1.0 \
    --preference-loss-mode dpo --dpo-beta 1.0 \
    --state-margin-weight 0.0 --gain-weight 1.0 --target-margin 0.0 --target-gain 0.0 \
    --apply-mode "$apply_mode" --causal-train-mask --score-mode avglogp --option-selection-mode model_max \
    --alpha-sweep "$ALPHA_SWEEP" \
    "${extra[@]}" \
    --device cuda --torch-dtype bfloat16 \
    --out-dir "$out_dir" >"$out_dir/train.log" 2>&1
  stage "train_$name" done
}

ensure_mlp_payloads() {
  if [[ -f "$MLP_POS_PAYLOAD" && -f "$MLP_NEG_PAYLOAD" ]]; then
    log "reuse mlp payloads: $MLP_POS_PAYLOAD ; $MLP_NEG_PAYLOAD"
    return
  fi
  if [[ "$TRAIN_MLP_IF_MISSING" != "1" ]]; then
    echo "Missing MLP payloads and TRAIN_MLP_IF_MISSING=0" >&2
    exit 1
  fi
  run_train_component mlp_positive_zero_train_512 "$POS_MLP_COMPONENTS" prefill 1
  run_train_component mlp_negative_zero_train_512 "$NEG_MLP_COMPONENTS" prefill 1
  MLP_POS_PAYLOAD="$OUT/train/mlp_positive_zero_train_512/fixed_actuator.pt"
  MLP_NEG_PAYLOAD="$OUT/train/mlp_negative_zero_train_512/fixed_actuator.pt"
}

run_component_zero_base() {
  local name="$1"
  local components="$2"
  local apply_mode="$3"
  local out_dir="$OUT/evaluation/val_select/$name"
  stage "eval_$name" running
  mkdir -p "$out_dir"
  "$PY" -m screscomp.cli.run_imdb_sentiment_actuator_generation \
    --model "$MODEL" \
    --prompts-jsonl "$ROOT/eval_env/val/prompts.jsonl" \
    --out-jsonl "$out_dir/generations.jsonl" \
    --control-name "$name" \
    --alpha-sweep 0 \
    --generation-apply-mode "$apply_mode" --component-apply-mode "$apply_mode" \
    --zero-components "$components" --zero-component-apply-mode "$apply_mode" \
    --split "$EVAL_SPLIT" $(eval_row_args) --samples-per-prompt 1 \
    --generation-batch-size "$GEN_BATCH" --max-new-tokens 256 \
    --temperature 1.0 --top-p 1.0 --top-k 50 --seed 42 --same-seed-across-alpha \
    --device cuda --torch-dtype bfloat16 >"$out_dir/generate.log" 2>&1
  score_and_kl "$out_dir"
  stage "eval_$name" done
}

run_component_vector_group() {
  local group="$1"
  local components="$2"
  local apply_mode="$3"
  local actuator_path="$4"
  local base_name="$5"
  local out_dir="$OUT/evaluation/val_select/$group"
  stage "eval_$group" running
  mkdir -p "$out_dir"
  "$PY" -m screscomp.cli.run_imdb_sentiment_actuator_generation \
    --model "$MODEL" \
    --prompts-jsonl "$ROOT/eval_env/val/prompts.jsonl" \
    --out-jsonl "$out_dir/nonzero_generations.jsonl" \
    --actuator "$actuator_path" \
    --control-name "$group" \
    --alpha-sweep "$NONZERO_SWEEP" \
    --generation-apply-mode "$apply_mode" --component-apply-mode "$apply_mode" \
    --zero-components "$components" --zero-component-apply-mode "$apply_mode" \
    --split "$EVAL_SPLIT" $(eval_row_args) --samples-per-prompt 1 \
    --generation-batch-size "$GEN_BATCH" --max-new-tokens 256 \
    --temperature 1.0 --top-p 1.0 --top-k 50 --seed 42 --same-seed-across-alpha \
    --device cuda --torch-dtype bfloat16 >"$out_dir/generate.log" 2>&1
  materialize_group_generations "$OUT/evaluation/val_select/$base_name" "$out_dir/nonzero_generations.jsonl" "$group" "$out_dir/generations.jsonl"
  score_and_kl "$out_dir"
  stage "eval_$group" done
}

run_full_zero_base() {
  local name="$1"
  local mlp_components="$2"
  local attn_components="$3"
  local out_dir="$OUT/evaluation/val_select/$name"
  stage "eval_$name" running
  mkdir -p "$out_dir"
  "$PY" -m screscomp.cli.run_imdb_sentiment_actuator_generation \
    --model "$MODEL" \
    --prompts-jsonl "$ROOT/eval_env/val/prompts.jsonl" \
    --out-jsonl "$out_dir/generations.jsonl" \
    --control-name "$name" \
    --alpha-sweep 0 \
    --generation-apply-mode prefill \
    --zero-component-groups "prefill=$mlp_components;all=$attn_components" \
    --split "$EVAL_SPLIT" $(eval_row_args) --samples-per-prompt 1 \
    --generation-batch-size "$GEN_BATCH" --max-new-tokens 256 \
    --temperature 1.0 --top-p 1.0 --top-k 50 --seed 42 --same-seed-across-alpha \
    --device cuda --torch-dtype bfloat16 >"$out_dir/generate.log" 2>&1
  score_and_kl "$out_dir"
  stage "eval_$name" done
}

run_full_vector_group() {
  local group="$1"
  local mlp_components="$2"
  local attn_components="$3"
  local mlp_payload="$4"
  local attn_payload="$5"
  local base_name="$6"
  local out_dir="$OUT/evaluation/val_select/$group"
  stage "eval_$group" running
  mkdir -p "$out_dir"
  "$PY" -m screscomp.cli.run_imdb_sentiment_actuator_generation \
    --model "$MODEL" \
    --prompts-jsonl "$ROOT/eval_env/val/prompts.jsonl" \
    --out-jsonl "$out_dir/nonzero_generations.jsonl" \
    --component-actuators "mlp=$mlp_payload:prefill;attn=$attn_payload:all" \
    --control-name "$group" \
    --alpha-sweep "$NONZERO_SWEEP" \
    --generation-apply-mode prefill \
    --zero-component-groups "prefill=$mlp_components;all=$attn_components" \
    --split "$EVAL_SPLIT" $(eval_row_args) --samples-per-prompt 1 \
    --generation-batch-size "$GEN_BATCH" --max-new-tokens 256 \
    --temperature 1.0 --top-p 1.0 --top-k 50 --seed 42 --same-seed-across-alpha \
    --device cuda --torch-dtype bfloat16 >"$out_dir/generate.log" 2>&1
  materialize_group_generations "$OUT/evaluation/val_select/$base_name" "$out_dir/nonzero_generations.jsonl" "$group" "$out_dir/generations.jsonl"
  score_and_kl "$out_dir"
  stage "eval_$group" done
}

stage preflight running
"$PY" -m py_compile \
  src/screscomp/cli/cecm_train_fixed_actuator.py \
  src/screscomp/cli/cecm_run_joint_actuator_generation.py \
  src/screscomp/cli/run_imdb_sentiment_actuator_generation.py \
  src/screscomp/cli/compute_imdb_generation_kl.py
test -f "$POS_ATTN_COMPONENTS"
test -f "$NEG_ATTN_COMPONENTS"
test -f "$POS_MLP_COMPONENTS"
test -f "$NEG_MLP_COMPONENTS"

POS_MLP_IDS="$(component_ids "$POS_MLP_COMPONENTS")"
NEG_MLP_IDS="$(component_ids "$NEG_MLP_COMPONENTS")"
POS_ATTN_IDS="$(component_ids "$POS_ATTN_COMPONENTS")"
NEG_ATTN_IDS="$(component_ids "$NEG_ATTN_COMPONENTS")"
ensure_mlp_payloads

cat > "$OUT/experiment_manifest.json" <<EOF
{
  "experiment": "IMDb post-O attention component actuator rerun",
  "source_run": "$ROOT",
  "model": "$MODEL",
  "gpu": "$GPU",
  "train_rows": $TRAIN_ROWS,
  "val_rows": $VAL_ROWS,
  "eval_prompts": "$ROOT/eval_env/val/prompts.jsonl",
  "positive_mlp_components": "$POS_MLP_IDS",
  "negative_mlp_components": "$NEG_MLP_IDS",
  "positive_attention_components": "$POS_ATTN_IDS",
  "negative_attention_components": "$NEG_ATTN_IDS",
  "mlp_positive_payload": "$MLP_POS_PAYLOAD",
  "mlp_negative_payload": "$MLP_NEG_PAYLOAD",
  "attention_component_timing": "post-O residual module output, apply-mode all",
  "full_control_timing": "MLP prefill + post-O attention all",
  "alpha_sweep": "$ALPHA_SWEEP",
  "nonzero_sweep": "$NONZERO_SWEEP"
}
EOF
stage preflight done

run_train_component attn_positive_normal_512 "$POS_ATTN_COMPONENTS" all 0
run_train_component attn_positive_zero_train_512 "$POS_ATTN_COMPONENTS" all 1
run_train_component attn_negative_normal_512 "$NEG_ATTN_COMPONENTS" all 0
run_train_component attn_negative_zero_train_512 "$NEG_ATTN_COMPONENTS" all 1

run_component_zero_base attn_pos_zero_base_512 "$POS_ATTN_IDS" all
run_component_vector_group attn_pos_postO_normal_vec_zeroctx_512 "$POS_ATTN_IDS" all "$OUT/train/attn_positive_normal_512/fixed_actuator.pt" attn_pos_zero_base_512
run_component_vector_group attn_pos_postO_zero_train_vec_zeroctx_512 "$POS_ATTN_IDS" all "$OUT/train/attn_positive_zero_train_512/fixed_actuator.pt" attn_pos_zero_base_512

run_component_zero_base attn_neg_zero_base_512 "$NEG_ATTN_IDS" all
run_component_vector_group attn_neg_postO_normal_vec_zeroctx_512 "$NEG_ATTN_IDS" all "$OUT/train/attn_negative_normal_512/fixed_actuator.pt" attn_neg_zero_base_512
run_component_vector_group attn_neg_postO_zero_train_vec_zeroctx_512 "$NEG_ATTN_IDS" all "$OUT/train/attn_negative_zero_train_512/fixed_actuator.pt" attn_neg_zero_base_512

run_full_zero_base full_pos_postO_zero_base_512 "$POS_MLP_IDS" "$POS_ATTN_IDS"
run_full_vector_group full_pos_postO_normal_vec_zeroctx_512 "$POS_MLP_IDS" "$POS_ATTN_IDS" "$MLP_POS_PAYLOAD" "$OUT/train/attn_positive_normal_512/fixed_actuator.pt" full_pos_postO_zero_base_512
run_full_vector_group full_pos_postO_zero_train_vec_zeroctx_512 "$POS_MLP_IDS" "$POS_ATTN_IDS" "$MLP_POS_PAYLOAD" "$OUT/train/attn_positive_zero_train_512/fixed_actuator.pt" full_pos_postO_zero_base_512

run_full_zero_base full_neg_postO_zero_base_512 "$NEG_MLP_IDS" "$NEG_ATTN_IDS"
run_full_vector_group full_neg_postO_normal_vec_zeroctx_512 "$NEG_MLP_IDS" "$NEG_ATTN_IDS" "$MLP_NEG_PAYLOAD" "$OUT/train/attn_negative_normal_512/fixed_actuator.pt" full_neg_postO_zero_base_512
run_full_vector_group full_neg_postO_zero_train_vec_zeroctx_512 "$NEG_MLP_IDS" "$NEG_ATTN_IDS" "$MLP_NEG_PAYLOAD" "$OUT/train/attn_negative_zero_train_512/fixed_actuator.pt" full_neg_postO_zero_base_512

stage summarize running
"$PY" -m screscomp.cli.summarize_imdb_sentiment_matrix \
  --root "$OUT/evaluation/val_select" \
  --out-csv "$OUT/evaluation/val_select/matrix_score_summary.csv" >"$OUT/evaluation/val_select/summarize.log" 2>&1
stage summarize done
stage complete done
log "done out=$OUT"
