#!/usr/bin/env bash
set -euo pipefail

# Locked IMDb main-table protocol. Scientific settings are intentionally not configurable.

cd LOCAL_HOME/RPEC/projects/screscomp

GPU="${GPU:-0}"
MODEL="edbeeching/gpt2-large-imdb"
EVENT="imdb_positive_sentiment"
SEED=42
ROOT="runs/imdb_directional_main_table_locked_$(date +%Y%m%d_%H%M%S)"
COMPONENT_SOURCE="runs/imdb_dpo_prompt_positive_rollout_scan_20260605_115624/discovery/directional_top4_selected"
COMPONENT_BASELINE_CSV="runs/imdb_dpo_prompt_positive_rollout_scan_20260605_115624/discovery/component_scan/component_delta_samples.csv"
HEAD_ROLLOUT_PROMPTS="runs/imdb_dpo_prompt_positive_rollout_scan_20260605_115624/env/prompts.jsonl"
HEAD_SOURCE="runs/imdb_directional_main_table_locked_20260606_214241/train/head_scan_pool"
PAIR_SOURCE_TRAIN_PROMPTS=768
PAIR_SOURCE_VAL_PROMPTS=192
PAIR_COMPLETIONS_PER_PREFIX=4
PAIR_TRAIN_TARGET=512
PAIR_VAL_TARGET=128
PAIR_MAX_NEW_TOKENS=128
PAIR_MIN_SCORE_MARGIN=0.0
PAIR_SELECTION_MODE="top_bottom"
HEAD_SCAN_SAMPLES=300
HEAD_TOPK=4
TRAIN_EPOCHS=2
MLP_APPLY_MODE="prefill"
HEAD_APPLY_MODE="all"
HEAD_SCAN_APPLY_MODE="all"
TRAIN_ALPHA_SWEEP="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"

VAL_PROMPTS=1024
TEST_PROMPTS=2048
GENERATION_BATCH_SIZE=16
KL_BATCH_SIZE=16

TORCH_DTYPE="bfloat16"
DEVICE="cuda"
SCORER_MODEL="siebert/sentiment-roberta-large-english"
SCORER_DEVICE="${SCORER_DEVICE:-$GPU}"
SCORE_BATCH_SIZE=16

COMPONENT_DIR="$ROOT/components"
TRAIN_ROOT="$ROOT/train"
PAIRS_DIR="$ROOT/pairs"
PAIRS_CSV="$PAIRS_DIR/pairs.csv"
EVAL_ENV_ROOT="$ROOT/eval_env"
EVAL_ROOT="$ROOT/evaluation"
AUDIT_ROOT="$ROOT/audit"
STATUS="$ROOT/status.tsv"
COST="$ROOT/cost.tsv"
MASTER_LOG="$ROOT/master.log"
CURRENT_STAGE="init"
CURRENT_STAGE_START="$(date +%s)"

mkdir -p "$ROOT" "$COMPONENT_DIR" "$TRAIN_ROOT" "$EVAL_ENV_ROOT" "$AUDIT_ROOT"
if [[ -f LOCAL_HOME/anaconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source LOCAL_HOME/anaconda3/etc/profile.d/conda.sh
  conda activate screscomp
fi
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

printf "time\tstage\tstatus\n" > "$STATUS"
printf "stage\tstart_epoch\tend_epoch\tseconds\n" > "$COST"

stage_start() {
  CURRENT_STAGE="$1"
  CURRENT_STAGE_START="$(date +%s)"
  printf "%s\t%s\trunning\n" "$(date -Is)" "$CURRENT_STAGE" >> "$STATUS"
  echo "[$(date -Is)] stage=$CURRENT_STAGE running" | tee -a "$MASTER_LOG"
}

stage_end() {
  local end
  end="$(date +%s)"
  printf "%s\t%s\tdone\n" "$(date -Is)" "$CURRENT_STAGE" >> "$STATUS"
  printf "%s\t%s\t%s\t%s\n" "$CURRENT_STAGE" "$CURRENT_STAGE_START" "$end" "$((end - CURRENT_STAGE_START))" >> "$COST"
  echo "[$(date -Is)] stage=$CURRENT_STAGE done seconds=$((end - CURRENT_STAGE_START))" | tee -a "$MASTER_LOG"
}

fail() {
  printf "%s\t%s\tfailed\n" "$(date -Is)" "$CURRENT_STAGE" >> "$STATUS"
  echo "[$(date -Is)] FAIL: $*" | tee -a "$MASTER_LOG" >&2
  exit 2
}

on_error() {
  local rc="$?"
  trap - ERR
  printf "%s\t%s\tfailed\n" "$(date -Is)" "$CURRENT_STAGE" >> "$STATUS"
  echo "[$(date -Is)] stage=$CURRENT_STAGE failed rc=$rc" | tee -a "$MASTER_LOG" >&2
  exit "$rc"
}
trap on_error ERR

count_pairs() {
  local split="$1"
  python - "$PAIRS_CSV" "$split" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
split = sys.argv[2]
if not path.exists():
    print(0)
    raise SystemExit(0)
with path.open("r", encoding="utf-8-sig", newline="") as f:
    print(sum(1 for row in csv.DictReader(f) if row.get("split") == split and row.get("admitted") == "1"))
PY
}

count_csv_rows() {
  python - "$1" <<'PY'
import csv
import sys

with open(sys.argv[1], "r", encoding="utf-8-sig", newline="") as f:
    print(sum(1 for _ in csv.DictReader(f)))
PY
}

extract_layers() {
  python - "$@" <<'PY'
import csv
import sys

seen = set()
layers = []
for raw_path in sys.argv[1:]:
    with open(raw_path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            layer = str(row.get("layer_idx", "")).strip()
            if layer and layer not in seen:
                seen.add(layer)
                layers.append(layer)
print(",".join(layers))
PY
}

write_protocol() {
  python - "$ROOT/protocol.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "protocol": "imdb_directional_main_table_locked_v1",
    "model": "edbeeching/gpt2-large-imdb",
    "event": "imdb_positive_sentiment",
    "seed": 42,
    "component_discovery": {
        "reuse": "runs/imdb_dpo_prompt_positive_rollout_scan_20260605_115624/discovery/directional_top4_selected",
        "rollout_samples_per_component": 300,
        "apply_mode": "all",
        "groups": ["mlp_positive", "mlp_negative", "attn_positive", "attn_negative"],
        "topk_per_group": 4,
    },
    "head_localization": {
        "reuse": "runs/imdb_directional_main_table_locked_20260606_214241/train/head_scan_pool",
        "scope": "union of reused positive and negative attention-layer pools",
        "rollout_prompts": "runs/imdb_dpo_prompt_positive_rollout_scan_20260605_115624/env/prompts.jsonl",
        "rollout_samples": 300,
        "operator": "zero one head during open generation",
        "selection": "healthy_pos_mass / healthy_neg_mass",
        "topk_boost": 4,
        "topk_suppress": 4,
        "apply_mode": "all",
        "rescan": False,
    },
    "pair_pool": {
        "source_split": "train",
        "source_train_prompts": 768,
        "source_val_prompts": 192,
        "final_train_pairs": 512,
        "final_val_pairs": 128,
        "completions_per_prompt": 4,
        "max_new_tokens": 128,
        "min_score_margin": 0.0,
        "pair_selection_mode": "top_bottom",
        "generated": True,
    },
    "training": {
        "train_pairs": 512,
        "val_pairs": 128,
        "epochs": 2,
        "loss": "reference-anchored DPO over length-normalized continuation conditional log-probability",
        "dpo_beta": 1.0,
        "mlp_apply_mode": "prefill",
        "head_apply_mode": "all",
        "causal_train_mask": True,
    },
    "evaluation": {
        "val_prompts": 1024,
        "test_prompts": 2048,
        "alpha_curve": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "main_table_alpha": 0.3,
        "main_table_alpha_policy": "pre-registered from prior low-KL health audit; not selected on this test",
        "shared_base": "one generation, reward score, and KL computation per split; reused identically across all six groups",
        "mean_sequence_kl": True,
        "max_new_tokens": 256,
        "generation_batch_size": 16,
        "reward_batch_size": 16,
        "kl_batch_size": 16,
    },
}
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

write_protocol
python - "$ROOT/code_manifest.json" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

out_path = Path(sys.argv[1])
tracked = [
    "scripts/run_imdb_directional_main_table_locked.sh",
    "scripts/run_imdb_directional_val_select_then_test.sh",
    "src/screscomp/cli/audit_imdb_locked_experiment.py",
    "src/screscomp/cli/scan_imdb_rollout_attention_heads.py",
    "src/screscomp/cli/select_imdb_rollout_heads.py",
    "src/screscomp/cli/prepare_imdb_sentiment_env.py",
    "src/screscomp/cli/prepare_imdb_sentiment_pairs.py",
    "src/screscomp/cli/cecm_train_fixed_actuator.py",
    "src/screscomp/cli/cecm_train_attention_head_actuator.py",
    "src/screscomp/cli/run_imdb_sentiment_actuator_generation.py",
    "src/screscomp/cli/score_imdb_sentiment_generations.py",
    "src/screscomp/cli/compute_imdb_generation_kl.py",
    "src/screscomp/cli/summarize_imdb_sentiment_matrix.py",
]
hashes = {}
for raw in tracked:
    path = Path(raw)
    hashes[raw] = hashlib.sha256(path.read_bytes()).hexdigest()
try:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
except Exception:
    commit = ""
out_path.write_text(json.dumps({"git_commit": commit, "sha256": hashes}, indent=2) + "\n", encoding="utf-8")
PY

stage_start "preflight_components"
for file in \
  selection_manifest.json \
  mlp_positive_components.csv \
  mlp_negative_components.csv \
  attn_positive_components.csv \
  attn_negative_components.csv; do
  test -f "$COMPONENT_SOURCE/$file" || fail "missing reused component artifact: $COMPONENT_SOURCE/$file"
  cp "$COMPONENT_SOURCE/$file" "$COMPONENT_DIR/$file"
done
CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.audit_imdb_locked_experiment \
  --stage components \
  --root "$AUDIT_ROOT" \
  --component-dir "$COMPONENT_SOURCE" \
  | tee -a "$MASTER_LOG"
stage_end

stage_start "generate_and_build_pairs"
mkdir -p "$ROOT/start_policy" "$PAIRS_DIR"
# Extract disjoint train/val prompt pools for pair generation.
CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.prepare_imdb_sentiment_env \
  --hf-dataset stanfordnlp/imdb \
  --hf-split train \
  --out-dir "$ROOT/pair_env" \
  --event "$EVENT" \
  --train-rows "$PAIR_SOURCE_TRAIN_PROMPTS" \
  --val-rows "$PAIR_SOURCE_VAL_PROMPTS" \
  --eval-rows 0 \
  --seed "$SEED" \
  --shuffle \
  | tee -a "$MASTER_LOG"
CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.generate_imdb_sentiment_completions \
  --model "$MODEL" \
  --prompts-jsonl "$ROOT/pair_env/prompts.jsonl" \
  --out-jsonl "$ROOT/start_policy/generations.jsonl" \
  --split all \
  --completions-per-prefix "$PAIR_COMPLETIONS_PER_PREFIX" \
  --generation-batch-size "$GENERATION_BATCH_SIZE" \
  --max-new-tokens "$PAIR_MAX_NEW_TOKENS" \
  --temperature 1.0 \
  --top-p 1.0 \
  --top-k 50 \
  --seed "$SEED" \
  --torch-dtype "$TORCH_DTYPE" \
  --device "$DEVICE" \
  | tee -a "$MASTER_LOG"
CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.score_imdb_sentiment_generations \
  --input "$ROOT/start_policy/generations.jsonl" \
  --out-jsonl "$ROOT/start_policy/scored_generations.jsonl" \
  --out-csv "$ROOT/start_policy/scored_generations.csv" \
  --scorer-model "$SCORER_MODEL" \
  --score-text completion \
  --batch-size "$SCORE_BATCH_SIZE" \
  --device "$SCORER_DEVICE" \
  | tee -a "$MASTER_LOG"
python -m screscomp.cli.prepare_imdb_sentiment_pairs \
  --input "$ROOT/start_policy/scored_generations.jsonl" \
  --out-dir "$PAIRS_DIR" \
  --event "$EVENT" \
  --min-score-margin "$PAIR_MIN_SCORE_MARGIN" \
  --pair-selection-mode "$PAIR_SELECTION_MODE" \
  --max-prompts-per-split 0 \
  --target-train-pairs "$PAIR_TRAIN_TARGET" \
  --target-val-pairs "$PAIR_VAL_TARGET" \
  | tee -a "$MASTER_LOG"
test "$(count_pairs train)" -eq "$PAIR_TRAIN_TARGET" \
  || fail "pair build produced train=$(count_pairs train), expected $PAIR_TRAIN_TARGET"
test "$(count_pairs val)" -eq "$PAIR_VAL_TARGET" \
  || fail "pair build produced val=$(count_pairs val), expected $PAIR_VAL_TARGET"
echo "[$(date -Is)] pairs: train=$(count_pairs train) val=$(count_pairs val)" | tee -a "$MASTER_LOG"
stage_end

stage_start "reuse_head_localization"
test -d "$HEAD_SOURCE" || fail "missing reused head localization directory: $HEAD_SOURCE"
mkdir -p "$TRAIN_ROOT" "$TRAIN_ROOT/head_groups"
cp -R "$HEAD_SOURCE" "$TRAIN_ROOT/head_scan_pool"
test "$(count_csv_rows "$TRAIN_ROOT/head_scan_pool/selected_boost_heads.csv")" -ge 1 \
  || fail "reused head localization did not produce boost heads"
test "$(count_csv_rows "$TRAIN_ROOT/head_scan_pool/selected_suppress_heads.csv")" -ge 1 \
  || fail "reused head localization did not produce suppress heads"
test "$(count_csv_rows "$TRAIN_ROOT/head_scan_pool/selected_boost_heads.csv")" -eq "$HEAD_TOPK" \
  || fail "boost head count mismatch"
test "$(count_csv_rows "$TRAIN_ROOT/head_scan_pool/selected_suppress_heads.csv")" -eq "$HEAD_TOPK" \
  || fail "suppress head count mismatch"
cp "$TRAIN_ROOT/head_scan_pool/selected_boost_heads.csv" "$TRAIN_ROOT/head_groups/head_positive_heads.csv"
cp "$TRAIN_ROOT/head_scan_pool/selected_boost_heads.txt" "$TRAIN_ROOT/head_groups/head_positive_heads.txt"
cp "$TRAIN_ROOT/head_scan_pool/selected_suppress_heads.csv" "$TRAIN_ROOT/head_groups/head_negative_heads.csv"
cp "$TRAIN_ROOT/head_scan_pool/selected_suppress_heads.txt" "$TRAIN_ROOT/head_groups/head_negative_heads.txt"
stage_end

train_mlp() {
  local components="$1"
  local out_dir="$2"
  local cost_name="$3"
  local start end
  start="$(date +%s)"
  mkdir -p "$out_dir"
  CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.cecm_train_fixed_actuator \
    --model "$MODEL" \
    --pairs-csv "$PAIRS_CSV" \
    --components-csv "$components" \
    --event "$EVENT" \
    --train-split train \
    --val-split val \
    --max-train-rows "$PAIR_TRAIN_TARGET" \
    --max-val-rows "$PAIR_VAL_TARGET" \
    --epochs "$TRAIN_EPOCHS" \
    --train-batch-size 16 \
    --lr 0.05 \
    --lambda-norm 1e-4 \
    --alpha-train 1.0 \
    --preference-loss-mode dpo \
    --dpo-beta 1.0 \
    --state-margin-weight 0.0 \
    --gain-weight 1.0 \
    --target-margin 0.0 \
    --target-gain 0.0 \
    --apply-mode "$MLP_APPLY_MODE" \
    --causal-train-mask \
    --score-mode avglogp \
    --alpha-sweep "$TRAIN_ALPHA_SWEEP" \
    --max-aliases-per-side 1 \
    --empty-cache-every 25 \
    --torch-dtype "$TORCH_DTYPE" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --out-dir "$out_dir" \
    > "$out_dir/train.log" 2>&1
  end="$(date +%s)"
  printf "%s\t%s\t%s\t%s\n" "$cost_name" "$start" "$end" "$((end - start))" >> "$COST"
}

train_head() {
  local heads_file="$1"
  local out_dir="$2"
  local cost_name="$3"
  local heads
  local start end
  start="$(date +%s)"
  heads="$(tr -d '\r\n' < "$heads_file")"
  if [ -z "$heads" ]; then
    echo "WARNING: empty selected head file: $heads_file, skipping" | tee -a "$MASTER_LOG"
    return 0
  fi
  mkdir -p "$out_dir"
  CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.cecm_train_attention_head_actuator \
    --model "$MODEL" \
    --pairs-csv "$PAIRS_CSV" \
    --event "$EVENT" \
    --heads "$heads" \
    --train-split train \
    --val-split val \
    --max-train-rows "$PAIR_TRAIN_TARGET" \
    --max-val-rows "$PAIR_VAL_TARGET" \
    --epochs "$TRAIN_EPOCHS" \
    --train-batch-size 16 \
    --lr 0.05 \
    --lambda-norm 1e-4 \
    --alpha-train 1.0 \
    --preference-loss-mode dpo \
    --dpo-beta 1.0 \
    --state-margin-weight 0.0 \
    --gain-weight 1.0 \
    --target-margin 0.0 \
    --target-gain 0.0 \
    --apply-mode "$HEAD_APPLY_MODE" \
    --causal-train-mask \
    --score-mode avglogp \
    --alpha-sweep "$TRAIN_ALPHA_SWEEP" \
    --max-aliases-per-side 1 \
    --empty-cache-every 25 \
    --torch-dtype "$TORCH_DTYPE" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --out-dir "$out_dir" \
    > "$out_dir/train.log" 2>&1
  end="$(date +%s)"
  printf "%s\t%s\t%s\t%s\n" "$cost_name" "$start" "$end" "$((end - start))" >> "$COST"
}

stage_start "train_four_actuators"
train_mlp "$COMPONENT_DIR/mlp_positive_components.csv" "$TRAIN_ROOT/train_mlp_positive" "train_mlp_positive"
train_mlp "$COMPONENT_DIR/mlp_negative_components.csv" "$TRAIN_ROOT/train_mlp_negative" "train_mlp_negative"
train_head "$TRAIN_ROOT/head_groups/head_positive_heads.txt" "$TRAIN_ROOT/train_head_positive" "train_head_positive"
train_head "$TRAIN_ROOT/head_groups/head_negative_heads.txt" "$TRAIN_ROOT/train_head_negative" "train_head_negative"
python -m screscomp.cli.audit_imdb_locked_experiment \
  --stage training \
  --root "$AUDIT_ROOT" \
  --pairs-csv "$PAIRS_CSV" \
  --train-root "$TRAIN_ROOT" \
  | tee -a "$MASTER_LOG"
stage_end

stage_start "prepare_eval_environments"
mkdir -p "$EVAL_ENV_ROOT/val" "$EVAL_ENV_ROOT/test"
python -m screscomp.cli.prepare_imdb_sentiment_env \
  --hf-dataset stanfordnlp/imdb \
  --hf-split train \
  --start 15000 \
  --max-source-rows 10000 \
  --out-dir "$EVAL_ENV_ROOT/val" \
  --event "$EVENT" \
  --source-dataset stanfordnlp/imdb \
  --train-rows 0 \
  --val-rows 0 \
  --eval-rows "$VAL_PROMPTS" \
  --seed "$SEED" \
  --shuffle \
  --prefix-mode tokenizer \
  --tokenizer "$MODEL" \
  --prefix-token-min 2 \
  --prefix-token-max 8 \
  --prompt-template "{prefix}" \
  --target-sentiment positive \
  --completions-per-prefix 1 \
  --reference-model "$MODEL" \
  > "$EVAL_ENV_ROOT/val/prepare.log" 2>&1
python -m screscomp.cli.prepare_imdb_sentiment_env \
  --hf-dataset stanfordnlp/imdb \
  --hf-split test \
  --start 0 \
  --max-source-rows 25000 \
  --out-dir "$EVAL_ENV_ROOT/test" \
  --event "$EVENT" \
  --source-dataset stanfordnlp/imdb \
  --train-rows 0 \
  --val-rows 0 \
  --eval-rows "$TEST_PROMPTS" \
  --seed "$SEED" \
  --shuffle \
  --prefix-mode tokenizer \
  --tokenizer "$MODEL" \
  --prefix-token-min 2 \
  --prefix-token-max 8 \
  --prompt-template "{prefix}" \
  --target-sentiment positive \
  --completions-per-prefix 1 \
  --reference-model "$MODEL" \
  > "$EVAL_ENV_ROOT/test/prepare.log" 2>&1
stage_end

stage_start "evaluate_shared_base_and_six_groups"
TRAIN_ROOT="$TRAIN_ROOT" \
OUT_ROOT="$EVAL_ROOT" \
VAL_ENV_DIR="$EVAL_ENV_ROOT/val" \
TEST_ENV_DIR="$EVAL_ENV_ROOT/test" \
GENERATION_BATCH_SIZE="$GENERATION_BATCH_SIZE" \
SCORE_BATCH_SIZE="$SCORE_BATCH_SIZE" \
KL_BATCH_SIZE="$KL_BATCH_SIZE" \
RESUME=1 \
bash scripts/run_imdb_directional_val_select_then_test.sh
stage_end

stage_start "audit_and_build_main_table"
python -m screscomp.cli.audit_imdb_locked_experiment \
  --stage eval \
  --root "$AUDIT_ROOT" \
  --eval-root "$EVAL_ROOT/val_select" \
  --expected-prompts "$VAL_PROMPTS" \
  | tee -a "$MASTER_LOG"
python -m screscomp.cli.audit_imdb_locked_experiment \
  --stage eval \
  --root "$AUDIT_ROOT" \
  --eval-root "$EVAL_ROOT/test_eval" \
  --expected-prompts "$TEST_PROMPTS" \
  | tee -a "$MASTER_LOG"
python -m screscomp.cli.audit_imdb_locked_experiment \
  --stage final \
  --root "$AUDIT_ROOT" \
  --eval-root "$EVAL_ROOT/test_eval" \
  --expected-prompts "$TEST_PROMPTS" \
  | tee -a "$MASTER_LOG"
stage_end

printf "%s\tall\tdone\n" "$(date -Is)" >> "$STATUS"
echo "[$(date -Is)] locked IMDb main-table experiment complete root=$ROOT table=$AUDIT_ROOT/main_table.csv" | tee -a "$MASTER_LOG"
