#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

GPU="${GPU:-3}"
MODEL="${MODEL:-LOCAL_HOME/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct}"
DATA="${DATA:-data_ckplug/confiqa_large_ck_default_test5000/02_prompts/confiqa_open_rows.jsonl}"
CK_CORE="${CK_CORE:-external/CK-PLUG-core}"
CK_TFM="${CK_TFM:-external/CK-PLUG-core/transformers-4.49}"

N_SCAN="${N_SCAN:-70}"
N_ALPHA="${N_ALPHA:-200}"
N_MAIN="${N_MAIN:-1000}"
K_COMPONENTS="${K_COMPONENTS:-4}"

OUT_ROOT="${OUT_ROOT:-data_ckplug/cecm_source_modelprior_main1000_base_v0}"
PRIOR_OUT="$OUT_ROOT/00_model_prior"
PAIR_OUT="$OUT_ROOT/01_pairs"
SCAN_OUT="$OUT_ROOT/02_component_scan"
AUDIT_OUT="$OUT_ROOT/03_confidence_audit"
GROUP_OUT="$OUT_ROOT/04_selected_components"
ALPHA_OUT="$OUT_ROOT/05_alpha_calibration"
BASE_OUT="$OUT_ROOT/06_baselines"
MAIN_OUT="$OUT_ROOT/07_main_eval"

mkdir -p "$PRIOR_OUT" "$PAIR_OUT" "$SCAN_OUT" "$AUDIT_OUT" "$GROUP_OUT" "$ALPHA_OUT" "$BASE_OUT" "$MAIN_OUT"

test -d "$MODEL"
test -f "$DATA"
test -f "$CK_CORE/ck.py"
test -d "$CK_TFM"
test -f .venv_ckplug/bin/activate

echo "[setup] OUT_ROOT=$OUT_ROOT GPU=$GPU N_SCAN=$N_SCAN N_ALPHA=$N_ALPHA N_MAIN=$N_MAIN"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  deactivate || true
fi
source LOCAL_HOME/anaconda3/etc/profile.d/conda.sh
conda activate screscomp
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

python -m screscomp.cli.cecm_discover_model_prior --help >/dev/null

echo "[1/8] discover model prior on all open rows"
CUDA_VISIBLE_DEVICES=$GPU python -m screscomp.cli.cecm_discover_model_prior \
  --input-jsonl "$DATA" \
  --out-jsonl "$PRIOR_OUT/open_rows_with_model_prior.jsonl" \
  --out-summary-csv "$PRIOR_OUT/summary.csv" \
  --out-manifest-json "$PRIOR_OUT/manifest.json" \
  --model "$MODEL" \
  --prompt-key base_no_rag \
  --max-new-tokens 16 \
  --stop-strings "Q:" \
  --torch-dtype bfloat16 \
  --device cuda

echo "[2/8] build strict model-prior source pairs"
python -m screscomp.cli.cecm_build_preference_pairs \
  --input-jsonl "$PRIOR_OUT/open_rows_with_model_prior.jsonl" \
  --out-dir "$PAIR_OUT" \
  --event source_context_over_prior \
  --prompt-key base_rag \
  --prior-source model_prior

echo "[3/8] few-shot component scan: 70+70 train"
run_scan () {
  local score_mode="$1"
  local start="$2"
  local tag="$3"
  local out="$SCAN_OUT/${score_mode}_${tag}"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES=$GPU python -m screscomp.cli.cecm_scan_component_contributions \
    --model "$MODEL" \
    --pairs-csv "$PAIR_OUT/pairs.csv" \
    --event source_context_over_prior \
    --split train \
    --start "$start" \
    --max-rows "$N_SCAN" \
    --component-types attn,mlp \
    --apply-mode decision_tokens \
    --score-mode "$score_mode" \
    --max-aliases-per-side 0 \
    --min-abs-delta 0.0 \
    --min-sign-consistency 0.55 \
    --max-components-per-direction 12 \
    --torch-dtype bfloat16 \
    --device cuda \
    --out-dir "$out"
}

run_scan avglogp 0 A
run_scan avglogp "$N_SCAN" B
run_scan answer_rest_margin 0 A
run_scan answer_rest_margin "$N_SCAN" B

echo "[4/8] confidence audit for few-shot selection"
python -m screscomp.cli.cecm_audit_component_confidence \
  --avglogp-a "$SCAN_OUT/avglogp_A/component_screen.csv" \
  --avglogp-b "$SCAN_OUT/avglogp_B/component_screen.csv" \
  --margin-a "$SCAN_OUT/answer_rest_margin_A/component_screen.csv" \
  --margin-b "$SCAN_OUT/answer_rest_margin_B/component_screen.csv" \
  --alpha 0.05 \
  --tau-delta 0.10 \
  --top-k 8 \
  --bootstrap 1000 \
  --permutations 1000 \
  --seed 42 \
  --out-dir "$AUDIT_OUT"

echo "[5/8] select components"
AUDIT_OUT="$AUDIT_OUT" GROUP_OUT="$GROUP_OUT" K_COMPONENTS="$K_COMPONENTS" python - <<'PY'
import csv
import json
import os
from pathlib import Path

audit = Path(os.environ["AUDIT_OUT"])
group = Path(os.environ["GROUP_OUT"])
k = int(os.environ["K_COMPONENTS"])
group.mkdir(parents=True, exist_ok=True)

rows = list(csv.DictReader((audit / "component_confidence.csv").open("r", encoding="utf-8", newline="")))

def margin_delta(r):
    return float(r.get("answer_rest_margin_mean_delta_over_splits") or 0)

# Main selection is intentionally simple: signed top-K causal contribution to the
# competitive margin. Confidence, split stability, CI, and score agreement remain
# audit fields, not selection gates.
pos = [r for r in rows if margin_delta(r) > 0]
neg = [r for r in rows if margin_delta(r) < 0]
pos.sort(key=margin_delta, reverse=True)
neg.sort(key=margin_delta)

if not pos or not neg:
    raise SystemExit("Cannot select both positive and negative components.")

ctx = ",".join(r["component_id"] for r in pos[:k])
pri = ",".join(r["component_id"] for r in neg[:k])

(group / "selected.env").write_text(
    f"CONTEXT_COMPONENTS={ctx}\nPRIOR_COMPONENTS={pri}\n",
    encoding="utf-8",
)
(group / "selected_components.json").write_text(json.dumps({
    "context_components": [r["component_id"] for r in pos[:k]],
    "prior_components": [r["component_id"] for r in neg[:k]],
    "selection_rule": "simple signed top-K by answer_rest_margin_mean_delta_over_splits",
    "audit_policy": (
        "confidence_level, robust_status, split status, sign consistency, CI, "
        "top-k overlap, and score-operator agreement are reported as audits, "
        "not used as selection gates"
    ),
    "context_component_deltas": {
        r["component_id"]: margin_delta(r)
        for r in pos[:k]
    },
    "prior_component_deltas": {
        r["component_id"]: margin_delta(r)
        for r in neg[:k]
    },
}, ensure_ascii=False, indent=2), encoding="utf-8")

print("CONTEXT_COMPONENTS", ctx)
print("PRIOR_COMPONENTS", pri)
PY

source "$GROUP_OUT/selected.env"

echo "[6/8] prompt-delta alpha calibration on val 200"
CUDA_VISIBLE_DEVICES=$GPU python -m screscomp.cli.cecm_run_source_prompt_delta_control \
  --model "$MODEL" \
  --vector-open-rows "$PRIOR_OUT/open_rows_with_model_prior.jsonl" \
  --eval-open-rows "$PRIOR_OUT/open_rows_with_model_prior.jsonl" \
  --vector-pairs-csv "$PAIR_OUT/pairs.csv" \
  --context-prompt-key base_rag \
  --prior-prompt-key base_no_rag \
  --prior-source model_prior \
  --vector-split train \
  --vector-start 0 \
  --vector-max-rows "$((2 * N_SCAN))" \
  --split val \
  --start 0 \
  --max-rows "$N_ALPHA" \
  --controls force_context,force_prior \
  --components "$CONTEXT_COMPONENTS" \
  --random-trials 1 \
  --random-baselines random_type_matched \
  --alpha-sweep 0,0.125,0.25,0.5,0.75,1.0,1.25,1.5,2.0 \
  --generation-apply-mode prefill \
  --max-new-tokens 64 \
  --torch-dtype bfloat16 \
  --device cuda \
  --out-dir "$ALPHA_OUT"

ALPHA_OUT="$ALPHA_OUT" GROUP_OUT="$GROUP_OUT" python - <<'PY'
import csv
import json
import os
from pathlib import Path

alpha_out = Path(os.environ["ALPHA_OUT"])
group = Path(os.environ["GROUP_OUT"])
rows = list(csv.DictReader((alpha_out / "generation_summary.csv").open("r", encoding="utf-8", newline="")))

candidates = []
for r in rows:
    if r["baseline_kind"] != "cecm_prompt_delta":
        continue
    if r["control_name"] != "force_context":
        continue
    if r["env_kind"] != "context":
        continue
    alpha = float(r["alpha"])
    score = float(r["context_only_rate"]) - float(r["prior_only_rate"]) - 0.5 * float(r["neither_rate"])
    candidates.append((score, alpha, r))

if not candidates:
    raise SystemExit("No alpha candidates found.")

candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
score, alpha, row = candidates[0]
alpha_text = f"{alpha:g}"
(group / "selected_alpha.env").write_text(f"ALPHA_STAR={alpha_text}\n", encoding="utf-8")
(group / "selected_alpha.json").write_text(json.dumps({
    "alpha_star": alpha_text,
    "selection_metric": "context_only - prior_only - 0.5*neither on val alpha calibration",
    "selected_row": row,
}, ensure_ascii=False, indent=2), encoding="utf-8")
print("ALPHA_STAR", alpha_text)
PY

source "$GROUP_OUT/selected_alpha.env"

echo "[7/8] CK-style baselines on heldout val 1000"
source .venv_ckplug/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

run_ck_base () {
  local mode="$1"
  local dir="$BASE_OUT/$mode"
  mkdir -p "$dir"
  CUDA_VISIBLE_DEVICES=$GPU python -m screscomp.cli.score_ckplug_official \
    --eval_jsonl "$DATA" \
    --ck_core_dir "$CK_CORE" \
    --ck_transformers_dir "$CK_TFM" \
    --model "$MODEL" \
    --out_generations_jsonl "$dir/generations.jsonl" \
    --out_summary_csv "$dir/summary.csv" \
    --out_manifest_csv "$dir/manifest.csv" \
    --device cuda \
    --mode "$mode" \
    --prompt_family shared_strong \
    --alpha 0.5 \
    --max_new_tokens 64 \
    --stop_strings "Q:" \
    --split val \
    --start 0 \
    --max_eval_rows "$N_MAIN" \
    --val_mod 5
}

run_ck_base base_no_rag
run_ck_base base_rag
run_ck_base ck

echo "[8/8] frozen prompt-delta CECM + random main eval on heldout val 1000"
deactivate || true
source LOCAL_HOME/anaconda3/etc/profile.d/conda.sh
conda activate screscomp
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
source "$GROUP_OUT/selected.env"
source "$GROUP_OUT/selected_alpha.env"

CUDA_VISIBLE_DEVICES=$GPU python -m screscomp.cli.cecm_run_source_prompt_delta_control \
  --model "$MODEL" \
  --vector-open-rows "$PRIOR_OUT/open_rows_with_model_prior.jsonl" \
  --eval-open-rows "$PRIOR_OUT/open_rows_with_model_prior.jsonl" \
  --vector-pairs-csv "$PAIR_OUT/pairs.csv" \
  --context-prompt-key base_rag \
  --prior-prompt-key base_no_rag \
  --prior-source model_prior \
  --vector-split train \
  --vector-start 0 \
  --vector-max-rows "$((2 * N_SCAN))" \
  --split val \
  --start 0 \
  --max-rows "$N_MAIN" \
  --controls force_context,force_prior \
  --components "$CONTEXT_COMPONENTS" \
  --random-trials 1 \
  --random-baselines random_type_matched \
  --alpha-sweep "0,$ALPHA_STAR" \
  --generation-apply-mode prefill \
  --max-new-tokens 64 \
  --torch-dtype bfloat16 \
  --device cuda \
  --ckplug-generations "$BASE_OUT/ck/generations.jsonl" \
  --ckplug-method-name ckplug_official \
  --ckplug-env-kind context \
  --out-dir "$MAIN_OUT"

echo "[done]"
echo "prior summary:      $PRIOR_OUT/summary.csv"
echo "pair summary:       $PAIR_OUT/pair_admission_summary.csv"
echo "component audit:    $AUDIT_OUT/component_confidence.csv"
echo "selected comps:     $GROUP_OUT/selected_components.json"
echo "selected alpha:     $GROUP_OUT/selected_alpha.json"
echo "base_no_rag:        $BASE_OUT/base_no_rag/summary.csv"
echo "base_rag:           $BASE_OUT/base_rag/summary.csv"
echo "ck:                 $BASE_OUT/ck/summary.csv"
echo "main cecm table:    $MAIN_OUT/generation_summary.csv"
