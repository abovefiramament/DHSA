#!/usr/bin/env bash
set -euo pipefail

# Discovery-only sample-size sweep for ConFiQA CAST.
# It reuses prepared pairs from an existing run and tests how few train pairs are
# enough to recover the same component/head pool. No actuator training or
# generation is performed.

cd "$(dirname "${BASH_SOURCE[0]}")/.."

GPU="${GPU:-3}"
MODEL="${MODEL:-LOCAL_HOME/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct}"
BASE_ROOT="${BASE_ROOT:-data_ckplug/cast_confiqa_train_size_stability_qa_v0/n300}"
OUT="${OUT:-data_ckplug/cast_confiqa_discovery_size_sweep_qa_v0}"

SIZES="${SIZES:-8 16 24 32 48 60 120}"
DISCOVERY_TOPK_MLP="${DISCOVERY_TOPK_MLP:-4}"
DISCOVERY_TOPK_ATTN_LAYERS="${DISCOVERY_TOPK_ATTN_LAYERS:-4}"
HEAD_SCAN_ROWS_CAP="${HEAD_SCAN_ROWS_CAP:-24}"
HEAD_SCAN_FACTORS="${HEAD_SCAN_FACTORS:-0.0,1.5}"
HEAD_TOPK="${HEAD_TOPK:-4}"
HEAD_REFINE_EVAL_ROWS="${HEAD_REFINE_EVAL_ROWS:-1}"
SCORE_MODE="${SCORE_MODE:-answer_rest_margin}"

REF_PRIOR_MLP="${REF_PRIOR_MLP:-L1.mlp,L17.mlp,L2.mlp,L20.mlp}"
REF_SUPPRESS_HEADS="${REF_SUPPRESS_HEADS:-L13.attn.h17,L14.attn.h7,L14.attn.h30,L14.attn.h12}"
REF_BOOST_HEADS="${REF_BOOST_HEADS:-L13.attn.h18,L31.attn.h14,L14.attn.h20,L31.attn.h3}"

mkdir -p "$OUT"
MASTER_LOG="$OUT/discovery_size_sweep.log"
SUMMARY="$OUT/discovery_size_summary.tsv"
echo -e "size\thead_scan_rows\tmlp_components\tattn_layers\tsuppress_heads\tboost_heads\tmlp_jaccard\tsuppress_jaccard\tboost_jaccard" > "$SUMMARY"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  deactivate || true
fi
source LOCAL_HOME/anaconda3/etc/profile.d/conda.sh
conda activate screscomp
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

test -f "$BASE_ROOT/pairs/pairs.csv"
test -f "$BASE_ROOT/eval_open_rows.jsonl"

for n in $SIZES; do
  root="$OUT/n${n}"
  mkdir -p "$root/component_scan" "$root/selected" "$root/head_refine"
  head_scan_rows="$n"
  if [[ "$head_scan_rows" -gt "$HEAD_SCAN_ROWS_CAP" ]]; then head_scan_rows="$HEAD_SCAN_ROWS_CAP"; fi
  if [[ "$head_scan_rows" -lt 1 ]]; then head_scan_rows=1; fi

  echo "[$(date -Is)] discovery-size n=$n head_scan_rows=$head_scan_rows" | tee -a "$MASTER_LOG"

  CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.cecm_scan_component_contributions \
    --model "$MODEL" \
    --pairs-csv "$BASE_ROOT/pairs/pairs.csv" \
    --event source_context_over_prior \
    --split train \
    --start 0 \
    --max-rows "$n" \
    --component-types attn,mlp \
    --apply-mode decision_tokens \
    --score-mode "$SCORE_MODE" \
    --max-aliases-per-side 0 \
    --min-abs-delta 0.0 \
    --min-sign-consistency 0.50 \
    --allow-ci-cross-zero \
    --max-components-per-direction 16 \
    --torch-dtype bfloat16 \
    --device cuda \
    --out-dir "$root/component_scan" \
    > "$root/component_scan.log" 2>&1

  COMPONENT_SCREEN="$root/component_scan/component_screen.csv" \
  SELECT_OUT="$root/selected" \
  DISCOVERY_TOPK_MLP="$DISCOVERY_TOPK_MLP" \
  DISCOVERY_TOPK_ATTN_LAYERS="$DISCOVERY_TOPK_ATTN_LAYERS" \
  python - <<'PY'
import csv
import json
import os
from pathlib import Path

screen = Path(os.environ["COMPONENT_SCREEN"])
out = Path(os.environ["SELECT_OUT"])
topk_mlp = int(os.environ["DISCOVERY_TOPK_MLP"])
topk_attn = int(os.environ["DISCOVERY_TOPK_ATTN_LAYERS"])
rows = list(csv.DictReader(screen.open("r", encoding="utf-8", newline="")))

def f(row, key, default=0.0):
    try:
        value = str(row.get(key, "")).strip()
        return float(value) if value else default
    except Exception:
        return default

mlp_rows = [r for r in rows if r.get("component_type") == "mlp"]
attn_rows = [r for r in rows if r.get("component_type") == "attn"]
mlp_neg = [r for r in mlp_rows if f(r, "mean_delta") < 0]
selected_mlp = (
    sorted(mlp_neg, key=lambda r: (f(r, "mean_delta"), -f(r, "sign_consistency")))[:topk_mlp]
    if mlp_neg
    else sorted(mlp_rows, key=lambda r: (-f(r, "abs_mean_delta"), -f(r, "sign_consistency")))[:topk_mlp]
)
selected_attn = []
seen_layers = set()
for row in sorted(attn_rows, key=lambda r: (-f(r, "abs_mean_delta"), -f(r, "sign_consistency"))):
    layer = str(row["layer_idx"])
    if layer in seen_layers:
        continue
    seen_layers.add(layer)
    selected_attn.append(row)
    if len(selected_attn) >= topk_attn:
        break

with (out / "prior_mlp_components.csv").open("w", encoding="utf-8", newline="") as fp:
    writer = csv.DictWriter(fp, fieldnames=["component_id", "layer_idx", "component_type"])
    writer.writeheader()
    for row in selected_mlp:
        writer.writerow({
            "component_id": row["component_id"],
            "layer_idx": row["layer_idx"],
            "component_type": row["component_type"],
        })

attn_layers = ",".join(str(r["layer_idx"]) for r in selected_attn)
(out / "selected.env").write_text(
    "PRIOR_MLP_COMPONENTS=%s\nDISCOVERED_ATTN_LAYERS=%s\n" % (
        out / "prior_mlp_components.csv",
        attn_layers,
    ),
    encoding="utf-8",
)
(out / "selected_components.json").write_text(json.dumps({
    "mlp_components": selected_mlp,
    "attention_layers": selected_attn,
}, indent=2), encoding="utf-8")
PY

  # shellcheck disable=SC1090
  source "$root/selected/selected.env"
  CUDA_VISIBLE_DEVICES="$GPU" python -m screscomp.cli.cecm_run_attention_head_refine \
    --model "$MODEL" \
    --pairs-csv "$BASE_ROOT/pairs/pairs.csv" \
    --event source_context_over_prior \
    --eval-open-rows "$BASE_ROOT/eval_open_rows.jsonl" \
    --split train \
    --scan-start 0 \
    --scan-max-rows "$head_scan_rows" \
    --attn-layers "$DISCOVERED_ATTN_LAYERS" \
    --score-mode "$SCORE_MODE" \
    --scan-factors "$HEAD_SCAN_FACTORS" \
    --topks "$HEAD_TOPK" \
    --generation-kinds suppress,boost \
    --generation-apply-modes prefill \
    --generation-prompt-key base_rag \
    --start 0 \
    --max-rows "$HEAD_REFINE_EVAL_ROWS" \
    --torch-dtype bfloat16 \
    --device cuda \
    --out-dir "$root/head_refine" \
    > "$root/head_refine.log" 2>&1

  HEAD_PLAN="$root/head_refine/head_group_plan.csv" \
  SELECTED_ENV="$root/selected/selected.env" \
  python - <<'PY'
import csv
import os
from pathlib import Path

plan = Path(os.environ["HEAD_PLAN"])
env = Path(os.environ["SELECTED_ENV"])
rows = list(csv.DictReader(plan.open("r", encoding="utf-8", newline="")))
groups = {}
for row in rows:
    heads = []
    for item in str(row.get("actions", "")).split(","):
        item = item.strip()
        if not item:
            continue
        heads.append(item.split(":", 1)[0])
    groups[row["control_name"]] = heads
with env.open("a", encoding="utf-8") as fp:
    fp.write("SUPPRESS_HEADS=%s\n" % ",".join(groups.get("suppress_top4", [])))
    fp.write("BOOST_HEADS=%s\n" % ",".join(groups.get("boost_top4", [])))
PY

  SIZE="$n" ROOT_DIR="$root" HEAD_SCAN_ROWS="$head_scan_rows" SUMMARY="$SUMMARY" \
  REF_PRIOR_MLP="$REF_PRIOR_MLP" REF_SUPPRESS_HEADS="$REF_SUPPRESS_HEADS" REF_BOOST_HEADS="$REF_BOOST_HEADS" \
  python - <<'PY'
import csv
import os
from pathlib import Path

def parse_csv_set(raw: str):
    return {item.strip() for item in raw.split(",") if item.strip()}

def jaccard(a, b):
    return len(a & b) / len(a | b) if (a or b) else 1.0

root = Path(os.environ["ROOT_DIR"])
summary = Path(os.environ["SUMMARY"])
env_values = {}
for line in (root / "selected" / "selected.env").read_text(encoding="utf-8").splitlines():
    if "=" in line:
        key, value = line.split("=", 1)
        env_values[key] = value

mlp_rows = list(csv.DictReader((root / "selected" / "prior_mlp_components.csv").open("r", encoding="utf-8", newline="")))
mlp = ",".join(row["component_id"] for row in mlp_rows)
suppress = env_values.get("SUPPRESS_HEADS", "")
boost = env_values.get("BOOST_HEADS", "")
attn_layers = ",".join(f"L{layer}.attn" for layer in parse_csv_set(env_values.get("DISCOVERED_ATTN_LAYERS", "")))

values = [
    os.environ["SIZE"],
    os.environ["HEAD_SCAN_ROWS"],
    mlp,
    attn_layers,
    suppress,
    boost,
    jaccard(parse_csv_set(mlp), parse_csv_set(os.environ["REF_PRIOR_MLP"])),
    jaccard(parse_csv_set(suppress), parse_csv_set(os.environ["REF_SUPPRESS_HEADS"])),
    jaccard(parse_csv_set(boost), parse_csv_set(os.environ["REF_BOOST_HEADS"])),
]
with summary.open("a", encoding="utf-8") as fp:
    fp.write("\t".join(str(v) for v in values) + "\n")
PY
done

cat "$SUMMARY"
