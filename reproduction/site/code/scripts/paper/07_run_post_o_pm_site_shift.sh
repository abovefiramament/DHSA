#!/usr/bin/env bash
set -euo pipefail

# Post-O site-specificity protocol for PM component pools.
#
# Families:
#   mlp_pm  : selected positive+negative MLP residual-write sites.
#   head_pm : selected positive+negative post-O attention residual-write sites
#             (implemented as Lx.attn component outputs, not pre-O heads).
#
# Controls:
#   selected          same selected residual sites, trained without native-write zeroing.
#   shift1_samevec    remap selected vectors to a deterministic random +/-1 layer shift.
#   shift1_retrained  retrain at that +/-1 shifted site pool with the same budget.
#   shift4_samevec    remap selected vectors to a deterministic random +/-4 layer shift.
#   shift4_retrained  retrain at that +/-4 shifted site pool with the same budget.
#
# Shift rule:
#   For each selected component and shift distance d, sample one of +d/-d.
#   If the sampled side is out of range or collides with the selected/target pool,
#   try the opposite side. If both sides are unavailable, keep the original site
#   for that component and record the fallback in the manifest.

export LC_ALL=C.UTF-8

REPO_ROOT="${REPO_ROOT:-LOCAL_HOME/RPEC/projects/screscomp}"
cd "$REPO_ROOT"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

PY="${PY:-LOCAL_HOME/anaconda3/envs/screscomp/bin/python}"
RUN_ROOT="${RUN_ROOT:-/dev/shm/screscomp_runs/post_o_pm_site_shift_$(date -u +%Y%m%d_%H%M%S)}"
TASKS="${TASKS:-imdb qa mr mc}"
FAMILIES="${FAMILIES:-mlp_pm head_pm}"
SHIFTS="${SHIFTS:-}"
MLP_SHIFTS="${MLP_SHIFTS:-${SHIFTS:-1}}"
HEAD_SHIFTS="${HEAD_SHIFTS:-${SHIFTS:-1 4}}"
MODES="${MODES:-same_vector retrained}"
GPU="${GPU:-0}"

ALPHA_SWEEP="${ALPHA_SWEEP:-0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}"
SEED="${SEED:-20260629}"

IMDB_ROOT="${IMDB_ROOT:-/dev/shm/screscomp_runs/imdb_ma921_expanded1024_sampled_paperkl_20260626_055222}"
IMDB_MODEL="${IMDB_MODEL:-LOCAL_HOME/.cache/huggingface/hub/models--ma921--gpt2-large-sft-imdb/snapshots/f480190690d5abfc0e003ccb4f7e650626019bb9}"
IMDB_N_LAYERS="${IMDB_N_LAYERS:-36}"
IMDB_TRAIN_ROWS="${IMDB_TRAIN_ROWS:-512}"
IMDB_VAL_ROWS="${IMDB_VAL_ROWS:-256}"
IMDB_EPOCHS="${IMDB_EPOCHS:-2}"
IMDB_TRAIN_BATCH_SIZE="${IMDB_TRAIN_BATCH_SIZE:-32}"

CONFIQA_MODEL="${CONFIQA_MODEL:-LOCAL_HOME/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct}"
CONFIQA_N_LAYERS="${CONFIQA_N_LAYERS:-32}"
FETCH_DIR="${FETCH_DIR:-data_ckplug/context_dpo_confiqa_official}"
DISCOVERY_BASE="${DISCOVERY_BASE:-data_ckplug/cast_confiqa_all3_smalltrain_heldout5000_v0}"
DISCOVERY_QA="${DISCOVERY_QA:-$DISCOVERY_BASE/qa}"
DISCOVERY_MR="${DISCOVERY_MR:-data_ckplug/cast_confiqa_mr_smalltrain_disc120_eval500_v0/mr}"
DISCOVERY_MC="${DISCOVERY_MC:-data_ckplug/cast_confiqa_mc_smalltrain_disc120_eval500_v0/mc}"
CONFIQA_TOPK_POS="${CONFIQA_TOPK_POS:-4}"
CONFIQA_TOPK_NEG="${CONFIQA_TOPK_NEG:-4}"
CONFIQA_TRAIN_SOURCE_ROWS="${CONFIQA_TRAIN_SOURCE_ROWS:-300}"
CONFIQA_VAL_MOD="${CONFIQA_VAL_MOD:-5}"
CONFIQA_TRAIN_ROWS="${CONFIQA_TRAIN_ROWS:-240}"
CONFIQA_VAL_ROWS="${CONFIQA_VAL_ROWS:-60}"
CONFIQA_EPOCHS="${CONFIQA_EPOCHS:-2}"
CONFIQA_TRAIN_BATCH_SIZE="${CONFIQA_TRAIN_BATCH_SIZE:-4}"

export CUDA_VISIBLE_DEVICES="$GPU"

mkdir -p "$RUN_ROOT/logs"
STATUS="$RUN_ROOT/status.tsv"
SUMMARY="$RUN_ROOT/site_alpha_summary.tsv"
MASTER_LOG="$RUN_ROOT/site_shift.log"
printf 'time\tstage\tstatus\n' > "$STATUS"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$MASTER_LOG"
}

stage() {
  printf '%s\t%s\t%s\n' "$(date -Is)" "$1" "$2" >> "$STATUS"
  log "stage=$1 $2"
}

contains_word() {
  local haystack="$1"
  local needle="$2"
  for item in $haystack; do
    [[ "$item" == "$needle" ]] && return 0
  done
  return 1
}

data_json_for_task() {
  case "$1" in
    qa) echo "$FETCH_DIR/ConFiQA-QA.json" ;;
    mr) echo "$FETCH_DIR/ConFiQA-MR.json" ;;
    mc) echo "$FETCH_DIR/ConFiQA-MC.json" ;;
    *) echo "unsupported ConFiQA task: $1" >&2; exit 2 ;;
  esac
}

component_screen_for_task() {
  case "$1" in
    qa) echo "$DISCOVERY_QA/discovery/component_scan/component_screen.csv" ;;
    mr) echo "$DISCOVERY_MR/discovery/component_scan/component_screen.csv" ;;
    mc) echo "$DISCOVERY_MC/discovery/component_scan/component_screen.csv" ;;
    *) echo "unsupported ConFiQA task: $1" >&2; exit 2 ;;
  esac
}

component_ids_csv() {
  "$PY" - "$1" <<'PY'
import csv, sys
from pathlib import Path
rows = list(csv.DictReader(Path(sys.argv[1]).open(encoding="utf-8-sig", newline="")))
print(",".join(row["component_id"] for row in rows if row.get("component_id")))
PY
}

write_protocol() {
  "$PY" - "$RUN_ROOT/protocol.json" <<'PY'
import json, os, sys
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "experiment": "post_o_pm_site_shift",
    "families": {
        "mlp_pm": "positive+negative MLP residual-write components as one pool",
        "head_pm": "positive+negative post-O attention residual outputs (Lx.attn), not pre-O heads",
    },
    "main_site": "RCM-selected post-O residual-write sites",
    "controls": [
        "random +/-1 layer shift with same-vector remap",
        "random +/-1 layer shift with same-budget retraining",
        "random +/-4 layer shift with same-vector remap",
        "random +/-4 layer shift with same-budget retraining",
    ],
    "shift_rule": (
        "For each selected component, sample +d or -d with a fixed seed. If that side is "
        "out of range or collides with the selected/target pool, try the opposite side. "
        "If both sides are unavailable, use the original selected site for that component "
        "and record fallback=original_site_allowed."
    ),
    "site_protocol": (
        "Site-dependent expression is evaluated without native-write zeroing. Selected, "
        "shifted same-vector, and shifted retrained actuators are trained/evaluated on "
        "the same natural forward distribution; the manipulated variable is the residual "
        "write site. Zero-and-recovery is a separate causal-substitution protocol."
    ),
    "alpha_sweep": os.environ.get("ALPHA_SWEEP", ""),
    "seed": int(os.environ.get("SEED", "0")),
    "tasks": os.environ.get("TASKS", ""),
    "families_requested": os.environ.get("FAMILIES", ""),
    "mlp_shifts": os.environ.get("MLP_SHIFTS", ""),
    "head_shifts": os.environ.get("HEAD_SHIFTS", ""),
    "modes": os.environ.get("MODES", ""),
}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

prepare_confiqa_common() {
  local task="$1"
  local task_out="$2"
  local common="$task_out/common"
  local pairs_dir="$common/pairs"
  local train_open_rows="$common/train_open_rows.jsonl"
  local data_json
  data_json="$(data_json_for_task "$task")"
  mkdir -p "$common"
  if [[ -f "$pairs_dir/pairs.csv" ]]; then
    return
  fi
  stage "prepare_${task}_pairs" running
  "$PY" -m screscomp.cli.cecm_fetch_confiqa --out-dir "$FETCH_DIR" --tasks "$task" \
    >"$common/fetch.log" 2>&1
  "$PY" -m screscomp.cli.prepare_ckplug_open \
    --dataset confiqa \
    --data_json "$data_json" \
    --out_jsonl "$train_open_rows" \
    --schema base \
    --alias_policy raw \
    --max_rows "$CONFIQA_TRAIN_SOURCE_ROWS" \
    >"$common/prepare_open.log" 2>&1
  "$PY" -m screscomp.cli.cecm_build_preference_pairs \
    --input-jsonl "$train_open_rows" \
    --out-dir "$pairs_dir" \
    --event source_context_over_prior \
    --prompt-key base_rag \
    --prior-source dataset_orig \
    --val-mod "$CONFIQA_VAL_MOD" \
    >"$common/build_pairs.log" 2>&1
  stage "prepare_${task}_pairs" done
}

generate_site_specs() {
  local task="$1"
  local family="$2"
  local task_out="$3"
  local source_kind="$4"
  local n_layers="$5"
  local component_type="$6"
  local family_shifts="$7"
  local out_dir="$task_out/site_specs/$family"
  local confiqa_screen=""
  mkdir -p "$out_dir"
  if [[ "$source_kind" == "confiqa" ]]; then
    confiqa_screen="$(component_screen_for_task "${task#confiqa_}")"
  fi

  SOURCE_KIND="$source_kind" TASK_NAME="$task" FAMILY_NAME="$family" N_LAYERS="$n_layers" \
  COMPONENT_TYPE="$component_type" OUT_DIR="$out_dir" SEED_VALUE="$SEED" SHIFTS_VALUE="$family_shifts" \
  IMDB_ROOT_VALUE="$IMDB_ROOT" CONFIQA_SCREEN="$confiqa_screen" \
  CONFIQA_TOPK_POS_VALUE="$CONFIQA_TOPK_POS" CONFIQA_TOPK_NEG_VALUE="$CONFIQA_TOPK_NEG" "$PY" - <<'PY'
import csv
import hashlib
import json
import os
import random
import re
from pathlib import Path

source_kind = os.environ["SOURCE_KIND"]
task = os.environ["TASK_NAME"]
family = os.environ["FAMILY_NAME"]
n_layers = int(os.environ["N_LAYERS"])
component_type = os.environ["COMPONENT_TYPE"]
out_dir = Path(os.environ["OUT_DIR"])
seed_base = int(os.environ["SEED_VALUE"])
shifts = [int(x) for x in os.environ["SHIFTS_VALUE"].split() if x.strip()]
out_dir.mkdir(parents=True, exist_ok=True)

def f(row, key, default=0.0):
    try:
        text = str(row.get(key, "")).strip()
        return float(text) if text else default
    except Exception:
        return default

def read_csv(path: Path):
    return list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))

def write_csv(path: Path, rows: list[dict]):
    fields = ["component_id", "layer_idx", "component_type"]
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def normalize_row(row: dict, *, signed_role: str, rank: int, source: str) -> dict:
    cid = str(row["component_id"])
    layer = int(float(row.get("layer_idx") or re.fullmatch(r"L(\d+)\.(?:attn|mlp)", cid).group(1)))
    return {
        **row,
        "component_id": cid,
        "layer_idx": str(layer),
        "component_type": component_type,
        "pm_role": signed_role,
        "pm_rank": str(rank),
        "site_source": source,
        "family": family,
    }

def select_from_screen(path: Path) -> list[dict]:
    rows = [r for r in read_csv(path) if r.get("component_type") == component_type]
    if not rows:
        raise SystemExit(f"No rows for component_type={component_type} in {path}")
    pos_topk = int(os.environ["CONFIQA_TOPK_POS_VALUE"])
    neg_topk = int(os.environ["CONFIQA_TOPK_NEG_VALUE"])
    pos = [r for r in rows if f(r, "mean_delta") > 0]
    neg = [r for r in rows if f(r, "mean_delta") < 0]
    pos_ranked = sorted(pos, key=lambda r: (f(r, "mean_delta"), f(r, "sign_consistency")), reverse=True)
    neg_ranked = sorted(neg, key=lambda r: (f(r, "mean_delta"), -f(r, "sign_consistency")))
    selected = []
    used = set()
    for role, ranked, k in (("positive", pos_ranked, pos_topk), ("negative", neg_ranked, neg_topk)):
        picked = []
        for row in ranked:
            if row["component_id"] in used:
                continue
            picked.append(row)
            used.add(row["component_id"])
            if len(picked) == k:
                break
        if len(picked) < k:
            fallback = sorted(
                [r for r in rows if r["component_id"] not in used],
                key=lambda r: (f(r, "abs_mean_delta"), f(r, "sign_consistency")),
                reverse=True,
            )
            for row in fallback:
                picked.append(row)
                used.add(row["component_id"])
                if len(picked) == k:
                    break
        if len(picked) < k:
            raise SystemExit(f"Need {k} {role} rows for {family}, found {len(picked)}")
        selected.extend(normalize_row(row, signed_role=role, rank=i + 1, source="component_screen") for i, row in enumerate(picked))
    return selected

def select_from_imdb(root: Path) -> list[dict]:
    prefix = "mlp" if component_type == "mlp" else "attn"
    pos_rows = read_csv(root / "components" / f"{prefix}_positive_components.csv")
    neg_rows = read_csv(root / "components" / f"{prefix}_negative_components.csv")
    selected = []
    selected.extend(normalize_row(row, signed_role="positive", rank=i + 1, source="imdb_locked") for i, row in enumerate(pos_rows))
    selected.extend(normalize_row(row, signed_role="negative", rank=i + 1, source="imdb_locked") for i, row in enumerate(neg_rows))
    return selected

def layer_of(component_id: str) -> int:
    m = re.fullmatch(r"L(\d+)\.(?:attn|mlp)", component_id)
    if not m:
        raise ValueError(component_id)
    return int(m.group(1))

def component_id(layer: int) -> str:
    return f"L{layer}.{component_type}"

def stable_seed(*parts: object) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return seed_base + int(h[:8], 16)

def shifted_rows(selected: list[dict], distance: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(stable_seed(task, family, distance))
    original_ids = {row["component_id"] for row in selected}
    used_targets: set[str] = set()
    rows: list[dict] = []
    decisions: list[dict] = []
    for idx, row in enumerate(selected):
        source_id = row["component_id"]
        source_layer = layer_of(source_id)
        first_sign = rng.choice([1, -1])
        offsets = [first_sign * distance, -first_sign * distance]
        chosen_id = None
        chosen_offset = None
        rejected = []
        for offset in offsets:
            target_layer = source_layer + offset
            target_id = component_id(target_layer)
            if target_layer < 0 or target_layer >= n_layers:
                rejected.append({"offset": offset, "target": target_id, "reason": "out_of_range"})
                continue
            if target_id in original_ids:
                rejected.append({"offset": offset, "target": target_id, "reason": "collides_with_original_pool"})
                continue
            if target_id in used_targets:
                rejected.append({"offset": offset, "target": target_id, "reason": "collides_with_target_pool"})
                continue
            chosen_id = target_id
            chosen_offset = offset
            break
        fallback = ""
        if chosen_id is None:
            chosen_id = source_id
            chosen_offset = 0
            fallback = "original_site_allowed"
        used_targets.add(chosen_id)
        target_layer = layer_of(chosen_id)
        target_row = {
            **row,
            "component_id": chosen_id,
            "layer_idx": str(target_layer),
            "component_type": component_type,
            "source_component_id": source_id,
            "source_layer_idx": str(source_layer),
            "shift_distance": str(distance),
            "shift_offset": str(chosen_offset),
            "shift_fallback": fallback,
            "site_source": f"shift{distance}",
            "site_control_kind": f"shift{distance}",
            "target_order": str(idx + 1),
        }
        rows.append(target_row)
        decisions.append({
            "source_component_id": source_id,
            "target_component_id": chosen_id,
            "source_layer_idx": source_layer,
            "target_layer_idx": target_layer,
            "sampled_first_offset": offsets[0],
            "chosen_offset": chosen_offset,
            "fallback": fallback,
            "rejected": rejected,
        })
    return rows, decisions

if source_kind == "imdb":
    selected = select_from_imdb(Path(os.environ["IMDB_ROOT_VALUE"]))
elif source_kind == "confiqa":
    selected = select_from_screen(Path(os.environ["CONFIQA_SCREEN"]))
else:
    raise SystemExit(f"Unsupported SOURCE_KIND={source_kind}")

selected_csv = out_dir / "selected_components.csv"
write_csv(selected_csv, selected)
(out_dir / "selected_ids.txt").write_text(",".join(row["component_id"] for row in selected) + "\n", encoding="utf-8")

manifest = {
    "task": task,
    "family": family,
    "component_type": component_type,
    "n_layers": n_layers,
    "source_kind": source_kind,
    "selected_components": [row["component_id"] for row in selected],
    "shift_controls": {},
}
for distance in shifts:
    rows, decisions = shifted_rows(selected, distance)
    write_csv(out_dir / f"shift{distance}_components.csv", rows)
    manifest["shift_controls"][f"shift{distance}"] = {
        "components": [row["component_id"] for row in rows],
        "decisions": decisions,
    }
(out_dir / "site_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps({"selected": manifest["selected_components"], "shifts": {k: v["components"] for k, v in manifest["shift_controls"].items()}}, indent=2), flush=True)
PY
}

remap_fixed_payload() {
  local src_payload="$1"
  local target_csv="$2"
  local dst_payload="$3"
  local task="$4"
  local family="$5"
  local control="$6"
  mkdir -p "$(dirname "$dst_payload")"
  SRC_PAYLOAD="$src_payload" TARGET_CSV="$target_csv" DST_PAYLOAD="$dst_payload" \
  TASK_NAME="$task" FAMILY_NAME="$family" CONTROL_NAME="$control" "$PY" - <<'PY'
import csv
import os
from pathlib import Path

import torch

src = Path(os.environ["SRC_PAYLOAD"])
target_csv = Path(os.environ["TARGET_CSV"])
dst = Path(os.environ["DST_PAYLOAD"])
task = os.environ["TASK_NAME"]
family = os.environ["FAMILY_NAME"]
control = os.environ["CONTROL_NAME"]

payload = torch.load(src, map_location="cpu")
source_components = [dict(row) for row in payload.get("components", [])]
target_rows = list(csv.DictReader(target_csv.open(encoding="utf-8-sig", newline="")))
if len(source_components) != len(target_rows):
    raise SystemExit(f"source/target component count mismatch: {len(source_components)} vs {len(target_rows)}")

new_components = []
new_vectors = {}
mapping = []
for source_row, target_row in zip(source_components, target_rows, strict=False):
    source_id = str(source_row["component_id"])
    target_id = str(target_row["component_id"])
    if source_id not in payload["vectors"]:
        raise SystemExit(f"missing source vector for {source_id}")
    vector = payload["vectors"][source_id]
    new_components.append({
        "component_id": target_id,
        "layer_idx": int(float(target_row["layer_idx"])),
        "component_type": str(target_row["component_type"]),
        "source_component_id": source_id,
        "site_control_kind": control,
    })
    new_vectors[target_id] = vector.detach().clone()
    mapping.append({"source_component_id": source_id, "target_component_id": target_id})

payload["components"] = new_components
payload["vectors"] = new_vectors
payload["site_control"] = {
    "task": task,
    "family": family,
    "control": control,
    "mode": "same_vector_remap",
    "mapping": mapping,
}
dst.parent.mkdir(parents=True, exist_ok=True)
torch.save(payload, dst)
print(f"[remap] {src} -> {dst} ({control})", flush=True)
PY
}

run_train_fixed() {
  local task="$1"
  local family="$2"
  local control="$3"
  local model="$4"
  local pairs_csv="$5"
  local components_csv="$6"
  local event="$7"
  local apply_mode="$8"
  local max_train_rows="$9"
  local max_val_rows="${10}"
  local epochs="${11}"
  local train_batch_size="${12}"
  local score_mode="${13}"
  local preference_mode="${14}"
  local out_dir="${15}"
  local init_payload="${16:-}"

  if [[ -f "$out_dir/alpha_summary.csv" && -f "$out_dir/fixed_actuator.pt" ]]; then
    stage "train_${task}_${family}_${control}" reused
    return
  fi
  stage "train_${task}_${family}_${control}" running
  mkdir -p "$out_dir"
  local extra=()
  if [[ -n "$init_payload" ]]; then
    extra+=(--init-fixed-actuator "$init_payload")
  fi
  if [[ "$task" == "imdb" ]]; then
    extra+=(--endpoint-objective pair_margin --dpo-beta 1.0 --causal-train-mask --option-selection-mode model_max)
  else
    extra+=(--endpoint-objective pair_margin --max-aliases-per-side 1)
  fi
  "$PY" -m screscomp.cli.cecm_train_fixed_actuator \
    --model "$model" \
    --pairs-csv "$pairs_csv" \
    --components-csv "$components_csv" \
    --event "$event" \
    --train-split train \
    --val-split val \
    --max-train-rows "$max_train_rows" \
    --max-val-rows "$max_val_rows" \
    --epochs "$epochs" \
    --train-batch-size "$train_batch_size" \
    --lr 0.05 \
    --lambda-norm 0.0001 \
    --alpha-train 1.0 \
    --alpha-sweep "$ALPHA_SWEEP" \
    --preference-loss-mode "$preference_mode" \
    --state-margin-weight 0.0 \
    --gain-weight 1.0 \
    --target-margin 0.0 \
    --target-gain 0.0 \
    --apply-mode "$apply_mode" \
    --score-mode "$score_mode" \
    --empty-cache-every 25 \
    --torch-dtype bfloat16 \
    --device cuda \
    "${extra[@]}" \
    --out-dir "$out_dir" \
    >"$out_dir/train.log" 2>&1
  stage "train_${task}_${family}_${control}" done
}

run_family() {
  local task="$1"
  local family="$2"
  local task_out="$3"
  local source_kind="$4"
  local component_type="$5"
  local n_layers="$6"
  local model="$7"
  local pairs_csv="$8"
  local event="$9"
  local apply_mode="${10}"
  local max_train_rows="${11}"
  local max_val_rows="${12}"
  local epochs="${13}"
  local train_batch_size="${14}"
  local score_mode="${15}"
  local preference_mode="${16}"
  local family_shifts="${17}"

  generate_site_specs "$task" "$family" "$task_out" "$source_kind" "$n_layers" "$component_type" "$family_shifts"

  local specs_dir="$task_out/site_specs/$family"
  local train_dir="$task_out/train/$family"
  local selected_csv="$specs_dir/selected_components.csv"
  run_train_fixed "$task" "$family" "selected" "$model" "$pairs_csv" "$selected_csv" "$event" "$apply_mode" \
    "$max_train_rows" "$max_val_rows" "$epochs" "$train_batch_size" "$score_mode" "$preference_mode" \
    "$train_dir/selected" ""

  for shift in $family_shifts; do
    local control="shift${shift}"
    local control_csv="$specs_dir/${control}_components.csv"
    local remap_payload="$task_out/site_payloads/$family/${control}_same_vector/fixed_actuator.pt"
    if contains_word "$MODES" "same_vector"; then
      remap_fixed_payload "$train_dir/selected/fixed_actuator.pt" "$control_csv" "$remap_payload" "$task" "$family" "$control"
      run_train_fixed "$task" "$family" "${control}_same_vector" "$model" "$pairs_csv" "$control_csv" "$event" "$apply_mode" \
        "$max_train_rows" "$max_val_rows" 0 "$train_batch_size" "$score_mode" "$preference_mode" \
        "$train_dir/${control}_same_vector" "$remap_payload"
    fi
    if contains_word "$MODES" "retrained"; then
      run_train_fixed "$task" "$family" "${control}_retrained" "$model" "$pairs_csv" "$control_csv" "$event" "$apply_mode" \
        "$max_train_rows" "$max_val_rows" "$epochs" "$train_batch_size" "$score_mode" "$preference_mode" \
        "$train_dir/${control}_retrained" ""
    fi
  done
}

run_imdb() {
  local task_out="$RUN_ROOT/imdb"
  mkdir -p "$task_out"
  local pairs_csv="$IMDB_ROOT/pairs/pairs.csv"
  test -f "$pairs_csv"
  if contains_word "$FAMILIES" "mlp_pm"; then
    run_family imdb mlp_pm "$task_out" imdb mlp "$IMDB_N_LAYERS" "$IMDB_MODEL" "$pairs_csv" \
      imdb_positive_sentiment prefill "$IMDB_TRAIN_ROWS" "$IMDB_VAL_ROWS" "$IMDB_EPOCHS" "$IMDB_TRAIN_BATCH_SIZE" avglogp dpo "$MLP_SHIFTS"
  fi
  if contains_word "$FAMILIES" "head_pm"; then
    run_family imdb head_pm "$task_out" imdb attn "$IMDB_N_LAYERS" "$IMDB_MODEL" "$pairs_csv" \
      imdb_positive_sentiment all "$IMDB_TRAIN_ROWS" "$IMDB_VAL_ROWS" "$IMDB_EPOCHS" "$IMDB_TRAIN_BATCH_SIZE" avglogp dpo "$HEAD_SHIFTS"
  fi
}

run_confiqa_task() {
  local short="$1"
  local task="confiqa_$short"
  local task_out="$RUN_ROOT/$task"
  mkdir -p "$task_out"
  prepare_confiqa_common "$short" "$task_out"
  local pairs_csv="$task_out/common/pairs/pairs.csv"
  test -f "$pairs_csv"
  if contains_word "$FAMILIES" "mlp_pm"; then
    run_family "$task" mlp_pm "$task_out" confiqa mlp "$CONFIQA_N_LAYERS" "$CONFIQA_MODEL" "$pairs_csv" \
      source_context_over_prior decision_tokens "$CONFIQA_TRAIN_ROWS" "$CONFIQA_VAL_ROWS" "$CONFIQA_EPOCHS" "$CONFIQA_TRAIN_BATCH_SIZE" answer_rest_margin margin_gain "$MLP_SHIFTS"
  fi
  if contains_word "$FAMILIES" "head_pm"; then
    run_family "$task" head_pm "$task_out" confiqa attn "$CONFIQA_N_LAYERS" "$CONFIQA_MODEL" "$pairs_csv" \
      source_context_over_prior decision_tokens "$CONFIQA_TRAIN_ROWS" "$CONFIQA_VAL_ROWS" "$CONFIQA_EPOCHS" "$CONFIQA_TRAIN_BATCH_SIZE" answer_rest_margin margin_gain "$HEAD_SHIFTS"
  fi
}

collect_summary() {
  stage collect_summary running
  "$PY" - "$RUN_ROOT" "$SUMMARY" <<'PY'
import csv
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
out = Path(sys.argv[2])
rows_out = []
for path in sorted(root.glob("**/alpha_summary.csv")):
    rel = path.relative_to(root).parts
    if len(rel) < 5 or rel[-1] != "alpha_summary.csv":
        continue
    task = rel[0]
    family = rel[2] if rel[1] == "train" else ""
    control = rel[3] if rel[1] == "train" and len(rel) >= 5 else path.parent.name
    mode = "selected"
    shift = ""
    m = re.fullmatch(r"shift(\d+)_(same_vector|retrained)", control)
    if m:
        shift = m.group(1)
        mode = m.group(2)
    with path.open(encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            rows_out.append({
                "task": task,
                "family": family,
                "control": control,
                "shift": shift,
                "mode": mode,
                "split": row.get("split", ""),
                "alpha": row.get("alpha", ""),
                "n": row.get("n", ""),
                "base_mean_margin": row.get("base_mean_margin", ""),
                "mean_margin": row.get("mean_margin", ""),
                "mean_margin_gain": row.get("mean_margin_gain", ""),
                "base_pref_rate": row.get("base_pref_rate", ""),
                "pref_rate": row.get("pref_rate", ""),
                "gain_positive_rate": row.get("gain_positive_rate", ""),
                "dpo_win_rate": row.get("dpo_win_rate", ""),
                "run_dir": str(path.parent),
            })
fields = [
    "task", "family", "control", "shift", "mode", "split", "alpha", "n",
    "base_mean_margin", "mean_margin", "mean_margin_gain", "base_pref_rate",
    "pref_rate", "gain_positive_rate", "dpo_win_rate", "run_dir",
]
with out.open("w", encoding="utf-8", newline="") as fp:
    writer = csv.DictWriter(fp, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows_out)
print(f"[summary] rows={len(rows_out)} out={out}", flush=True)
PY
  stage collect_summary done
}

stage preflight running
write_protocol
"$PY" -m py_compile \
  src/screscomp/cli/cecm_train_fixed_actuator.py \
  src/screscomp/cli/cecm_fetch_confiqa.py \
  src/screscomp/cli/prepare_ckplug_open.py \
  src/screscomp/cli/cecm_build_preference_pairs.py
stage preflight done

for task in $TASKS; do
  case "$task" in
    imdb) run_imdb ;;
    qa|mr|mc) run_confiqa_task "$task" ;;
    confiqa_qa) run_confiqa_task qa ;;
    confiqa_mr) run_confiqa_task mr ;;
    confiqa_mc) run_confiqa_task mc ;;
    *) echo "Unknown task: $task" >&2; exit 2 ;;
  esac
done

collect_summary
stage complete done
log "done run_root=$RUN_ROOT"
cat "$SUMMARY"
