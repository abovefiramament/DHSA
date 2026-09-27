#!/usr/bin/env bash
set -euo pipefail

export LC_ALL=C.UTF-8

REPO_ROOT="${REPO_ROOT:-LOCAL_HOME/RPEC/projects/screscomp}"
cd "$REPO_ROOT"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

PY="${PY:-LOCAL_HOME/anaconda3/envs/screscomp/bin/python}"
ROOT="${ROOT:-/dev/shm/screscomp_runs/imdb_ma921_expanded1024_sampled_paperkl_20260626_055222}"
ZERO_OUT="${ZERO_OUT:-/dev/shm/screscomp_runs/imdb_zero512_mlp_then_head_20260627_091106}"
OUT="${OUT:-/dev/shm/screscomp_runs/imdb_site_controls_512_$(date -u +%Y%m%d_%H%M%S)}"
MODEL="${MODEL:-LOCAL_HOME/.cache/huggingface/hub/models--ma921--gpt2-large-sft-imdb/snapshots/f480190690d5abfc0e003ccb4f7e650626019bb9}"

GPU="${GPU:-0}"
AXES="${AXES:-positive,negative}"
CONTROLS="${CONTROLS:-shifted,random,highact_lowrcm}"
MODES="${MODES:-same_vector,retrained}"
SITE_SPEC_ONLY="${SITE_SPEC_ONLY:-0}"

TRAIN_ROWS="${TRAIN_ROWS:-512}"
VAL_ROWS="${VAL_ROWS:-256}"
ALPHA_SWEEP="${ALPHA_SWEEP:-0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}"
NONZERO_SWEEP="${NONZERO_SWEEP:-0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}"
GEN_BATCH="${GEN_BATCH:-32}"
SCORE_BATCH="${SCORE_BATCH:-32}"
KL_BATCH="${KL_BATCH:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"

POS_HEADS="${POS_HEADS:-L16.attn.h13,L15.attn.h14,L0.attn.h5,L9.attn.h6}"
NEG_HEADS="${NEG_HEADS:-L14.attn.h5,L16.attn.h17,L16.attn.h4,L7.attn.h14}"
POS_MLP_COMPONENTS="${POS_MLP_COMPONENTS:-$ROOT/components/mlp_positive_components.csv}"
NEG_MLP_COMPONENTS="${NEG_MLP_COMPONENTS:-$ROOT/components/mlp_negative_components.csv}"

export CUDA_VISIBLE_DEVICES="$GPU"

mkdir -p "$OUT/logs" "$OUT/site_specs" "$OUT/train" "$OUT/evaluation/site_controls"
STATUS="$OUT/status.tsv"
if [[ ! -f "$STATUS" ]]; then
  printf 'time\tstage\tstatus\n' > "$STATUS"
fi

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$OUT/logs/site_controls.log"
}

stage() {
  printf '%s\t%s\t%s\n' "$(date -Is)" "$1" "$2" >> "$STATUS"
  log "stage=$1 $2"
}

component_ids() {
  "$PY" - "$1" <<'PY'
import csv, sys
with open(sys.argv[1], newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))
print(",".join(row["component_id"] for row in rows if row.get("component_id")))
PY
}

summary_rows() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo 0
    return
  fi
  local lines
  lines="$(wc -l < "$path")"
  if [[ "$lines" -le 0 ]]; then
    echo 0
  else
    echo $((lines - 1))
  fi
}

group_done() {
  local out_dir="$1"
  local expected_rows="$2"
  [[ "$(summary_rows "$out_dir/score_summary.csv")" -ge "$expected_rows" ]]
}

contains_csv_item() {
  local list="$1"
  local needle="$2"
  IFS=',' read -r -a arr <<< "$list"
  for item in "${arr[@]}"; do
    [[ "$item" == "$needle" ]] && return 0
  done
  return 1
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

write_protocol() {
  cat > "$OUT/protocol.json" <<EOF
{
  "experiment": "imdb_substitution_site_specificity",
  "source_run": "$ROOT",
  "same_site_zero_run": "$ZERO_OUT",
  "model": "$MODEL",
  "axes": "$AXES",
  "controls": "$CONTROLS",
  "modes": "$MODES",
  "train_rows": $TRAIN_ROWS,
  "val_rows": $VAL_ROWS,
  "eval_prompts": "$ROOT/eval_env/val/prompts.jsonl",
  "pairs_csv": "$ROOT/pairs/pairs.csv",
  "alpha_sweep": "$ALPHA_SWEEP",
  "nonzero_sweep": "$NONZERO_SWEEP",
  "scientific_setting": "Original RCM-selected MLP/head sites are zeroed. Same-vector controls remount the learned zero-trained vector at alternate sites without optimization. Retrained controls use the same train-row budget and objective but can write only through alternate sites while original sites remain zeroed."
}
EOF
}

generate_site_specs() {
  stage generate_site_specs running
  POS_HEADS="$POS_HEADS" NEG_HEADS="$NEG_HEADS" "$PY" - "$ROOT" "$ZERO_OUT" "$OUT" "$POS_MLP_COMPONENTS" "$NEG_MLP_COMPONENTS" <<'PY'
import csv
import json
import os
import random
import re
import shlex
import statistics
import sys
from pathlib import Path

import torch

root = Path(sys.argv[1])
zero_out = Path(sys.argv[2])
out = Path(sys.argv[3])
pos_mlp_csv = Path(sys.argv[4])
neg_mlp_csv = Path(sys.argv[5])
spec_dir = out / "site_specs"
payload_dir = out / "site_payloads"
spec_dir.mkdir(parents=True, exist_ok=True)
payload_dir.mkdir(parents=True, exist_ok=True)

N_LAYERS = int(os.environ.get("MODEL_N_LAYERS", "36"))
NUM_HEADS = int(os.environ.get("MODEL_NUM_HEADS", "20"))
HEAD_DIM = int(os.environ.get("MODEL_HEAD_DIM", "64"))
SHIFT_OFFSETS = (3, 4, -3, -4, 5, -5, 6, -6, 7, -7, 8, -8, 9, -9, 10, -10, 1, -1, 2, -2, 11, -11, 12, -12)

def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def write_csv(path: Path, rows: list[dict], preferred_fields: list[str] | None = None):
    fields = []
    if preferred_fields:
        fields.extend(preferred_fields)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def load_manifest_json(path: Path) -> dict:
    try:
        return json.load(path.open(encoding="utf-8"))
    except FileNotFoundError:
        return {}

selection_manifest = load_manifest_json(root / "components" / "selection_manifest.json")
scan_path = Path(selection_manifest.get("component_directionality_csv", ""))
if not scan_path.exists():
    scan_path = root / "components" / "component_directionality.csv"
head_manifest = load_manifest_json(root / "train" / "head_scan_pool" / "head_selection_manifest.json")
head_scan_path = Path(head_manifest.get("directionality_csv", ""))

scan_rows = read_csv(scan_path) if scan_path.exists() else []
head_scan_rows = read_csv(head_scan_path) if head_scan_path.exists() else []
scan_by_id = {row.get("component_id", ""): row for row in scan_rows}
head_scan_by_id = {row.get("component_id", ""): row for row in head_scan_rows}

pos_mlp_rows = read_csv(pos_mlp_csv)
neg_mlp_rows = read_csv(neg_mlp_csv)
orig = {
    "positive": {
        "short": "pos",
        "mlp_rows": pos_mlp_rows,
        "mlp_ids": [row["component_id"] for row in pos_mlp_rows],
        "heads": [x for x in os.environ["POS_HEADS"].split(",") if x],
        "mlp_payload": zero_out / "train" / "mlp_positive_zero_train_512" / "fixed_actuator.pt",
        "head_payload": zero_out / "train" / "head_positive_zero_train_512" / "head_actuator.pt",
        "target": "positive",
    },
    "negative": {
        "short": "neg",
        "mlp_rows": neg_mlp_rows,
        "mlp_ids": [row["component_id"] for row in neg_mlp_rows],
        "heads": [x for x in os.environ["NEG_HEADS"].split(",") if x],
        "mlp_payload": zero_out / "train" / "mlp_negative_zero_train_512" / "fixed_actuator.pt",
        "head_payload": zero_out / "train" / "head_negative_zero_train_512" / "head_actuator.pt",
        "target": "negative",
    },
}

all_selected_mlp = set(orig["positive"]["mlp_ids"]) | set(orig["negative"]["mlp_ids"])
all_selected_heads = set(orig["positive"]["heads"]) | set(orig["negative"]["heads"])

mlp_pat = re.compile(r"^L(\d+)\.mlp$")
head_pat = re.compile(r"^L(\d+)\.attn\.h(\d+)$")

def parse_mlp(component_id: str) -> int:
    m = mlp_pat.match(component_id)
    if not m:
        raise ValueError(component_id)
    return int(m.group(1))

def parse_head(head_id: str) -> tuple[int, int]:
    m = head_pat.match(head_id)
    if not m:
        raise ValueError(head_id)
    return int(m.group(1)), int(m.group(2))

def shifted_mlp(ids: list[str], blocked: set[str]) -> list[str]:
    used = set(blocked)
    out_ids = []
    for component_id in ids:
        layer = parse_mlp(component_id)
        for offset in SHIFT_OFFSETS:
            new_layer = layer + offset
            new_id = f"L{new_layer}.mlp"
            if 0 <= new_layer < N_LAYERS and new_id not in used:
                out_ids.append(new_id)
                used.add(new_id)
                break
        else:
            raise RuntimeError(f"Could not shift {component_id}")
    return out_ids

def shifted_heads(ids: list[str], blocked: set[str]) -> list[str]:
    used = set(blocked)
    out_ids = []
    for head_id in ids:
        layer, head = parse_head(head_id)
        for offset in SHIFT_OFFSETS:
            new_layer = layer + offset
            new_id = f"L{new_layer}.attn.h{head}"
            if 0 <= new_layer < N_LAYERS and new_id not in used:
                out_ids.append(new_id)
                used.add(new_id)
                break
        else:
            raise RuntimeError(f"Could not shift {head_id}")
    return out_ids

def f(row: dict, key: str, default: float = 0.0) -> float:
    try:
        text = str(row.get(key, "")).strip()
        return float(text) if text else default
    except Exception:
        return default

def healthy(row: dict) -> bool:
    if str(row.get("health_ok", "1")) not in {"1", "true", "True", ""}:
        return False
    if row.get("mean_ablated_format_ok") and f(row, "mean_ablated_format_ok") < 0.90:
        return False
    if row.get("healthy_pair_rate") and f(row, "healthy_pair_rate") < 0.86:
        return False
    if row.get("format_collapse_rate") and f(row, "format_collapse_rate") > 0.12:
        return False
    return True

def valid_mlp_candidates(blocked: set[str]) -> list[dict]:
    return [
        row for row in scan_rows
        if row.get("component_type") == "mlp"
        and row.get("component_id") not in blocked
        and mlp_pat.match(row.get("component_id", ""))
        and healthy(row)
    ]

def valid_head_candidates(blocked: set[str]) -> list[dict]:
    return [
        row for row in head_scan_rows
        if row.get("component_type") in {"head", "attn_head"}
        and row.get("component_id") not in blocked
        and head_pat.match(row.get("component_id", ""))
        and healthy(row)
    ]

def choose_random(rows: list[dict], k: int, seed: int) -> list[str]:
    ids = [row["component_id"] for row in rows]
    if len(ids) < k:
        raise RuntimeError(f"Need {k} random candidates, found {len(ids)}")
    rng = random.Random(seed)
    rng.shuffle(ids)
    return ids[:k]

def choose_highact_lowrcm(rows: list[dict], *, target: str, k: int) -> list[str]:
    active_key = f"healthy_{'pos' if target == 'positive' else 'neg'}_active_mean"
    mass_key = f"healthy_{'pos' if target == 'positive' else 'neg'}_mass"
    active = [f(row, active_key) for row in rows]
    threshold = statistics.median(active) if active else 0.0
    high_active = [row for row in rows if f(row, active_key) >= threshold]
    pool = high_active if len(high_active) >= k else rows
    ranked = sorted(pool, key=lambda row: (f(row, mass_key), -f(row, active_key), row.get("component_id", "")))
    if len(ranked) < k:
        raise RuntimeError(f"Need {k} highact-lowrcm candidates, found {len(ranked)}")
    return [row["component_id"] for row in ranked[:k]]

def mlp_row_for(component_id: str, *, rank: int, group_name: str, direction: str, control: str) -> dict:
    layer = parse_mlp(component_id)
    row = dict(scan_by_id.get(component_id, {}))
    row.update({
        "component_id": component_id,
        "layer_idx": str(layer),
        "component_type": "mlp",
        "rank": str(rank),
        "group_name": group_name,
        "selection_direction": direction,
        "site_control_kind": control,
    })
    return row

def head_row_for(head_id: str, *, rank: int, group_name: str, direction: str, control: str) -> dict:
    layer, head = parse_head(head_id)
    row = dict(head_scan_by_id.get(head_id, {}))
    row.update({
        "rank": str(rank),
        "head_id": head_id,
        "component_id": head_id,
        "layer_idx": str(layer),
        "head_idx": str(head),
        "component_type": "head",
        "selection_direction": direction,
        "site_control_kind": control,
    })
    return row

def write_head_txt(path: Path, heads: list[str]) -> None:
    path.write_text(",".join(heads) + "\n", encoding="utf-8")

def remap_mlp_payload(src: Path, targets: list[str], dst: Path, *, axis: str, control: str) -> None:
    payload = torch.load(src, map_location="cpu")
    source_ids = [row["component_id"] for row in payload["components"]]
    if len(source_ids) != len(targets):
        raise RuntimeError(f"MLP source/target length mismatch for {axis}/{control}")
    new_components = []
    new_vectors = {}
    mapping = []
    for source_id, target_id in zip(source_ids, targets):
        layer = parse_mlp(target_id)
        vector = payload["vectors"][source_id]
        new_components.append({
            "component_id": target_id,
            "layer_idx": layer,
            "component_type": "mlp",
            "source_component_id": source_id,
            "site_control_kind": control,
        })
        new_vectors[target_id] = vector.detach().clone()
        mapping.append({"source": source_id, "target": target_id})
    payload["components"] = new_components
    payload["vectors"] = new_vectors
    payload["site_control"] = {
        "axis": axis,
        "control": control,
        "mode": "same_vector_remap",
        "mapping": mapping,
        "zeroed_original_sites": orig[axis]["mlp_ids"],
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, dst)

def remap_head_payload(src: Path, targets: list[str], dst: Path, *, axis: str, control: str) -> None:
    payload = torch.load(src, map_location="cpu")
    source_ids = [row["head_id"] for row in payload["heads"]]
    if len(source_ids) != len(targets):
        raise RuntimeError(f"Head source/target length mismatch for {axis}/{control}")
    new_heads = []
    new_vectors = {}
    mapping = []
    for source_id, target_id in zip(source_ids, targets):
        layer, head = parse_head(target_id)
        vector = payload["vectors"][source_id]
        new_heads.append({
            "head_id": target_id,
            "layer_idx": layer,
            "head_idx": head,
            "head_dim": HEAD_DIM,
            "num_heads": NUM_HEADS,
            "source_head_id": source_id,
            "site_control_kind": control,
        })
        new_vectors[target_id] = vector.detach().clone()
        mapping.append({"source": source_id, "target": target_id})
    payload["heads"] = new_heads
    payload["vectors"] = new_vectors
    payload["site_control"] = {
        "axis": axis,
        "control": control,
        "mode": "same_vector_remap",
        "mapping": mapping,
        "zeroed_original_heads": orig[axis]["heads"],
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, dst)

manifest = {
    "source_run": str(root),
    "same_site_zero_run": str(zero_out),
    "scan_path": str(scan_path),
    "head_scan_path": str(head_scan_path),
    "selection_rule": {
        "shifted": "Deterministic nearby-layer remap, blocking all original selected sites.",
        "random": "Seeded sample from healthy non-selected scan candidates.",
        "highact_lowrcm": "Healthy non-selected candidates with above-median target active mean, ranked by low target mass then high active mean.",
    },
    "axes": {},
}
env = {}

for axis, cfg in orig.items():
    short = cfg["short"]
    target = cfg["target"]
    direction = target
    orig_mlp_ids = cfg["mlp_ids"]
    orig_heads = cfg["heads"]
    blocked_mlp = set(all_selected_mlp)
    blocked_heads = set(all_selected_heads)

    shifted_mlp_ids = shifted_mlp(orig_mlp_ids, blocked_mlp)
    shifted_head_ids = shifted_heads(orig_heads, blocked_heads)
    blocked_mlp_for_candidates = blocked_mlp | set(shifted_mlp_ids)
    blocked_heads_for_candidates = blocked_heads | set(shifted_head_ids)
    mlp_candidates = valid_mlp_candidates(blocked_mlp_for_candidates)
    head_candidates = valid_head_candidates(blocked_heads_for_candidates)
    random_mlp_ids = choose_random(mlp_candidates, len(orig_mlp_ids), seed=20260628 + (0 if axis == "positive" else 100))
    random_head_ids = choose_random(head_candidates, len(orig_heads), seed=20260628 + (10 if axis == "positive" else 110))
    blocked_mlp_for_hal = blocked_mlp_for_candidates | set(random_mlp_ids)
    blocked_heads_for_hal = blocked_heads_for_candidates | set(random_head_ids)
    hal_mlp_ids = choose_highact_lowrcm(valid_mlp_candidates(blocked_mlp_for_hal), target=target, k=len(orig_mlp_ids))
    hal_head_ids = choose_highact_lowrcm(valid_head_candidates(blocked_heads_for_hal), target=target, k=len(orig_heads))

    controls = {
        "shifted": {"mlp": shifted_mlp_ids, "heads": shifted_head_ids},
        "random": {"mlp": random_mlp_ids, "heads": random_head_ids},
        "highact_lowrcm": {"mlp": hal_mlp_ids, "heads": hal_head_ids},
    }
    manifest["axes"][axis] = {
        "original_mlp": orig_mlp_ids,
        "original_heads": orig_heads,
        "controls": controls,
    }
    env[f"{axis.upper()}_MLP_IDS"] = ",".join(orig_mlp_ids)
    env[f"{axis.upper()}_HEADS"] = ",".join(orig_heads)
    env[f"{axis.upper()}_SHORT"] = short

    for control, sites in controls.items():
        mlp_csv = spec_dir / f"mlp_{axis}_{control}_components.csv"
        head_csv = spec_dir / f"head_{axis}_{control}_heads.csv"
        head_txt = spec_dir / f"head_{axis}_{control}_heads.txt"
        mlp_rows = [
            mlp_row_for(component_id, rank=idx + 1, group_name=f"mlp_{axis}_{control}", direction=direction, control=control)
            for idx, component_id in enumerate(sites["mlp"])
        ]
        head_rows = [
            head_row_for(head_id, rank=idx + 1, group_name=f"head_{axis}_{control}", direction=direction, control=control)
            for idx, head_id in enumerate(sites["heads"])
        ]
        write_csv(mlp_csv, mlp_rows)
        write_csv(head_csv, head_rows)
        write_head_txt(head_txt, sites["heads"])
        mlp_payload = payload_dir / f"{axis}_{control}_same_vector" / "fixed_actuator.pt"
        head_payload = payload_dir / f"{axis}_{control}_same_vector" / "head_actuator.pt"
        remap_mlp_payload(cfg["mlp_payload"], sites["mlp"], mlp_payload, axis=axis, control=control)
        remap_head_payload(cfg["head_payload"], sites["heads"], head_payload, axis=axis, control=control)
        prefix = f"{axis.upper()}_{control.upper()}"
        env[f"{prefix}_MLP_CSV"] = str(mlp_csv)
        env[f"{prefix}_HEADS"] = ",".join(sites["heads"])
        env[f"{prefix}_HEAD_CSV"] = str(head_csv)
        env[f"{prefix}_SAME_MLP_PAYLOAD"] = str(mlp_payload)
        env[f"{prefix}_SAME_HEAD_PAYLOAD"] = str(head_payload)

(out / "site_control_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
with (spec_dir / "site_env.sh").open("w", encoding="utf-8") as fenv:
    for key in sorted(env):
        fenv.write(f"{key}={shlex.quote(env[key])}\n")
print(json.dumps(manifest["axes"], indent=2), flush=True)
PY
  # shellcheck disable=SC1090
  source "$OUT/site_specs/site_env.sh"
  stage generate_site_specs done
}

run_train_mlp_site() {
  local axis="$1"
  local control="$2"
  local components_csv="$3"
  local zero_components="$4"
  local out_dir="$OUT/train/mlp_${axis}_${control}_retrained_512"
  if [[ -f "$out_dir/fixed_actuator.pt" ]]; then
    stage "train_mlp_${axis}_${control}" skipped
    return
  fi
  stage "train_mlp_${axis}_${control}" running
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
    --apply-mode prefill --causal-train-mask --score-mode avglogp --option-selection-mode model_max \
    --alpha-sweep "$ALPHA_SWEEP" \
    --zero-components "$zero_components" --zero-component-apply-mode prefill \
    --device cuda --torch-dtype bfloat16 \
    --out-dir "$out_dir" >"$out_dir/train.log" 2>&1
  stage "train_mlp_${axis}_${control}" done
}

run_train_head_site() {
  local axis="$1"
  local control="$2"
  local heads="$3"
  local zero_heads="$4"
  local out_dir="$OUT/train/head_${axis}_${control}_retrained_512"
  if [[ -f "$out_dir/head_actuator.pt" ]]; then
    stage "train_head_${axis}_${control}" skipped
    return
  fi
  stage "train_head_${axis}_${control}" running
  mkdir -p "$out_dir"
  "$PY" -m screscomp.cli.cecm_train_attention_head_actuator \
    --model "$MODEL" \
    --pairs-csv "$ROOT/pairs/pairs.csv" \
    --event imdb_positive_sentiment \
    --endpoint-objective pair_margin \
    --heads "$heads" \
    --train-split train --val-split val \
    --max-train-rows "$TRAIN_ROWS" --max-val-rows "$VAL_ROWS" \
    --epochs 2 --train-batch-size 32 \
    --lr 0.05 --lambda-norm 0.0001 --alpha-train 1.0 \
    --preference-loss-mode dpo --dpo-beta 1.0 \
    --state-margin-weight 0.0 --gain-weight 1.0 --target-margin 0.0 --target-gain 0.0 \
    --apply-mode all --causal-train-mask --score-mode avglogp --option-selection-mode model_max \
    --alpha-sweep "$ALPHA_SWEEP" \
    --zero-heads "$zero_heads" --zero-head-apply-mode all \
    --device cuda --torch-dtype bfloat16 \
    --out-dir "$out_dir" >"$out_dir/train.log" 2>&1
  stage "train_head_${axis}_${control}" done
}

run_full_site_group() {
  local axis="$1"
  local axis_short="$2"
  local group="$3"
  local zero_components="$4"
  local zero_heads="$5"
  local mlp_payload="$6"
  local head_payload="$7"
  local base_name="full_${axis_short}_zero_base_512"
  local base_dir="$ZERO_OUT/evaluation/val_select/$base_name"
  local out_dir="$OUT/evaluation/site_controls/$group"
  if group_done "$out_dir" 11; then
    stage "eval_$group" skipped
    return
  fi
  if [[ ! -f "$base_dir/generations.jsonl" ]]; then
    log "missing base generations: $base_dir/generations.jsonl"
    exit 1
  fi
  stage "eval_$group" running
  mkdir -p "$out_dir"
  "$PY" -m screscomp.cli.run_imdb_sentiment_actuator_generation \
    --model "$MODEL" \
    --prompts-jsonl "$ROOT/eval_env/val/prompts.jsonl" \
    --out-jsonl "$out_dir/nonzero_generations.jsonl" \
    --actuator "$mlp_payload" \
    --head-actuator "$head_payload" \
    --control-name "$group" \
    --alpha-sweep "$NONZERO_SWEEP" \
    --generation-apply-mode all --component-apply-mode prefill --head-apply-mode all \
    --zero-components "$zero_components" --zero-component-apply-mode prefill \
    --zero-heads "$zero_heads" --zero-head-apply-mode all \
    --split eval --samples-per-prompt 1 \
    --generation-batch-size "$GEN_BATCH" --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature 1.0 --top-p 1.0 --top-k 50 --seed 42 --same-seed-across-alpha \
    --device cuda --torch-dtype bfloat16 >"$out_dir/generate.log" 2>&1
  materialize_group_generations "$base_dir" "$out_dir/nonzero_generations.jsonl" "$group" "$out_dir/generations.jsonl"
  score_and_kl "$out_dir"
  stage "eval_$group" done
}

run_axis_control() {
  local axis="$1"
  local control="$2"
  local upper_axis="${axis^^}"
  local upper_control="${control^^}"
  local axis_short
  local orig_mlp
  local orig_heads
  local control_mlp_csv
  local control_heads
  local same_mlp_payload
  local same_head_payload
  eval "axis_short=\${${upper_axis}_SHORT}"
  eval "orig_mlp=\${${upper_axis}_MLP_IDS}"
  eval "orig_heads=\${${upper_axis}_HEADS}"
  eval "control_mlp_csv=\${${upper_axis}_${upper_control}_MLP_CSV}"
  eval "control_heads=\${${upper_axis}_${upper_control}_HEADS}"
  eval "same_mlp_payload=\${${upper_axis}_${upper_control}_SAME_MLP_PAYLOAD}"
  eval "same_head_payload=\${${upper_axis}_${upper_control}_SAME_HEAD_PAYLOAD}"

  if contains_csv_item "$MODES" "retrained"; then
    run_train_mlp_site "$axis" "$control" "$control_mlp_csv" "$orig_mlp"
    run_train_head_site "$axis" "$control" "$control_heads" "$orig_heads"
  fi

  if contains_csv_item "$MODES" "same_vector"; then
    run_full_site_group "$axis" "$axis_short" "full_${axis_short}_${control}_samevec_zeroctx_512" \
      "$orig_mlp" "$orig_heads" "$same_mlp_payload" "$same_head_payload"
  fi
  if contains_csv_item "$MODES" "retrained"; then
    run_full_site_group "$axis" "$axis_short" "full_${axis_short}_${control}_retrained_zeroctx_512" \
      "$orig_mlp" "$orig_heads" \
      "$OUT/train/mlp_${axis}_${control}_retrained_512/fixed_actuator.pt" \
      "$OUT/train/head_${axis}_${control}_retrained_512/head_actuator.pt"
  fi
}

summarize_site_controls() {
  stage summarize_site_controls running
  "$PY" -m screscomp.cli.summarize_imdb_sentiment_matrix \
    --root "$OUT/evaluation/site_controls" \
    --out-csv "$OUT/evaluation/site_controls/matrix_score_summary.csv" >"$OUT/evaluation/site_controls/summarize.log" 2>&1
  "$PY" - "$ZERO_OUT" "$OUT" <<'PY'
import csv
import json
import sys
from pathlib import Path

zero_out = Path(sys.argv[1])
out = Path(sys.argv[2])
rows = []
same_site_csv = zero_out / "evaluation" / "val_select" / "matrix_score_summary.csv"
if same_site_csv.exists():
    with same_site_csv.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            group = row.get("group", "") or row.get("control_name", "")
            if group.startswith("full_pos_") or group.startswith("full_neg_"):
                row = dict(row)
                row["site_control_family"] = "same_site_existing"
                rows.append(row)
site_csv = out / "evaluation" / "site_controls" / "matrix_score_summary.csv"
if site_csv.exists():
    with site_csv.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            row = dict(row)
            row["site_control_family"] = "site_control"
            rows.append(row)
dest = out / "evaluation" / "site_controls" / "site_control_with_same_site_rows.csv"
fields = []
for row in rows:
    for key in row:
        if key not in fields:
            fields.append(key)
with dest.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps({"rows": len(rows), "out": str(dest)}), flush=True)
PY
  stage summarize_site_controls done
}

main() {
  stage preflight running
  write_protocol
  test -f "$ROOT/eval_env/val/prompts.jsonl"
  test -f "$ROOT/pairs/pairs.csv"
  test -f "$ZERO_OUT/evaluation/val_select/full_pos_zero_base_512/generations.jsonl"
  test -f "$ZERO_OUT/evaluation/val_select/full_neg_zero_base_512/generations.jsonl"
  "$PY" -m py_compile \
    src/screscomp/cecm/actuator.py \
    src/screscomp/cli/cecm_train_fixed_actuator.py \
    src/screscomp/cli/cecm_train_attention_head_actuator.py \
    src/screscomp/cli/run_imdb_sentiment_actuator_generation.py \
    src/screscomp/cli/score_imdb_sentiment_generations.py \
    src/screscomp/cli/compute_imdb_generation_kl.py \
    src/screscomp/cli/summarize_imdb_sentiment_matrix.py
  stage preflight done

  generate_site_specs
  if [[ "$SITE_SPEC_ONLY" == "1" ]]; then
    stage site_spec_only done
    log "site spec dry run done out=$OUT"
    return
  fi

  IFS=',' read -r -a axis_list <<< "$AXES"
  IFS=',' read -r -a control_list <<< "$CONTROLS"
  for axis in "${axis_list[@]}"; do
    axis="$(echo "$axis" | xargs)"
    [[ -z "$axis" ]] && continue
    for control in "${control_list[@]}"; do
      control="$(echo "$control" | xargs)"
      [[ -z "$control" ]] && continue
      run_axis_control "$axis" "$control"
    done
  done

  summarize_site_controls
  stage complete done
  log "done out=$OUT"
}

main "$@"
