#!/usr/bin/env bash
set -euo pipefail

# Pre-O head-channel site protocol.
#
# This is the clean head-site rerun:
#   - true head actuators: Lx.attn.hj, trained in the head_dim channel before O.
#   - no same-vector transfer.
#   - selected heads are retrained by default with the same budget as controls.
#   - shift1/shift2/shift3 controls are retrained against the same historical
#     pair files and training budget.
#   - optional random-head stress controls can be enabled with
#     RANDOM_HEAD_CONTROLS="same_layer_random_retrained same_head_outside_band_retrained outside_band_random_retrained".
#   - IMDb trains positive and negative head roles separately; generation combines them.
#   - ConFiQA trains suppress and boost head roles separately; generation combines them.

export LC_ALL=C.UTF-8
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

REPO_ROOT="${REPO_ROOT:-LOCAL_HOME/RPEC/projects/screscomp}"
cd "$REPO_ROOT"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

PY="${PY:-LOCAL_HOME/anaconda3/envs/screscomp/bin/python}"
RUN_ROOT="${RUN_ROOT:-/dev/shm/screscomp_runs/pre_o_head_site_shift_$(date -u +%Y%m%d_%H%M%S)}"
GPU="${GPU:-0}"
TASKS="${TASKS:-imdb qa mr mc}"
SHIFTS="${SHIFTS:-1 2 3}"
ALPHA_SWEEP="${ALPHA_SWEEP:-0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}"
SEED="${SEED:-20260702}"
RANDOM_HEAD_CONTROLS="${RANDOM_HEAD_CONTROLS:-}"
TRAIN_SHIFT_CONTROLS="${TRAIN_SHIFT_CONTROLS:-1}"
RETRAIN_SELECTED="${RETRAIN_SELECTED:-1}"

IMDB_ROOT="${IMDB_ROOT:-/dev/shm/screscomp_runs/imdb_ma921_expanded1024_sampled_paperkl_20260626_055222}"
IMDB_MODEL="${IMDB_MODEL:-LOCAL_HOME/.cache/huggingface/hub/models--ma921--gpt2-large-sft-imdb/snapshots/f480190690d5abfc0e003ccb4f7e650626019bb9}"
IMDB_N_LAYERS="${IMDB_N_LAYERS:-36}"
IMDB_N_HEADS="${IMDB_N_HEADS:-20}"
IMDB_POS_HEADS="${IMDB_POS_HEADS:-L16.attn.h13,L15.attn.h14,L0.attn.h5,L9.attn.h6}"
IMDB_NEG_HEADS="${IMDB_NEG_HEADS:-L14.attn.h5,L16.attn.h17,L16.attn.h4,L7.attn.h14}"
IMDB_SELECTED_POS_OUT="${IMDB_SELECTED_POS_OUT:-$IMDB_ROOT/train/train_head_positive}"
IMDB_SELECTED_NEG_OUT="${IMDB_SELECTED_NEG_OUT:-$IMDB_ROOT/train/train_head_negative}"
IMDB_TRAIN_ROWS="${IMDB_TRAIN_ROWS:-1024}"
IMDB_VAL_ROWS="${IMDB_VAL_ROWS:-256}"
IMDB_EPOCHS="${IMDB_EPOCHS:-2}"
IMDB_TRAIN_BATCH_SIZE="${IMDB_TRAIN_BATCH_SIZE:-32}"
IMDB_HEAD_APPLY_MODE="${IMDB_HEAD_APPLY_MODE:-all}"

CONFIQA_MODEL="${CONFIQA_MODEL:-LOCAL_HOME/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct}"
CONFIQA_N_LAYERS="${CONFIQA_N_LAYERS:-32}"
CONFIQA_N_HEADS="${CONFIQA_N_HEADS:-32}"
FETCH_DIR="${FETCH_DIR:-data_ckplug/context_dpo_confiqa_official}"
DISCOVERY_BASE="${DISCOVERY_BASE:-data_ckplug/cast_confiqa_all3_disc120_heldout500_v0}"
DISCOVERY_QA="${DISCOVERY_QA:-$DISCOVERY_BASE/qa}"
DISCOVERY_MR="${DISCOVERY_MR:-$DISCOVERY_BASE/mr}"
DISCOVERY_MC="${DISCOVERY_MC:-$DISCOVERY_BASE/mc}"
CONFIQA_TRAIN_SOURCE_ROWS="${CONFIQA_TRAIN_SOURCE_ROWS:-300}"
CONFIQA_VAL_MOD="${CONFIQA_VAL_MOD:-5}"
CONFIQA_TRAIN_ROWS="${CONFIQA_TRAIN_ROWS:-240}"
CONFIQA_VAL_ROWS="${CONFIQA_VAL_ROWS:-60}"
CONFIQA_EPOCHS="${CONFIQA_EPOCHS:-2}"
CONFIQA_TRAIN_BATCH_SIZE="${CONFIQA_TRAIN_BATCH_SIZE:-4}"
CONFIQA_HEAD_APPLY_MODE="${CONFIQA_HEAD_APPLY_MODE:-decision_tokens}"

export CUDA_VISIBLE_DEVICES="$GPU"

mkdir -p "$RUN_ROOT/logs"
STATUS="$RUN_ROOT/status.tsv"
MASTER_LOG="$RUN_ROOT/pre_o_head_site_shift.log"
SUMMARY="$RUN_ROOT/head_train_summary.tsv"
printf 'time\tstage\tstatus\n' > "$STATUS"
printf 'task\trole\tcontrol\theads\tout_dir\n' > "$SUMMARY"

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

discovery_root_for_task() {
  case "$1" in
    qa) echo "$DISCOVERY_QA" ;;
    mr) echo "$DISCOVERY_MR" ;;
    mc) echo "$DISCOVERY_MC" ;;
    *) echo "unsupported ConFiQA task: $1" >&2; exit 2 ;;
  esac
}

selected_env_for_task() {
  local root
  root="$(discovery_root_for_task "$1")"
  echo "$root/discovery/selected/selected.env"
}

prepare_confiqa_common() {
  local short="$1"
  local task_out="$2"
  local common="$task_out/common"
  local pairs_dir="$common/pairs"
  local train_open_rows="$common/train_open_rows.jsonl"
  local data_json
  data_json="$(data_json_for_task "$short")"
  mkdir -p "$common"
  if [[ -f "$pairs_dir/pairs.csv" ]]; then
    return
  fi
  stage "prepare_${short}_pairs" running
  "$PY" -m screscomp.cli.cecm_fetch_confiqa --out-dir "$FETCH_DIR" --tasks "$short" \
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
  stage "prepare_${short}_pairs" done
}

write_protocol() {
  "$PY" - "$RUN_ROOT/protocol.json" <<'PY'
import json, os, sys
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "experiment": "pre_o_head_site_shift",
    "intervention_object": "attention head channel before the O projection (Lx.attn.hj)",
    "controls": [
        "selected retrained with the same site protocol and same budget by default",
        "shift1 retrained",
        "shift2 retrained",
        "shift3 retrained",
        "optional same_layer_random_retrained",
        "optional same_head_outside_band_retrained",
        "optional outside_band_random_retrained",
    ],
    "omitted_controls": {
        "same_vector": "not run in this protocol",
        "full": "no MLP+head combination; only head-role combinations are generated later",
        "selected_external": "only enabled when RETRAIN_SELECTED=0; historical/provenance control, not the clean mechanism comparison",
    },
    "fairness_invariant": (
        "For mechanism rows, the only manipulated variable is the write interface. "
        "Selected, shifted, and random controls share the same pair file, train/val caps, "
        "optimizer budget, objective, alpha grid, role composition, and apply mode."
    ),
    "retrain_selected": os.environ.get("RETRAIN_SELECTED", "1"),
    "imdb_head_apply_mode": os.environ.get("IMDB_HEAD_APPLY_MODE", "all"),
    "confiqa_head_apply_mode": os.environ.get("CONFIQA_HEAD_APPLY_MODE", "decision_tokens"),
    "shift_rule": (
        "For each selected head, sample +d or -d with a fixed seed. If the sampled side is "
        "out of range or collides with the original/target pool, try the opposite side. "
        "If both sides are unavailable, keep the original site and record fallback=original_site_allowed."
    ),
    "random_head_controls": {
        "enabled": os.environ.get("RANDOM_HEAD_CONTROLS", ""),
        "train_shift_controls": os.environ.get("TRAIN_SHIFT_CONTROLS", "1"),
        "same_layer_random_retrained": (
            "For each selected head, keep the selected layer fixed and sample a different "
            "head index from the same layer. This tests whether head identity matters once "
            "the RCM-selected layer/interface has been fixed."
        ),
        "same_head_outside_band_retrained": (
            "For each selected head, keep the selected head index fixed and sample a layer "
            "outside the selected-layer plus/minus max-shift neighborhood. This tests whether "
            "the same head channel remains usable outside the RCM interface band."
        ),
        "outside_band_random_retrained": (
            "Sample the same number of heads outside the selected-layer neighborhood, where "
            "the excluded neighborhood is every selected layer plus/minus the maximum shift "
            "distance configured for this run. This tests whether useful head layers are "
            "effectively unrestricted or concentrated near the RCM interface band."
        ),
    },
    "imdb": {
        "roles": ["positive", "negative"],
        "role_training": "separate head actuators; later combined during generation",
        "selected_positive_out": os.environ.get("IMDB_SELECTED_POS_OUT", ""),
        "selected_negative_out": os.environ.get("IMDB_SELECTED_NEG_OUT", ""),
        "train_rows": int(os.environ.get("IMDB_TRAIN_ROWS", "1024")),
        "val_rows": int(os.environ.get("IMDB_VAL_ROWS", "256")),
        "pair_source": os.environ.get("IMDB_ROOT", ""),
    },
    "confiqa": {
        "roles": ["suppress", "boost"],
        "role_training": "separate head actuators; later combined during generation",
        "train_source_rows": int(os.environ.get("CONFIQA_TRAIN_SOURCE_ROWS", "300")),
        "val_mod": int(os.environ.get("CONFIQA_VAL_MOD", "5")),
        "train_rows": int(os.environ.get("CONFIQA_TRAIN_ROWS", "240")),
        "val_rows": int(os.environ.get("CONFIQA_VAL_ROWS", "60")),
    },
    "alpha_sweep": os.environ.get("ALPHA_SWEEP", ""),
    "seed": int(os.environ.get("SEED", "0")),
}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

generate_head_specs() {
  local task="$1"
  local task_out="$2"
  local n_layers="$3"
  local n_heads="$4"
  local role_a="$5"
  local heads_a="$6"
  local role_b="$7"
  local heads_b="$8"
  local out_dir="$task_out/site_specs/head_pre_o"
  mkdir -p "$out_dir"
  TASK_NAME="$task" OUT_DIR="$out_dir" N_LAYERS="$n_layers" N_HEADS="$n_heads" SHIFTS_VALUE="$SHIFTS" \
  RANDOM_HEAD_CONTROLS_VALUE="$RANDOM_HEAD_CONTROLS" SEED_VALUE="$SEED" \
  ROLE_A="$role_a" HEADS_A="$heads_a" ROLE_B="$role_b" HEADS_B="$heads_b" "$PY" - <<'PY'
import hashlib
import json
import os
import random
import re
from pathlib import Path

task = os.environ["TASK_NAME"]
out_dir = Path(os.environ["OUT_DIR"])
n_layers = int(os.environ["N_LAYERS"])
n_heads = int(os.environ["N_HEADS"])
seed_base = int(os.environ["SEED_VALUE"])
shifts = [int(x) for x in os.environ["SHIFTS_VALUE"].split() if x.strip()]
random_controls = [x for x in os.environ["RANDOM_HEAD_CONTROLS_VALUE"].split() if x.strip()]
roles = [
    (os.environ["ROLE_A"], [x for x in os.environ["HEADS_A"].split(",") if x]),
    (os.environ["ROLE_B"], [x for x in os.environ["HEADS_B"].split(",") if x]),
]
valid_random_controls = {
    "same_layer_random_retrained",
    "same_head_outside_band_retrained",
    "outside_band_random_retrained",
}
unknown = sorted(set(random_controls) - valid_random_controls)
if unknown:
    raise SystemExit(f"Unknown RANDOM_HEAD_CONTROLS: {unknown}")
head_pat = re.compile(r"^L(\d+)\.attn\.h(\d+)$")

def parse_head(head_id: str) -> tuple[int, int]:
    m = head_pat.fullmatch(head_id)
    if not m:
        raise SystemExit(f"Bad head id: {head_id}")
    return int(m.group(1)), int(m.group(2))

def stable_seed(*parts: object) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return seed_base + int(h[:8], 16)

def write_heads(path: Path, heads: list[str]) -> None:
    path.write_text(",".join(heads) + "\n", encoding="utf-8")

def head_id(layer: int, head_idx: int) -> str:
    return f"L{layer}.attn.h{head_idx}"

def selected_layers() -> set[int]:
    return {parse_head(head)[0] for _, heads in roles for head in heads}

def max_shift_distance() -> int:
    return max(shifts) if shifts else 0

def shifted_all(distance: int):
    rng = random.Random(stable_seed(task, "pre_o_head", distance))
    original_ids = {h for _, heads in roles for h in heads}
    used_targets: set[str] = set()
    role_targets: dict[str, list[str]] = {}
    decisions: list[dict] = []
    for role, heads in roles:
        targets = []
        for idx, source_id in enumerate(heads):
            source_layer, head_idx = parse_head(source_id)
            first_sign = rng.choice([1, -1])
            offsets = [first_sign * distance, -first_sign * distance]
            chosen_id = None
            chosen_offset = None
            rejected = []
            for offset in offsets:
                target_layer = source_layer + offset
                target_id = f"L{target_layer}.attn.h{head_idx}"
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
            target_layer, target_head = parse_head(chosen_id)
            targets.append(chosen_id)
            decisions.append({
                "role": role,
                "source_head_id": source_id,
                "target_head_id": chosen_id,
                "source_layer_idx": source_layer,
                "target_layer_idx": target_layer,
                "head_idx": target_head,
                "sampled_first_offset": offsets[0],
                "chosen_offset": chosen_offset,
                "fallback": fallback,
                "rejected": rejected,
                "target_order": idx + 1,
            })
        role_targets[role] = targets
    return role_targets, decisions

def same_layer_random_all():
    rng = random.Random(stable_seed(task, "same_layer_random_retrained"))
    original_ids = {h for _, heads in roles for h in heads}
    used_targets: set[str] = set()
    role_targets: dict[str, list[str]] = {}
    decisions: list[dict] = []
    for role, heads in roles:
        targets = []
        for idx, source_id in enumerate(heads):
            source_layer, source_head = parse_head(source_id)
            candidates = [
                head_id(source_layer, h)
                for h in range(n_heads)
                if head_id(source_layer, h) not in original_ids and head_id(source_layer, h) not in used_targets
            ]
            rng.shuffle(candidates)
            fallback = ""
            if candidates:
                chosen_id = candidates[0]
            else:
                chosen_id = source_id
                fallback = "original_site_allowed_no_same_layer_candidate"
            used_targets.add(chosen_id)
            target_layer, target_head = parse_head(chosen_id)
            targets.append(chosen_id)
            decisions.append({
                "role": role,
                "source_head_id": source_id,
                "target_head_id": chosen_id,
                "source_layer_idx": source_layer,
                "target_layer_idx": target_layer,
                "source_head_idx": source_head,
                "target_head_idx": target_head,
                "fallback": fallback,
                "target_order": idx + 1,
                "candidate_rule": "same selected layer, random non-selected head index",
            })
        role_targets[role] = targets
    return role_targets, decisions

def outside_band_random_all():
    rng = random.Random(stable_seed(task, "outside_band_random_retrained"))
    original_ids = {h for _, heads in roles for h in heads}
    selected = selected_layers()
    width = max_shift_distance()
    excluded_layers = {
        layer + offset
        for layer in selected
        for offset in range(-width, width + 1)
        if 0 <= layer + offset < n_layers
    }
    candidate_layers = [layer for layer in range(n_layers) if layer not in excluded_layers]
    fallback_rule = ""
    if not candidate_layers:
        candidate_layers = [layer for layer in range(n_layers) if layer not in selected]
        fallback_rule = "no_outside_neighborhood_layers_available_used_non_selected_layers"
    used_targets: set[str] = set()
    role_targets: dict[str, list[str]] = {}
    decisions: list[dict] = []
    for role, heads in roles:
        targets = []
        for idx, source_id in enumerate(heads):
            source_layer, source_head = parse_head(source_id)
            candidates = [
                head_id(layer, h)
                for layer in candidate_layers
                for h in range(n_heads)
                if head_id(layer, h) not in original_ids and head_id(layer, h) not in used_targets
            ]
            rng.shuffle(candidates)
            fallback = fallback_rule
            if candidates:
                chosen_id = candidates[0]
            else:
                chosen_id = source_id
                fallback = "original_site_allowed_no_outside_band_candidate"
            used_targets.add(chosen_id)
            target_layer, target_head = parse_head(chosen_id)
            targets.append(chosen_id)
            decisions.append({
                "role": role,
                "source_head_id": source_id,
                "target_head_id": chosen_id,
                "source_layer_idx": source_layer,
                "target_layer_idx": target_layer,
                "source_head_idx": source_head,
                "target_head_idx": target_head,
                "fallback": fallback,
                "target_order": idx + 1,
                "candidate_rule": "random head outside selected-layer +/- max_shift neighborhood",
                "excluded_layers": sorted(excluded_layers),
                "candidate_layers": candidate_layers,
                "max_shift_distance": width,
            })
        role_targets[role] = targets
    meta = {
        "selected_layers": sorted(selected),
        "max_shift_distance": width,
        "excluded_layers": sorted(excluded_layers),
        "candidate_layers": candidate_layers,
        "fallback_rule": fallback_rule,
    }
    return role_targets, decisions, meta

def same_head_outside_band_all():
    rng = random.Random(stable_seed(task, "same_head_outside_band_retrained"))
    original_ids = {h for _, heads in roles for h in heads}
    selected = selected_layers()
    width = max_shift_distance()
    excluded_layers = {
        layer + offset
        for layer in selected
        for offset in range(-width, width + 1)
        if 0 <= layer + offset < n_layers
    }
    candidate_layers = [layer for layer in range(n_layers) if layer not in excluded_layers]
    fallback_rule = ""
    if not candidate_layers:
        candidate_layers = [layer for layer in range(n_layers) if layer not in selected]
        fallback_rule = "no_outside_neighborhood_layers_available_used_non_selected_layers"
    used_targets: set[str] = set()
    role_targets: dict[str, list[str]] = {}
    decisions: list[dict] = []
    for role, heads in roles:
        targets = []
        for idx, source_id in enumerate(heads):
            source_layer, source_head = parse_head(source_id)
            candidates = [
                head_id(layer, source_head)
                for layer in candidate_layers
                if head_id(layer, source_head) not in original_ids and head_id(layer, source_head) not in used_targets
            ]
            rng.shuffle(candidates)
            fallback = fallback_rule
            if candidates:
                chosen_id = candidates[0]
            else:
                chosen_id = source_id
                fallback = "original_site_allowed_no_same_head_outside_band_candidate"
            used_targets.add(chosen_id)
            target_layer, target_head = parse_head(chosen_id)
            targets.append(chosen_id)
            decisions.append({
                "role": role,
                "source_head_id": source_id,
                "target_head_id": chosen_id,
                "source_layer_idx": source_layer,
                "target_layer_idx": target_layer,
                "source_head_idx": source_head,
                "target_head_idx": target_head,
                "fallback": fallback,
                "target_order": idx + 1,
                "candidate_rule": "same selected head index, random layer outside selected-layer +/- max_shift neighborhood",
                "excluded_layers": sorted(excluded_layers),
                "candidate_layers": candidate_layers,
                "max_shift_distance": width,
            })
        role_targets[role] = targets
    meta = {
        "selected_layers": sorted(selected),
        "max_shift_distance": width,
        "excluded_layers": sorted(excluded_layers),
        "candidate_layers": candidate_layers,
        "fallback_rule": fallback_rule,
    }
    return role_targets, decisions, meta

manifest = {
    "task": task,
    "n_layers": n_layers,
    "n_heads": n_heads,
    "selected": {},
    "shift_controls": {},
    "random_head_controls": {},
}
for role, heads in roles:
    write_heads(out_dir / f"selected_{role}_heads.txt", heads)
    manifest["selected"][role] = heads

for distance in shifts:
    targets_by_role, decisions = shifted_all(distance)
    control = f"shift{distance}"
    manifest["shift_controls"][control] = {"roles": targets_by_role, "decisions": decisions}
    for role, heads in targets_by_role.items():
        write_heads(out_dir / f"{control}_{role}_heads.txt", heads)

if "same_layer_random_retrained" in random_controls:
    targets_by_role, decisions = same_layer_random_all()
    control = "same_layer_random_retrained"
    manifest["random_head_controls"][control] = {"roles": targets_by_role, "decisions": decisions}
    for role, heads in targets_by_role.items():
        write_heads(out_dir / f"{control}_{role}_heads.txt", heads)

if "same_head_outside_band_retrained" in random_controls:
    targets_by_role, decisions, meta = same_head_outside_band_all()
    control = "same_head_outside_band_retrained"
    manifest["random_head_controls"][control] = {"roles": targets_by_role, "decisions": decisions, "meta": meta}
    for role, heads in targets_by_role.items():
        write_heads(out_dir / f"{control}_{role}_heads.txt", heads)

if "outside_band_random_retrained" in random_controls:
    targets_by_role, decisions, meta = outside_band_random_all()
    control = "outside_band_random_retrained"
    manifest["random_head_controls"][control] = {"roles": targets_by_role, "decisions": decisions, "meta": meta}
    for role, heads in targets_by_role.items():
        write_heads(out_dir / f"{control}_{role}_heads.txt", heads)

(out_dir / "site_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
PY
}

run_train_head() {
  local task="$1"
  local role="$2"
  local control="$3"
  local model="$4"
  local pairs_csv="$5"
  local heads="$6"
  local out_dir="$7"
  local event="$8"
  local apply_mode="$9"
  local max_train_rows="${10}"
  local max_val_rows="${11}"
  local epochs="${12}"
  local train_batch_size="${13}"
  local score_mode="${14}"
  local preference_mode="${15}"
  shift 15
  local extra=("$@")

  if [[ -f "$out_dir/head_actuator.pt" ]]; then
    stage "train_${task}_${role}_${control}" reused
  else
    stage "train_${task}_${role}_${control}" running
    mkdir -p "$out_dir"
    "$PY" -m screscomp.cli.cecm_train_attention_head_actuator \
      --model "$model" \
      --pairs-csv "$pairs_csv" \
      --event "$event" \
      --heads "$heads" \
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
    stage "train_${task}_${role}_${control}" done
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' "$task" "$role" "$control" "$heads" "$out_dir" >> "$SUMMARY"
}

register_external_selected_head() {
  local task="$1"
  local role="$2"
  local heads="$3"
  local out_dir="$4"

  test -f "$out_dir/head_actuator.pt"
  stage "selected_${task}_${role}" external
  printf '%s\t%s\t%s\t%s\t%s\n' "$task" "$role" selected "$heads" "$out_dir" >> "$SUMMARY"
}

run_task_roles() {
  local task="$1"
  local task_out="$2"
  local model="$3"
  local pairs_csv="$4"
  local event="$5"
  local apply_mode="$6"
  local max_train_rows="$7"
  local max_val_rows="$8"
  local epochs="$9"
  local train_batch_size="${10}"
  local score_mode="${11}"
  local preference_mode="${12}"
  local role_a="${13}"
  local role_b="${14}"
  local selected_out_a="${15}"
  local selected_out_b="${16}"
  shift 16
  local extra=("$@")

  local specs="$task_out/site_specs/head_pre_o"
  local selected_heads_a selected_heads_b
  selected_heads_a="$(tr -d '\n\r ' < "$specs/selected_${role_a}_heads.txt")"
  selected_heads_b="$(tr -d '\n\r ' < "$specs/selected_${role_b}_heads.txt")"
  if [[ "$RETRAIN_SELECTED" == "1" ]]; then
    run_train_head "$task" "$role_a" selected "$model" "$pairs_csv" "$selected_heads_a" \
      "$task_out/train/head_pre_o/selected/$role_a" "$event" "$apply_mode" \
      "$max_train_rows" "$max_val_rows" "$epochs" "$train_batch_size" "$score_mode" "$preference_mode" \
      "${extra[@]}"
    run_train_head "$task" "$role_b" selected "$model" "$pairs_csv" "$selected_heads_b" \
      "$task_out/train/head_pre_o/selected/$role_b" "$event" "$apply_mode" \
      "$max_train_rows" "$max_val_rows" "$epochs" "$train_batch_size" "$score_mode" "$preference_mode" \
      "${extra[@]}"
  else
    register_external_selected_head "$task" "$role_a" "$selected_heads_a" "$selected_out_a"
    register_external_selected_head "$task" "$role_b" "$selected_heads_b" "$selected_out_b"
  fi

  for role in "$role_a" "$role_b"; do
    local selected_heads
    selected_heads="$(tr -d '\n\r ' < "$specs/selected_${role}_heads.txt")"
    if [[ "$TRAIN_SHIFT_CONTROLS" == "1" ]]; then
      for shift in $SHIFTS; do
        local shifted_heads
        shifted_heads="$(tr -d '\n\r ' < "$specs/shift${shift}_${role}_heads.txt")"
        run_train_head "$task" "$role" "shift${shift}_retrained" "$model" "$pairs_csv" "$shifted_heads" \
          "$task_out/train/head_pre_o/shift${shift}_retrained/$role" "$event" "$apply_mode" \
          "$max_train_rows" "$max_val_rows" "$epochs" "$train_batch_size" "$score_mode" "$preference_mode" \
          "${extra[@]}"
      done
    fi
    for control in $RANDOM_HEAD_CONTROLS; do
      local random_heads_file="$specs/${control}_${role}_heads.txt"
      test -f "$random_heads_file"
      local random_heads
      random_heads="$(tr -d '\n\r ' < "$random_heads_file")"
      run_train_head "$task" "$role" "$control" "$model" "$pairs_csv" "$random_heads" \
        "$task_out/train/head_pre_o/$control/$role" "$event" "$apply_mode" \
        "$max_train_rows" "$max_val_rows" "$epochs" "$train_batch_size" "$score_mode" "$preference_mode" \
        "${extra[@]}"
    done
  done
}

run_imdb() {
  local task_out="$RUN_ROOT/imdb"
  mkdir -p "$task_out"
  local pairs_csv="$IMDB_ROOT/pairs/pairs.csv"
  test -f "$pairs_csv"
  generate_head_specs imdb "$task_out" "$IMDB_N_LAYERS" "$IMDB_N_HEADS" positive "$IMDB_POS_HEADS" negative "$IMDB_NEG_HEADS"
  run_task_roles imdb "$task_out" "$IMDB_MODEL" "$pairs_csv" \
    imdb_positive_sentiment "$IMDB_HEAD_APPLY_MODE" "$IMDB_TRAIN_ROWS" "$IMDB_VAL_ROWS" "$IMDB_EPOCHS" "$IMDB_TRAIN_BATCH_SIZE" \
    avglogp dpo positive negative "$IMDB_SELECTED_POS_OUT" "$IMDB_SELECTED_NEG_OUT" \
    --endpoint-objective pair_margin --dpo-beta 1.0 --causal-train-mask --option-selection-mode model_max
}

run_confiqa_task() {
  local short="$1"
  local task="confiqa_$short"
  local task_out="$RUN_ROOT/$task"
  local source_root
  source_root="$(discovery_root_for_task "$short")"
  mkdir -p "$task_out"
  local selected_env
  selected_env="$(selected_env_for_task "$short")"
  test -f "$selected_env"
  # shellcheck disable=SC1090
  source "$selected_env"
  test -n "${SUPPRESS_HEADS:-}"
  test -n "${BOOST_HEADS:-}"
  local pairs_csv="$source_root/pairs/pairs.csv"
  local selected_suppress_out="$source_root/train_decision_tokens_head_suppress"
  local selected_boost_out="$source_root/train_decision_tokens_head_boost"
  test -f "$pairs_csv"
  generate_head_specs "$task" "$task_out" "$CONFIQA_N_LAYERS" "$CONFIQA_N_HEADS" suppress "$SUPPRESS_HEADS" boost "$BOOST_HEADS"
  run_task_roles "$task" "$task_out" "$CONFIQA_MODEL" "$pairs_csv" \
    source_context_over_prior "$CONFIQA_HEAD_APPLY_MODE" "$CONFIQA_TRAIN_ROWS" "$CONFIQA_VAL_ROWS" "$CONFIQA_EPOCHS" "$CONFIQA_TRAIN_BATCH_SIZE" \
    answer_rest_margin margin_gain suppress boost "$selected_suppress_out" "$selected_boost_out" \
    --endpoint-objective pair_margin --max-aliases-per-side 1
}

stage preflight running
write_protocol
"$PY" -m py_compile \
  src/screscomp/cli/cecm_train_attention_head_actuator.py \
  src/screscomp/cli/prepare_ckplug_open.py \
  src/screscomp/cli/cecm_build_preference_pairs.py
stage preflight done

for task in $TASKS; do
  case "$task" in
    imdb) run_imdb ;;
    qa|mr|mc) run_confiqa_task "$task" ;;
    *) echo "Unknown task: $task" >&2; exit 2 ;;
  esac
done

stage complete done
log "done run_root=$RUN_ROOT"
