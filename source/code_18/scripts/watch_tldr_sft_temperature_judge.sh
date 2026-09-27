#!/usr/bin/env bash
set -u

cd LOCAL_HOME/RPEC/projects/screscomp

OUT_ROOT="${OUT_ROOT:-runs/tldr_gptj_dpo_20260608_163146}"
SWEEP_ROOT="${SWEEP_ROOT:-$OUT_ROOT/open_test/sft_temp_sweep}"
TEMPS="${TEMPS:-0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0}"
EXPECTED_ROWS="${TEST_MAX_ROWS:-320}"
POLL_SECONDS="${POLL_SECONDS:-120}"

temp_tag() {
  local temp="$1"
  python - "$temp" <<'PY'
import sys
t = float(sys.argv[1])
if abs(t) < 1e-9:
    print("temp0")
elif abs(t - round(t)) < 1e-9:
    print(f"temp{int(round(t))}")
else:
    print(("temp%.1f" % t).replace(".", "p"))
PY
}

sft_path_for_temp() {
  local temp="$1"
  local tag
  tag="$(temp_tag "$temp")"
  if [[ "$tag" == "temp0p7" ]]; then
    echo "$OUT_ROOT/open_test/base/generations.jsonl"
  else
    echo "$SWEEP_ROOT/sft_${tag}/generations.jsonl"
  fi
}

generation_done() {
  local temp="$1"
  local path lines
  path="$(sft_path_for_temp "$temp")"
  [[ -s "$path" ]] || return 1
  lines="$(grep -cve '^$' "$path" 2>/dev/null || echo 0)"
  [[ "$lines" -eq "$EXPECTED_ROWS" ]]
}

judge_done() {
  local temp="$1"
  local tag
  tag="$(temp_tag "$temp")"
  [[ -s "$SWEEP_ROOT/ds4_sft_${tag}_fullswap/pairwise_summary.csv" ]]
}

while true; do
  remaining=0
  for temp in $TEMPS; do
    if judge_done "$temp"; then
      continue
    fi
    remaining=$((remaining + 1))
    if generation_done "$temp"; then
      echo "[sft-watch] judge temp=$temp"
      RUN_GENERATION=0 RUN_JUDGE=1 TEMPS="$temp" bash scripts/run_tldr_sft_temperature_sweep.sh || true
    fi
  done
  if [[ "$remaining" -eq 0 ]]; then
    echo "[sft-watch] done"
    exit 0
  fi
  sleep "$POLL_SECONDS"
done
