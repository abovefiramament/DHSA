#!/usr/bin/env bash
set -euo pipefail

RUN=/dev/shm/screscomp_runs/site_position_full_20260719_v12
CURRENT_JOB=confiqa__mr__qwen25_14b__rcm_zero_signed__primary

while pgrep -f "17_run_site_job.py.*--job-id ${CURRENT_JOB}" >/dev/null; do
  sleep 60
done

cd LOCAL_HOME/RPEC/projects/screscomp
unset CUDA_VISIBLE_DEVICES
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=src

while IFS= read -r JOB_ID; do
  [ -n "$JOB_ID" ] || continue
  LOCAL_HOME/anaconda3/envs/screscomp/bin/python scripts/paper/17_run_site_job.py \
    --config configs/site_position_full_fixed_protocol_20260719_v12.json \
    --job-id "$JOB_ID" \
    --artifact-root "$RUN" \
    --device cuda:3 \
    --stages prepare,selector,train,dev,test \
    --execution-amendment \
      configs/site_position_v12_qwen_exact_memory_execution_20260731_v2.json
done < "$RUN/protocol/qwen_confiqa_job_ids.txt"
