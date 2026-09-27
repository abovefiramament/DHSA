#!/bin/bash --login
#SBATCH --job-name=imdb-ext-budget
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0-08:00:00
#SBATCH --partition=ext_intersect
#SBATCH --qos=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --account=a_acu
#SBATCH --export=NONE
#SBATCH --output=LOCAL_HOME/rpec_runs/imdb_external_budget_20260923/logs/%j.out
#SBATCH --error=LOCAL_HOME/rpec_runs/imdb_external_budget_20260923/logs/%j.err
set -Eeuo pipefail
METHOD="$1"
PHASE="$2"
case "$METHOD" in bipo|loreft) ;; *) exit 2;; esac
case "$PHASE" in controller|test) ;; *) exit 2;; esac
module purge
module load python/3.11.3-gcccore-12.3.0
export PYTHONPATH=LOCAL_HOME/rpec_runs/imdb_external_budget_20260923/code_bipo_fix
if test "$METHOD" = loreft; then export PYTHONPATH="$PYTHONPATH:LOCAL_HOME/external/pyreft-dafd0995"; PY=LOCAL_HOME/venvs/rpec_external/bin/python; else PY=LOCAL_HOME/venvs/rpec_bipo/bin/python; fi
export HF_HOME=LOCAL_HOME/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
cd LOCAL_HOME/rpec_runs/imdb_external_budget_20260923/code_bipo_fix
RUN=LOCAL_HOME/rpec_runs/imdb_external_budget_20260923
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,utilization.gpu,memory.used --format=csv > "$RUN/logs/${SLURM_JOB_ID}_gpu_start.csv"
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw --format=csv,noheader,nounits -l 1 > "$RUN/logs/${SLURM_JOB_ID}_${METHOD}_${PHASE}_gpu.csv" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT
"$PY" -u "$RUN/run_budget_bipo_fix.py" "$METHOD" "$PHASE"
