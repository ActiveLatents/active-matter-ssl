#!/usr/bin/env bash
#SBATCH --job-name=hier_search
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

set -euo pipefail

mkdir -p logs

export SIF="${SIF:-/share/apps/images/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif}"
export OVERLAY="${OVERLAY:-/scratch/$USER/my_env/overlay-25GB-500K.ext3:ro}"
export CONDA_ENV="${CONDA_ENV:-am_ssl}"
export ENV_PREFIX="${ENV_PREFIX:-/scratch/$USER/my_env/conda_envs/${CONDA_ENV}}"
export PYTHON_BIN="${PYTHON_BIN:-${ENV_PREFIX}/bin/python}"
export WANDB_PROJECT="${WANDB_PROJECT:-dl_project_sst}"

export SEARCH_MODE="${SEARCH_MODE:-grid}"
export NUM_TRIALS="${NUM_TRIALS:-8}"
export MAX_CONFIGS="${MAX_CONFIGS:-0}"
export EPOCHS="${EPOCHS:-3}"
export VAL_EVERY="${VAL_EVERY:-1}"
export LOG_EVERY="${LOG_EVERY:-50}"
export PROBE_EPOCHS="${PROBE_EPOCHS:-20}"
export PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-256}"
export KNN_K="${KNN_K:-20}"
export RUN_PREFIX="${RUN_PREFIX:-search-hier}"
export RUNS_ROOT="${RUNS_ROOT:-/scratch/$USER/active-matter-ssl/runs/search_hierarchical}"

export WITHIN_MASK_RATIOS="${WITHIN_MASK_RATIOS:-0.85}"
export WEIGHT_DECAYS="${WEIGHT_DECAYS:-0.10}"
export PREDICTOR_DIMS="${PREDICTOR_DIMS:-128}"
export PREDICTOR_DEPTHS="${PREDICTOR_DEPTHS:-2}"
export LAMBDA_WITHIN_VALUES="${LAMBDA_WITHIN_VALUES:-1.0}"
export LAMBDA_CROSS_VALUES="${LAMBDA_CROSS_VALUES:-0.5,1.0}"
export LAMBDA_FUTURE_VALUES="${LAMBDA_FUTURE_VALUES:-0.5,1.0}"
export LAMBDA_SIGREG_VALUES="${LAMBDA_SIGREG_VALUES:-1.0}"
export LAMBDA_ACTION_VALUES="${LAMBDA_ACTION_VALUES:-0.1}"

export TRAIN_SCRIPT="${TRAIN_SCRIPT:-scripts/train_ssl_hierarchical.slurm}"
export EVAL_SCRIPT="${EVAL_SCRIPT:-scripts/evaluate.slurm}"
export TRAIN_ACCOUNT="${TRAIN_ACCOUNT:-torch_pr_494_general}"
export TRAIN_PARTITION="${TRAIN_PARTITION:-}"
export TRAIN_GRES="${TRAIN_GRES:-}"

export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/scratch/$USER/my_env/conda_pkgs}"
export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-/scratch/$USER/my_env/conda_envs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/scratch/$USER/my_env/pip_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/$USER/my_env/.cache}"
export HOME="${HOME:-/scratch/$USER/my_env/home}"
export TMPDIR="${TMPDIR:-/scratch/$USER/my_env/tmp}"

mkdir -p "${CONDA_PKGS_DIRS}" "${CONDA_ENVS_PATH}" "${PIP_CACHE_DIR}" "${XDG_CACHE_HOME}" "${HOME}" "${TMPDIR}"

SEARCH_EXTRA_ARGS=()
if [[ -n "${TRAIN_PARTITION}" ]]; then
  SEARCH_EXTRA_ARGS+=(--partition "${TRAIN_PARTITION}")
fi
if [[ -n "${TRAIN_GRES}" ]]; then
  SEARCH_EXTRA_ARGS+=(--gres "${TRAIN_GRES}")
fi

singularity exec --overlay "${OVERLAY}" "${SIF}" /bin/bash -lc "
  set -euo pipefail
  source /ext3/env.sh
  export HOME='${HOME}'
  export TMPDIR='${TMPDIR}'
  export CONDA_PKGS_DIRS='${CONDA_PKGS_DIRS}'
  export CONDA_ENVS_PATH='${CONDA_ENVS_PATH}'
  export PIP_CACHE_DIR='${PIP_CACHE_DIR}'
  export XDG_CACHE_HOME='${XDG_CACHE_HOME}'
  export CONDA_NO_PLUGINS=true
  cd /scratch/\$USER/active-matter-ssl
  conda activate '${ENV_PREFIX}'

  ${PYTHON_BIN} scripts/submit_hierarchical_tuning.py \
    --submit \
    --account ${TRAIN_ACCOUNT} \
    --search-mode ${SEARCH_MODE} \
    --num-trials ${NUM_TRIALS} \
    --max-configs ${MAX_CONFIGS} \
    --epochs ${EPOCHS} \
    --val-every ${VAL_EVERY} \
    --log-every ${LOG_EVERY} \
    --probe-epochs ${PROBE_EPOCHS} \
    --probe-batch-size ${PROBE_BATCH_SIZE} \
    --knn-k ${KNN_K} \
    --run-prefix ${RUN_PREFIX} \
    --runs-root ${RUNS_ROOT} \
    --within-mask-ratios ${WITHIN_MASK_RATIOS} \
    --weight-decays ${WEIGHT_DECAYS} \
    --predictor-dims ${PREDICTOR_DIMS} \
    --predictor-depths ${PREDICTOR_DEPTHS} \
    --lambda-within-values ${LAMBDA_WITHIN_VALUES} \
    --lambda-cross-values ${LAMBDA_CROSS_VALUES} \
    --lambda-future-values ${LAMBDA_FUTURE_VALUES} \
    --lambda-sigreg-values ${LAMBDA_SIGREG_VALUES} \
    --lambda-action-values ${LAMBDA_ACTION_VALUES} \
    --train-script ${TRAIN_SCRIPT} \
    --eval-script ${EVAL_SCRIPT} \
    ${SEARCH_EXTRA_ARGS[*]}
"
