#!/usr/bin/env bash
#SBATCH --job-name=diffusion_eval
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --requeue
#SBATCH --signal=USR1@120

set -euo pipefail

mkdir -p logs

export SIF="${SIF:-/share/apps/images/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif}"
export OVERLAY="${OVERLAY:-/scratch/$USER/my_env/overlay-25GB-500K.ext3:ro}"
export CONDA_ENV="${CONDA_ENV:-am_ssl}"
export ENV_PREFIX="${ENV_PREFIX:-/scratch/$USER/my_env/conda_envs/${CONDA_ENV}}"
export PYTHON_BIN="${PYTHON_BIN:-${ENV_PREFIX}/bin/python}"
export DATA_DIR="${DATA_DIR:-/scratch/zl6113/active-matter/data}"
export CHECKPOINT="${CHECKPOINT:-/scratch/$USER/active-matter-ssl/runs/diffusion/best_diffusion.pt}"
export RUN_NAME="${RUN_NAME:-diffusion_eval}"
export WANDB_PROJECT="${WANDB_PROJECT:-dl_project_sst}"
export EVAL_STRIDE="${EVAL_STRIDE:-2}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export PROBE_LR="${PROBE_LR:-1e-3}"
export PROBE_EPOCHS="${PROBE_EPOCHS:-100}"
export PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-256}"
export KNN_K="${KNN_K:-20}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/scratch/$USER/my_env/conda_pkgs}"
export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-/scratch/$USER/my_env/conda_envs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/scratch/$USER/my_env/pip_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/$USER/my_env/.cache}"
export HOME="${HOME:-/scratch/$USER/my_env/home}"
export TMPDIR="${TMPDIR:-/scratch/$USER/my_env/tmp}"

mkdir -p "${CONDA_PKGS_DIRS}" "${CONDA_ENVS_PATH}" "${PIP_CACHE_DIR}" "${XDG_CACHE_HOME}" "${HOME}" "${TMPDIR}"

singularity exec --nv --overlay "${OVERLAY}" "${SIF}" /bin/bash -lc "
  set -euo pipefail
  source /ext3/env.sh
  export HOME='${HOME}'
  export TMPDIR='${TMPDIR}'
  export CONDA_PKGS_DIRS='${CONDA_PKGS_DIRS}'
  export CONDA_ENVS_PATH='${CONDA_ENVS_PATH}'
  export PIP_CACHE_DIR='${PIP_CACHE_DIR}'
  export XDG_CACHE_HOME='${XDG_CACHE_HOME}'
  export CONDA_NO_PLUGINS=true
  cd /scratch/$USER/active-matter-ssl
  export WANDB_API_KEY=\$(tr -d '\r\n' < /scratch/$USER/.cache/wandb/api_key)
  conda activate '${ENV_PREFIX}'

  ${PYTHON_BIN} -m src.evaluate_diffusion \
    --checkpoint ${CHECKPOINT} \
    --data_dir ${DATA_DIR} \
    --eval_stride ${EVAL_STRIDE} \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --probe_lr ${PROBE_LR} \
    --probe_epochs ${PROBE_EPOCHS} \
    --probe_batch_size ${PROBE_BATCH_SIZE} \
    --knn_k ${KNN_K} \
    --wandb_project ${WANDB_PROJECT} \
    --run_name ${RUN_NAME}
"
