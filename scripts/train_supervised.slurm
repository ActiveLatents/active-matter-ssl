#!/usr/bin/env bash
#SBATCH --job-name=supervised_baseline
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
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
export OUTPUT_DIR="${OUTPUT_DIR:-/scratch/$USER/active-matter-ssl/runs/supervised}"
export RUN_NAME="${RUN_NAME:-supervised-cnn-baseline}"
export WANDB_PROJECT="${WANDB_PROJECT:-dl_project_sst}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/scratch/$USER/my_env/conda_pkgs}"
export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-/scratch/$USER/my_env/conda_envs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/scratch/$USER/my_env/pip_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/$USER/my_env/.cache}"
export HOME="${HOME:-/scratch/$USER/my_env/home}"
export TMPDIR="${TMPDIR:-/scratch/$USER/my_env/tmp}"

mkdir -p "${CONDA_PKGS_DIRS}" "${CONDA_ENVS_PATH}" "${PIP_CACHE_DIR}" "${XDG_CACHE_HOME}" "${HOME}" "${TMPDIR}" "${OUTPUT_DIR}"

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

  ${PYTHON_BIN} -m src.train_supervised \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --epochs 50 \
    --batch_size 16 \
    --n_frames 16 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --stride 1 \
    --num_workers 8 \
    --prefetch_factor 4 \
    --allow_tf32 \
    --wandb_project ${WANDB_PROJECT} \
    --wandb_run ${RUN_NAME} \
    --resume ${OUTPUT_DIR}/last.pt
"
