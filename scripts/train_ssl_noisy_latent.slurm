#!/usr/bin/env bash
#SBATCH --job-name=cfjepa_noisy
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
export PYTHON_BIN="${PYTHON_BIN:-/scratch/$USER/my_env/conda_envs/${CONDA_ENV}/bin/python}"
export DATA_DIR="${DATA_DIR:-/scratch/zl6113/active-matter/data}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-/scratch/$USER/active-matter-ssl/runs/ssl_noisy}"
export RUN_NAME="${RUN_NAME:-cfjepa-noisyfuture-concentrationmask}"
export WANDB_PROJECT="${WANDB_PROJECT:-dl_project_sst}"
export WANDB_ENTITY="${WANDB_ENTITY:-team_active_matter}"
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
  export WANDB_API_KEY=\$(tr -d '\r\n' < ./cache/wandb/api_key)
  export WANDB_ENTITY='${WANDB_ENTITY}'
  conda activate '${CONDA_ENV}'

  ${PYTHON_BIN} -m src.train_ssl \
    --data_dir ${DATA_DIR} \
    --n_frames 16 \
    --train_stride 2 \
    --val_stride 1 \
    --batch_size 6 \
    --num_workers 8 \
    --prefetch_factor 4 \
    --embed_dim 384 \
    --encoder_depth 12 \
    --encoder_heads 6 \
    --predictor_dim 192 \
    --predictor_depth 4 \
    --predictor_heads 6 \
    --within_mask_ratio 0.75 \
    --mask_strategy concentration \
    --future_mode noisy \
    --lambda_within 1.0 \
    --lambda_cross 1.0 \
    --lambda_future 1.0 \
    --lambda_sigreg 1.0 \
    --lambda_vorticity 0.0 \
    --lambda_spectral 0.0 \
    --epochs 50 \
    --base_lr 1.5e-4 \
    --weight_decay 0.05 \
    --warmup_epochs 10 \
    --grad_clip 1.0 \
    --allow_tf32 \
    --compile_model \
    --log_every 50 \
    --val_every 5 \
    --wandb_project ${WANDB_PROJECT} \
    --run_name ${RUN_NAME} \
    --checkpoint_dir ${CHECKPOINT_DIR}
"
