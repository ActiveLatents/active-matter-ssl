#!/usr/bin/env bash
#SBATCH --job-name=cfjepa_hier
#SBATCH --partition=h200_public
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
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
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-/scratch/$USER/active-matter-ssl/runs/ssl_hierarchical}"
export RUN_NAME="${RUN_NAME:-cfjepa-hierarchicalfuture-concentrationmask}"
export WANDB_PROJECT="${WANDB_PROJECT:-dl_project_sst}"
export N_FRAMES="${N_FRAMES:-16}"
export TRAIN_STRIDE="${TRAIN_STRIDE:-2}"
export VAL_STRIDE="${VAL_STRIDE:-1}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
export EMBED_DIM="${EMBED_DIM:-384}"
export ENCODER_DEPTH="${ENCODER_DEPTH:-12}"
export ENCODER_HEADS="${ENCODER_HEADS:-6}"
export PREDICTOR_DIM="${PREDICTOR_DIM:-128}"
export PREDICTOR_DEPTH="${PREDICTOR_DEPTH:-2}"
export PREDICTOR_HEADS="${PREDICTOR_HEADS:-4}"
export DROP_RATE="${DROP_RATE:-0.10}"
export ATTN_DROP_RATE="${ATTN_DROP_RATE:-0.10}"
export DROP_PATH_RATE="${DROP_PATH_RATE:-0.10}"
export WITHIN_MASK_RATIO="${WITHIN_MASK_RATIO:-0.85}"
export MASK_STRATEGY="${MASK_STRATEGY:-concentration}"
export FUTURE_MODE="${FUTURE_MODE:-hierarchical}"
export LAMBDA_WITHIN="${LAMBDA_WITHIN:-1.0}"
export LAMBDA_CROSS="${LAMBDA_CROSS:-1.0}"
export LAMBDA_FUTURE="${LAMBDA_FUTURE:-1.0}"
export LAMBDA_SIGREG="${LAMBDA_SIGREG:-1.0}"
export LAMBDA_LATENT_KL="${LAMBDA_LATENT_KL:-0.0}"
export LAMBDA_VORTICITY="${LAMBDA_VORTICITY:-0.0}"
export LAMBDA_SPECTRAL="${LAMBDA_SPECTRAL:-0.0}"
export LAMBDA_ACTION="${LAMBDA_ACTION:-0.1}"
export EPOCHS="${EPOCHS:-50}"
export BASE_LR="${BASE_LR:-1.5e-4}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.10}"
export WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
export GRAD_CLIP="${GRAD_CLIP:-1.0}"
export LOG_EVERY="${LOG_EVERY:-50}"
export VAL_EVERY="${VAL_EVERY:-5}"
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

  ${PYTHON_BIN} -m src.train_ssl \
    --data_dir ${DATA_DIR} \
    --n_frames ${N_FRAMES} \
    --train_stride ${TRAIN_STRIDE} \
    --val_stride ${VAL_STRIDE} \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --prefetch_factor ${PREFETCH_FACTOR} \
    --embed_dim ${EMBED_DIM} \
    --encoder_depth ${ENCODER_DEPTH} \
    --encoder_heads ${ENCODER_HEADS} \
    --predictor_dim ${PREDICTOR_DIM} \
    --predictor_depth ${PREDICTOR_DEPTH} \
    --predictor_heads ${PREDICTOR_HEADS} \
    --drop_rate ${DROP_RATE} \
    --attn_drop_rate ${ATTN_DROP_RATE} \
    --drop_path_rate ${DROP_PATH_RATE} \
    --within_mask_ratio ${WITHIN_MASK_RATIO} \
    --mask_strategy ${MASK_STRATEGY} \
    --future_mode ${FUTURE_MODE} \
    --lambda_within ${LAMBDA_WITHIN} \
    --lambda_cross ${LAMBDA_CROSS} \
    --lambda_future ${LAMBDA_FUTURE} \
    --lambda_sigreg ${LAMBDA_SIGREG} \
    --lambda_latent_kl ${LAMBDA_LATENT_KL} \
    --lambda_vorticity ${LAMBDA_VORTICITY} \
    --lambda_spectral ${LAMBDA_SPECTRAL} \
    --lambda_action ${LAMBDA_ACTION} \
    --epochs ${EPOCHS} \
    --base_lr ${BASE_LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --warmup_epochs ${WARMUP_EPOCHS} \
    --grad_clip ${GRAD_CLIP} \
    --allow_tf32 \
    --compile_model \
    --log_every ${LOG_EVERY} \
    --val_every ${VAL_EVERY} \
    --wandb_project ${WANDB_PROJECT} \
    --run_name ${RUN_NAME} \
    --checkpoint_dir ${CHECKPOINT_DIR}
"
