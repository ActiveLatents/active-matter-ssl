#!/usr/bin/env bash
set -euo pipefail

export SIF="${SIF:-/share/apps/images/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif}"
export OVERLAY="${OVERLAY:-/scratch/$USER/my_env/overlay-25GB-500K.ext3:rw}"
export CONDA_ENV="${CONDA_ENV:-amssl}"
export ENV_PREFIX="${ENV_PREFIX:-/scratch/$USER/my_env/conda_envs/${CONDA_ENV}}"
export PYTHON_BIN="${PYTHON_BIN:-${ENV_PREFIX}/bin/python}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/scratch/$USER/my_env/conda_pkgs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/scratch/$USER/my_env/pip_cache}"
export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-/scratch/$USER/my_env/conda_envs}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/$USER/my_env/.cache}"
export HOME="${HOME:-/scratch/$USER/my_env/home}"
export TMPDIR="${TMPDIR:-/scratch/$USER/my_env/tmp}"

mkdir -p /scratch/$USER/my_env
mkdir -p "${CONDA_PKGS_DIRS}"
mkdir -p "${PIP_CACHE_DIR}"
mkdir -p "${CONDA_ENVS_PATH}"
mkdir -p "${XDG_CACHE_HOME}"
mkdir -p "${HOME}"
mkdir -p "${TMPDIR}"

singularity exec --fakeroot --nv --overlay "${OVERLAY}" "${SIF}" /bin/bash -lc "
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

  if [[ ! -x '${PYTHON_BIN}' ]]; then
    conda create --solver classic -y -n '${CONDA_ENV}' python=3.10
  fi

  conda activate '${CONDA_ENV}'
  python -m pip install --upgrade pip
  python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
  python -m pip install -r requirements.txt

  python - <<'PY'
import torch, numpy, scipy, sklearn, yaml, h5py, wandb, einops, timm, tqdm
print('torch', torch.__version__)
print('cuda available', torch.cuda.is_available())
print('numpy', numpy.__version__)
print('scipy', scipy.__version__)
print('sklearn', sklearn.__version__)
print('pyyaml', yaml.__version__)
print('h5py', h5py.__version__)
print('wandb', wandb.__version__)
print('einops', einops.__version__)
print('timm', timm.__version__)
PY
"
