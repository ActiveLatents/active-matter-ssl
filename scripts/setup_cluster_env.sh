#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH_ROOT="/scratch/${USER}/active-matter-ssl"
OVERLAY_PATH="${SCRATCH_ROOT}/overlay-25GB-500K.ext3"
IMAGE_PATH="/share/apps/images/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif"
ENV_NAME="active-matter-ssl"
PYTHON_VERSION="3.10"

mkdir -p "${SCRATCH_ROOT}"

if [[ ! -f "${OVERLAY_PATH}" ]]; then
  echo "Missing overlay: ${OVERLAY_PATH}" >&2
  echo "Create or copy it first under /scratch/\$USER/active-matter-ssl" >&2
  exit 1
fi

rsync -av --delete "${REPO_ROOT}/" "${SCRATCH_ROOT}/"

singularity exec --nv --overlay "${OVERLAY_PATH}:rw" "${IMAGE_PATH}" /bin/bash -lc "
  set -euo pipefail
  source /ext3/env.sh
  cd /scratch/${USER}/active-matter-ssl

  if ! conda env list | awk '{print \$1}' | grep -qx '${ENV_NAME}'; then
    conda create -y -n '${ENV_NAME}' python=${PYTHON_VERSION}
  fi

  conda activate '${ENV_NAME}'
  python -m pip install --upgrade pip
  python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
  python -m pip install -r requirements.txt
  python -m pip install jupyterlab

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
