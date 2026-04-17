#!/bin/bash
#SBATCH --requeue
#SBATCH --job-name=supervised_baseline
#SBATCH --partition=c12m85-a100-1
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --signal=USR1@120
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sk12590@nyu.edu

# ── Environment ───────────────────────────────────────────────────────────────
module purge
singularity exec --nv \
    --overlay /scratch/sk12590/active-matter-ssl/overlay-25GB-500K.ext3:rw \
    /share/apps/images/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif \
    /bin/bash -c "
        source /ext3/env.sh

        cd /scratch/sk12590/active-matter-ssl

        python -m src.train_supervised \
            --data_dir    /scratch/sk12590/dl_project_data/data \
            --output_dir  runs/supervised \
            --epochs      50 \
            --batch_size  4 \
            --lr          1e-3 \
            --stride      1 \
            --num_workers 8 \
            --wandb_project active-matter-ssl \
            --wandb_run   supervised-baseline \
            --resume      runs/supervised/last.pt
    "
