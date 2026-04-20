#!/bin/bash
#SBATCH --job-name=cfjepa-ssl
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --output=logs/ssl_%j.out
#SBATCH --error=logs/ssl_%j.err
#SBATCH --requeue
#SBATCH --signal=USR1@120
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sk12590@nyu.edu

# Activate environment
singularity exec --nv --overlay /scratch/sk12590/active-matter-ssl/overlay-25GB-500K.ext3:rw \
    /share/apps/images/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif \
    /bin/bash -c "
        source /ext3/env.sh

        cd /scratch/sk12590/active-matter-ssl

        python -m src.train_ssl \
            --data_dir /scratch/sk12590/dl_project_data/data \
            --n_frames 16 \
            --train_stride 5 \
            --val_stride 1 \
            --batch_size 6 \
            --num_workers 4 \
            --embed_dim 384 \
            --encoder_depth 12 \
            --encoder_heads 6 \
            --predictor_dim 192 \
            --predictor_depth 4 \
            --predictor_heads 6 \
            --within_mask_ratio 0.75 \
            --lambda_within 1.0 \
            --lambda_cross 1.0 \
            --lambda_sigreg 0.01 \
            --epochs 100 \
            --base_lr 1.5e-4 \
            --weight_decay 0.05 \
            --warmup_epochs 10 \
            --grad_clip 1.0 \
            --log_every 50 \
            --val_every 5 \
            --wandb_project active-matter-ssl \
            --run_name cfjepa-v1 \
            --checkpoint_dir /scratch/sk12590/active-matter-ssl/runs/ssl
        "
