#!/bin/bash
#SBATCH --job-name=cfjepa-ssl
#SBATCH --account=torch_pr_494_general
#SBATCH --partition=torch_pr_494_general
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ssl_%j.out
#SBATCH --error=logs/ssl_%j.err
#SBATCH --requeue
#SBATCH --signal=USR1@120
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sk12590@nyu.edu

# Activate environment
singularity exec --nv --overlay /scratch/$USER/my_env/overlay-25GB-500K.ext3:ro \
    /share/apps/images/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif \
    /bin/bash -c "
        source /ext3/env.sh
        conda activate active-matter-ssl

        cd /scratch/$USER/active-matter-ssl
        source ./wandb_env.sh

        python -m src.train_ssl \
            --data_dir /scratch/zl6113/active-matter/data \
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
            --wandb_project active-matter-ssl \
            --run_name cfjepa-future-concentrationmask-h100 \
            --checkpoint_dir /scratch/$USER/active-matter-ssl/runs/ssl
        "
