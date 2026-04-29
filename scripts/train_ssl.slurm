#!/bin/bash
#SBATCH --job-name=cfjepa-single-embed
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ssl_%j.out
#SBATCH --error=logs/ssl_%j.err
#SBATCH --requeue
#SBATCH --signal=USR1@120
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sk12590@nyu.edu

# Activate environment
singularity exec --nv --overlay /scratch/sk12590/cross-modal-brain-retrieval/overlay-25GB-500K.ext3:ro \
    /share/apps/images/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif \
    /bin/bash -c "
        source /ext3/env.sh

        cd /scratch/sk12590/active-matter-ssl

        python -m src.train_ssl \
            --data_dir /scratch/sk12590/dl_project_data/data \
            --n_frames 16 \
            --train_stride 1 \
            --val_stride 1 \
            --batch_size 6 \
            --num_workers 4 \
            --embed_dim 384 \
            --encoder_depth 12 \
            --encoder_heads 6 \
            --predictor_dim 192 \
            --predictor_depth 4 \
            --predictor_heads 6 \
            --within_mask_ratio 0.90 \
            --lambda_within 1.0 \
            --lambda_cross 0.0 \
            --lambda_sigreg 1.0 \
            --epochs 30 \
            --base_lr 1.5e-4 \
            --weight_decay 0.05 \
            --warmup_epochs 10 \
            --grad_clip 1.0 \
            --log_every 50 \
            --val_every 5 \
            --wandb_project active-matter-ssl \
            --run_name cfjepa-ablation-single-patch-embed \
            --single_patch_embed \
            --checkpoint_dir /scratch/sk12590/active-matter-ssl/runs/ssl_single_embed_mask90
        "
