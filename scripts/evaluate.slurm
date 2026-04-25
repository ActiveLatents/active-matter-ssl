#!/bin/bash
#SBATCH --job-name=cfjepa-eval
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

singularity exec --nv --overlay /scratch/sk12590/active-matter-ssl/overlay-25GB-500K.ext3:rw \
    /share/apps/images/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif \
    /bin/bash -c "
        source /ext3/env.sh

        cd /scratch/sk12590/active-matter-ssl

        python -m src.evaluate \
            --checkpoint /scratch/sk12590/active-matter-ssl/runs/ssl/best.pt \
            --data_dir /scratch/sk12590/dl_project_data/data \
            --eval_stride 2 \
            --batch_size 8 \
            --num_workers 4 \
            --probe_lr 1e-3 \
            --probe_epochs 100 \
            --probe_batch_size 256 \
            --knn_k 20 \
            --wandb_org dl_project_sst \
            --wandb_project team_active_matter \
            --run_name linear_probe_and_knn_eval_no_target_enc_${SLURM_JOB_ID} \
        "
