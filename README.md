# active-matter-ssl (`optimize` branch)

Future-prediction extensions for JEPA-style representation learning on the `active_matter` dataset.

This branch is focused on the future-prediction variants built on top of the channel-factored JEPA baseline:

- `future_mode=latent`
- `future_mode=noisy`
- `future_mode=action`
- `future_mode=hybrid`
- `future_mode=hierarchical`

The core idea is to keep the ViT JEPA backbone fixed and vary how future latent targets are predicted:

- `latent`: direct future latent prediction
- `noisy`: future latent denoising
- `action`: future latent prediction conditioned on an inferred transition code
- `hybrid`: denoising plus action conditioning
- `hierarchical`: hybrid prediction plus coarse chunk-level transition prediction

## Reproducibility checklist

### Seeds and determinism

- Random seeds should be fixed for all reported runs.
- The hierarchical search script exposes `--seed` and records it in the search manifest.
- Supervised training already exposes `--seed`.
- Mask generation uses the PyTorch RNG, so runs are reproducible under fixed Torch seeds.
- Exact seeds used for final reported runs should be recorded in W\&B or Slurm logs.

### Data preprocessing

- Dataset: `active_matter` from The Well
- Input clip length: typically `16` frames for JEPA/future-prediction runs
- Spatial resolution: `256 x 256`
- Channels:
  - concentration: `1`
  - velocity: `2`
  - orientation tensor: `4`
  - strain-rate tensor: `4`
- Tubelet size: `2 x 16 x 16`
- Normalization:
  - per-channel z-score normalization using training-set statistics loaded from `stats.yaml`
- Augmentations:
  - none
- Masking:
  - base JEPA and future-prediction variants use block masking
  - most future-prediction runs in this branch use `mask_strategy=concentration`

### Separation of stages

The project is organized in two clearly separated stages:

1. Representation learning / pretraining
   - `src/train_ssl.py`
   - `src/train_diffusion.py`
2. Frozen-feature evaluation
   - `src/evaluate.py`
   - `src/evaluate_diffusion.py`
   - `src/evaluate_supervised.py`

Linear probing and k-NN are always run on frozen representations after pretraining.

## Model families in this branch

### JEPA backbone

Main JEPA model:

- `src/ssl_model/cfjepa.py`

Backbone details:

- Vision Transformer encoder with 3D RoPE
- channel-factored patch embedding by default
- EMA target encoder
- narrow transformer predictor
- SIGReg regularization

The default future mode in code is:

- `future_mode=direct`

Additional future-prediction modes implemented in this branch:

- `latent`
- `noisy`
- `action`
- `hybrid`
- `hierarchical`

### Diffusion baseline

- `src/diffusion_model.py`
- trained with `src/train_diffusion.py`
- evaluated with `src/evaluate_diffusion.py`

### Supervised baseline

- `src/models/supervised_baseline.py`
- trained with `src/train_supervised.py`
- uses `resnet18(weights=None)`, i.e. no external pretrained weights

## Parameter counts

The reportable JEPA backbones are under the 100M limit for trainable parameters:

- ViT-S JEPA: about `25.4M` trainable params
- ViT-B JEPA: about `91.5M` trainable params

The full online+EMA total parameter count is higher, but the trainable model remains below 100M.

You can print parameter counts from:

- `CFJEPA.param_count()`
- `FutureLatentDiffusion.param_count()`

## Main training scripts

### Core JEPA

- `scripts/train_ssl.slurm`

### Future-prediction variants

- `scripts/train_ssl_latent.slurm`
- `scripts/train_ssl_noisy_latent.slurm`
- `scripts/train_ssl_action.slurm`
- `scripts/train_ssl_hybrid.slurm`
- `scripts/train_ssl_hierarchical.slurm`

### Diffusion

- `scripts/train_diffusion.slurm`

### Supervised

- `scripts/train_supervised.slurm`

## Evaluation scripts

### JEPA / future-prediction checkpoints

- `scripts/evaluate.slurm`

### Diffusion checkpoints

- `scripts/evaluate_diffusion.slurm`

### Supervised checkpoints

- `scripts/evaluate_supervised.slurm`

## Hyperparameter search in this branch

The hierarchical search is implemented in:

- `scripts/submit_hierarchical_tuning.py`
- `scripts/hyperparam_search_hierarchical.slurm`

This search can vary:

- `lambda_within`
- `lambda_cross`
- `lambda_future`
- `lambda_sigreg`
- `lambda_action`
- `within_mask_ratio`
- `weight_decay`
- `predictor_dim`
- `predictor_depth`

Example small grid used in this branch:

- fixed:
  - `within_mask_ratio = 0.9`
  - `weight_decay = 0.1`
  - `predictor_dim = 128`
  - `predictor_depth = 2`
  - `lambda_future = 1.0`
  - `lambda_sigreg = 1.0`
- varied:
  - `lambda_within ∈ {0.5, 1.0}`
  - `lambda_cross ∈ {0.5, 1.0}`
  - `lambda_action ∈ {0.05, 0.1}`

## Example commands

### Train latent future JEPA

```bash
sbatch --account=torch_pr_494_general scripts/train_ssl_latent.slurm
```

### Train noisy future JEPA

```bash
sbatch --account=torch_pr_494_general scripts/train_ssl_noisy_latent.slurm
```

### Train action-conditioned future JEPA

```bash
sbatch --account=torch_pr_494_general scripts/train_ssl_action.slurm
```

### Train hybrid future JEPA

```bash
sbatch --account=torch_pr_494_general scripts/train_ssl_hybrid.slurm
```

### Train hierarchical future JEPA

```bash
sbatch --account=torch_pr_494_general scripts/train_ssl_hierarchical.slurm
```

### Evaluate JEPA / future-prediction checkpoint

```bash
sbatch --account=torch_pr_494_general \
  --export=ALL,CHECKPOINT=/scratch/$USER/active-matter-ssl/runs/ssl_hybrid/best.pt,RUN_NAME=eval_hybrid \
  scripts/evaluate.slurm
```

### Evaluate diffusion checkpoint

```bash
sbatch --account=torch_pr_494_general scripts/evaluate_diffusion.slurm
```

## Compute accounting

Document the following for reported experiments:

- GPU type
- number of GPUs per run
- wall-clock runtime
- total GPU-hours
- peak memory
- use of mixed precision

In this repo:

- JEPA and diffusion training use mixed precision
- common hardware target has been `A100 40GB`
- future-prediction full runs have typically taken multiple GPU hours each

Use `sacct` on the cluster login node to recover exact GPU-hours for the project.

## No external models or weights

This codebase does not rely on external pretrained model weights for the reported JEPA, diffusion, or supervised baselines.

- ViT backbones are trained from scratch
- the supervised ResNet baseline uses `weights=None`
- no external datasets beyond `active_matter` should be referenced in reported experiments

## Logging and configs

- Training and evaluation are logged to W\&B
- Slurm launchers encode the exact command-line configuration
- hierarchical search writes a manifest:
  - `runs/search_hierarchical/search_manifest.tsv`

For final reported runs, retain:

- Slurm job script
- W\&B run id
- checkpoint path
- evaluation output
- seed
