# active-matter-ssl

Self-supervised representation learning for active matter physical simulations. We train a Channel-Factored JEPA (CF-JEPA) on spatiotemporal field data and evaluate how well the learned representations capture two physical parameters: activity (zeta) and noise (alpha).

## Task

Given video-like simulations of active matter (concentration, velocity, orientation tensor, strain-rate tensor — 11 channels total, 256x256 spatial, up to 32 frames), learn representations that predict the physical parameters driving each simulation.

## Method

CF-JEPA is a masked prediction SSL framework with two design choices under study:

**Channel-Factored (CF) embedding** — four separate Conv3D projections (one per field group) producing 8,192 tokens per clip, with field-identity embeddings added to distinguish groups.

**Single embedding** — one Conv3D(11→D) over all 11 channels concatenated, producing 2,048 tokens per clip.

Both variants share the same ViT encoder (ViT-S or ViT-B), EMA target encoder, and predictor. The online encoder sees only visible tokens (within-field block masking at 75% or 90%); the predictor reconstructs masked tokens in the target encoder's embedding space. SIGReg (Signature Regularization) prevents representation collapse.

## Repository structure

```
src/
  dataset.py              ActiveMatter dataset and dataloader
  train_ssl.py            CF-JEPA pretraining loop
  train_supervised.py     Supervised baseline training
  evaluate.py             Linear probe + kNN evaluation on frozen encoder
  ssl_model/
    cfjepa.py             Main model (CFJEPA class)
    encoder.py            ViT encoder with 3D RoPE
    patch_embed.py        Tubelet patch embedding (CF and single)
    predictor.py          Narrow ViT predictor
    masking.py            Spatiotemporal block masking
    losses.py             Normalized MSE + SIGReg
  models/
    supervised_baseline.py  Per-frame CNN + temporal mean pool

scripts/
  train_ssl.slurm         Slurm job for SSL pretraining
  train_supervised.slurm  Slurm job for supervised baseline
  evaluate.slurm          Slurm job for LP/kNN evaluation
```

## Setup

```bash
pip install -r requirements.txt
```

Data should be placed at (or symlinked from) `/scratch/$NETID/dl_project_data/data` with the standard train/valid/test split as HDF5 files.

## Training

SSL pretraining:

```bash
python -m src.train_ssl \
    --data_dir /scratch/$NETID/dl_project_data/data \
    --checkpoint_dir runs/ssl \
    --epochs 30 \
    --embed_dim 384 \
    --encoder_depth 12 \
    --encoder_heads 6
```

Supervised baseline:

```bash
python -m src.train_supervised \
    --data_dir /scratch/$NETID/dl_project_data/data \
    --output_dir runs/supervised \
    --epochs 50
```

## Evaluation

Runs linear probe (100 epochs) and kNN regression on frozen encoder features:

```bash
python -m src.evaluate \
    --checkpoint runs/ssl/best.pt \
    --data_dir /scratch/$NETID/dl_project_data/data \
    --output_dir runs/eval
```

## Key results

Best run overall: Run 10 (Single embed, ViT-B, W+S loss, 75% masking) — LP valid MSE 0.0518, LP test MSE 0.0450, kNN valid MSE 0.0815.

Single embedding consistently outperforms channel-factored at both ViT-S and ViT-B scale. Within-field masked prediction (W+S) is more useful than cross-field masking.
