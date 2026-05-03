"""
Loss functions for Channel-Factored JEPA.

Two components:
  1. Prediction loss (normalized MSE): cosine alignment between L2-normalized
     predictions and targets. Prevents mean-predictor collapse.
  2. SIGReg (Sketched Isotropic Gaussian Regularization): forces the empirical
     distribution of token representations to match a standard Gaussian by
     comparing Empirical Characteristic Functions (ECF). Unlike VICReg/variance
     regularization (which only constrains 2nd moments), ECF matching constrains
     ALL moments — a mixture of B point masses (per-sample collapse mode) looks
     nothing like a Gaussian and is strongly penalized.
"""

import torch
import torch.nn.functional as F
import math


def prediction_loss(predictions, targets):
    """
    Normalized MSE (alignment loss) between predicted and target representations.

    Both are L2-normalized before MSE, equivalent to 2 - 2*cosine_similarity.
    Targets should already be L2-normalized (pre-normalized in cfjepa.py);
    the normalization here is applied to predictions only (idempotent on targets).

    Args:
        predictions: (B, N_mask, D) -- predictor output at masked positions
        targets:     (B, N_mask, D) -- unit-normalized encoder targets

    Returns:
        scalar loss in [0, 4]
    """
    predictions = F.normalize(predictions, dim=-1)
    targets = F.normalize(targets, dim=-1)
    return F.mse_loss(predictions, targets, reduction='none').sum(dim=-1).mean()


def sigreg_loss(x, sketch_dim=64):
    """
    Per-sample Sketched Isotropic Gaussian Regularization (SIGReg).
    
    Computes the exact closed-form analytical solution of the Epps-Pulley 
    statistic (which is mathematically equivalent to Gaussian RBF MMD) 
    along 1D random projections.
    """
    if x.dim() == 2:
        x = x.unsqueeze(0) 

    B, N, D = x.shape
    x = x.float()

    A = torch.randn(D, sketch_dim, device=x.device)
    A = A / (A.norm(p=2, dim=0, keepdim=True) + 1e-6)

    proj = x @ A

    
    # Term 1: 1/N^2 * sum(exp(-0.5 * (z_j - z_k)^2))
    z1 = proj.unsqueeze(2)  # (B, N, 1, sketch_dim)
    z2 = proj.unsqueeze(1)  # (B, 1, N, sketch_dim)
    diff_sq = (z1 - z2).square()  # (B, N, N, sketch_dim)
    term1 = torch.exp(-0.5 * diff_sq).mean(dim=(1, 2))  # (B, sketch_dim)

    # Term 2: sqrt(2)/N * sum(exp(-0.25 * z_j^2))
    term2 = math.sqrt(2.0) * torch.exp(-0.25 * proj.square()).mean(dim=1)  # (B, sketch_dim)

    # Term 3: Constant 1/sqrt(3)
    term3 = 1.0 / math.sqrt(3.0)

    loss = (term1 - term2 + term3) * math.sqrt(2 * math.pi)

    return loss.mean()  # average over batch and sketch dimensions


if __name__ == "__main__":

    B, N, D = 4, 512, 384

    x_gauss = torch.randn(B, N, D)
    loss_gauss = sigreg_loss(x_gauss)
    print(f"SIGReg (per-sample Gaussian):  {loss_gauss:.6f}  (expect ~0)")

    x_collapsed = torch.stack([
        torch.randn(1, D).expand(N, D) for _ in range(B)
    ])
    loss_global_ok = sigreg_loss(x_collapsed.reshape(-1, D).unsqueeze(0).squeeze(0)
                                  .reshape(1, B * N, D))
    loss_persample = sigreg_loss(x_collapsed)
    print(f"SIGReg global (collapsed):     {loss_global_ok:.6f}  (may look ~0 — misses collapse)")
    print(f"SIGReg per-sample (collapsed): {loss_persample:.6f}  (expect >> 0)")

    pred = torch.randn(B, 50, D)
    tgt  = F.normalize(torch.randn(B, 50, D), dim=-1)
    print(f"Prediction loss (random):  {prediction_loss(pred, tgt):.4f}  (expect ~2.0)")
    print(f"Prediction loss (perfect): {prediction_loss(tgt, tgt):.6f}  (expect 0)")
