# src/ssl_model/losses.py

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


# ── Prediction loss ─────────────────────────────────────────────────────────

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
    return F.mse_loss(predictions, targets)


# ── SIGReg ───────────────────────────────────────────────────────────────────

def sigreg_loss(x, sketch_dim=64, n_integration=17):
    """
    Sketched Isotropic Gaussian Regularization (SIGReg).

    Forces the empirical distribution of x to match a standard Gaussian by
    comparing their Empirical Characteristic Functions (ECF) via random
    projection. Based on LeJEPA Algorithm 1 (https://github.com/kreasof-ai/sigreg).

    Algorithm:
      1. Project x → sketch_dim via a random unit-column matrix A
      2. Compute ECF(t) = E[exp(it * x@A)] for t in [-5, 5]
      3. Compare to Gaussian CF: G(t) = exp(-t²/2)
      4. Loss = integral |ECF(t) - G(t)|² * G(t) dt  (Gaussian-weighted L2)

    The Gaussian weighting focuses the comparison on the region where the
    Gaussian CF has support (|t| < 3), avoiding noise at the tails.

    Why ECF > VICReg:
      - VICReg enforces variance ≈ 1 and low covariance (2nd moments only).
        A mixture of B point masses (B=6 per-sample collapse) can satisfy this
        if the B directions are diverse.
      - ECF matching enforces ALL moments. A mixture of 6 Dirac deltas has
        ECF = (1/6) Σ exp(it·x_i), which is never Gaussian-shaped.

    Args:
        x:              (N, D) or (B, N, D) — flattened to (B*N, D) if 3D
        sketch_dim:     random projection dimension (trade-off: coverage vs cost)
        n_integration:  number of quadrature points for trapezoid integration

    Returns:
        scalar loss (≥ 0; equals 0 when x is exactly standard Gaussian)
    """
    if x.dim() == 3:
        x = x.reshape(-1, x.shape[-1])

    # Cast to float32: complex exponential loses precision in bfloat16
    x = x.float()
    N, C = x.shape

    # 1. Random unit-column projection: C → sketch_dim
    A = torch.randn(C, sketch_dim, device=x.device)
    A = A / (A.norm(p=2, dim=0, keepdim=True) + 1e-6)

    # 2. Integration points and theoretical Gaussian CF
    t = torch.linspace(-5, 5, n_integration, device=x.device)
    gauss_cf = torch.exp(-0.5 * t ** 2)  # (T,)

    # 3. Empirical CF: E_N[exp(it * x@A)]
    proj = x @ A                                            # (N, sketch_dim)
    args = proj.unsqueeze(2) * t.view(1, 1, -1)            # (N, sketch_dim, T)
    ecf = torch.exp(1j * args).mean(dim=0)                 # (sketch_dim, T), complex

    # 4. Gaussian-weighted L2 distance, integrated via trapezoid rule
    diff_sq = (ecf - gauss_cf.unsqueeze(0)).abs().square() # (sketch_dim, T)
    err = diff_sq * gauss_cf.unsqueeze(0)                  # downweight tails
    loss = torch.trapezoid(err, t, dim=1)                  # (sketch_dim,)

    return loss.mean()


# ── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch

    B, D = 4, 384

    # Gaussian input → loss should be near 0
    x_gauss = torch.randn(B, 100, D)
    loss_gauss = sigreg_loss(x_gauss)
    print(f"SIGReg (Gaussian input):  {loss_gauss:.6f}  (expect ~0)")

    # Collapsed input (all tokens same per sample) → loss should be high
    x_collapsed = torch.zeros(B, 100, D)
    for b in range(B):
        x_collapsed[b] = torch.randn(1, D).expand(100, D)
    loss_collapsed = sigreg_loss(x_collapsed)
    print(f"SIGReg (collapsed input): {loss_collapsed:.6f}  (expect >> 0)")

    # Prediction loss
    pred = torch.randn(B, 50, D)
    tgt  = F.normalize(torch.randn(B, 50, D), dim=-1)
    ploss = prediction_loss(pred, tgt)
    print(f"Prediction loss (random): {ploss:.4f}  (expect ~2.0)")

    pred_perfect = tgt.clone()
    print(f"Prediction loss (perfect): {prediction_loss(pred_perfect, tgt):.6f}  (expect 0)")
