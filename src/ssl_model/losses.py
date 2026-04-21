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
    Per-sample Sketched Isotropic Gaussian Regularization (SIGReg).

    Applies ECF matching INDEPENDENTLY for each sample's N token representations,
    then averages over the batch. This is the critical distinction from global
    (batch-level) ECF matching.

    The collapse mode we are fighting: at initialization, all N tokens within
    sample i collapse to the same representation direction c_i. The global
    distribution of (B*N) tokens can look Gaussian (if c_i ~ N(0,I) across
    samples), satisfying batch-level SIGReg. But per-sample, each sample is a
    Dirac delta with ECF = exp(it·proj(c_i)), which looks nothing like a
    Gaussian CF = exp(-t²/2). Per-sample SIGReg directly detects and penalizes
    this.

    Algorithm (per sample b):
      1. Project sample b's N tokens: proj_b = x_b @ A  (N, sketch_dim)
      2. Empirical CF per sample: ECF_b(t) = mean_n[exp(it * proj_b)]  (sketch_dim, T)
      3. Compare to Gaussian CF: G(t) = exp(-t²/2)
      4. Loss_b = integral |ECF_b(t) - G(t)|² * G(t) dt
      5. Return mean over B samples

    Args:
        x:              (B, N, D) — one row of N token representations per sample.
                        If (N, D), treated as a single sample (B=1).
        sketch_dim:     random projection dimension
        n_integration:  number of quadrature points for trapezoid integration

    Returns:
        scalar loss (≥ 0; equals 0 when each sample's tokens are Gaussian)
    """
    if x.dim() == 2:
        x = x.unsqueeze(0)  # treat as B=1

    B, N, D = x.shape

    # Cast to float32: complex exponential loses precision in bfloat16
    x = x.float()

    # 1. Shared random unit-column projection: D → sketch_dim
    A = torch.randn(D, sketch_dim, device=x.device)
    A = A / (A.norm(p=2, dim=0, keepdim=True) + 1e-6)

    # 2. Integration points and theoretical Gaussian CF
    t = torch.linspace(-5, 5, n_integration, device=x.device)       # (T,)
    gauss_cf = torch.exp(-0.5 * t ** 2)                              # (T,)

    # 3. Per-sample empirical CF
    #    proj:  (B, N, sketch_dim)
    #    args:  (B, N, sketch_dim, T)
    #    ecf:   (B, sketch_dim, T)  — mean over N tokens within each sample
    proj = x @ A
    args = proj.unsqueeze(-1) * t.view(1, 1, 1, -1)
    ecf  = torch.exp(1j * args).mean(dim=1)

    # 4. Gaussian-weighted L2, integrated via trapezoid rule
    diff_sq = (ecf - gauss_cf.view(1, 1, -1)).abs().square()         # (B, sketch_dim, T)
    err     = diff_sq * gauss_cf.view(1, 1, -1)
    loss    = torch.trapezoid(err, t, dim=-1)                         # (B, sketch_dim)

    return loss.mean()  # average over batch and sketch dimensions


# ── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch

    B, N, D = 4, 512, 384

    # Per-sample Gaussian tokens → each sample's ECF matches Gaussian → loss ≈ 0
    x_gauss = torch.randn(B, N, D)
    loss_gauss = sigreg_loss(x_gauss)
    print(f"SIGReg (per-sample Gaussian):  {loss_gauss:.6f}  (expect ~0)")

    # Per-sample collapse: all N tokens in sample b = same direction c_b
    # Global distribution of B*N tokens looks Gaussian (c_b ~ N(0,I))
    # but per-sample ECF is a Dirac delta → loss should be large
    x_collapsed = torch.stack([
        torch.randn(1, D).expand(N, D) for _ in range(B)
    ])
    loss_global_ok = sigreg_loss(x_collapsed.reshape(-1, D).unsqueeze(0).squeeze(0)
                                  .reshape(1, B * N, D))
    loss_persample = sigreg_loss(x_collapsed)
    print(f"SIGReg global (collapsed):     {loss_global_ok:.6f}  (may look ~0 — misses collapse)")
    print(f"SIGReg per-sample (collapsed): {loss_persample:.6f}  (expect >> 0)")

    # Prediction loss
    pred = torch.randn(B, 50, D)
    tgt  = F.normalize(torch.randn(B, 50, D), dim=-1)
    print(f"Prediction loss (random):  {prediction_loss(pred, tgt):.4f}  (expect ~2.0)")
    print(f"Prediction loss (perfect): {prediction_loss(tgt, tgt):.6f}  (expect 0)")
