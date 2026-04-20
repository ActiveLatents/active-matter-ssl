# src/model/losses.py

"""
Loss functions for Channel-Factored JEPA.

Three components:
  1. Prediction loss (smooth L1): measures how well the predictor
     reconstructs the encoder's representations at masked positions.
  2. SIGReg (variance + covariance regularization): prevents representation
     collapse by ensuring each feature dimension has high variance and
     low correlation with other dimensions.
  3. Combined loss: weighted sum of prediction losses and SIGReg.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Prediction loss ─────────────────────────────────────────────────────────

def prediction_loss(predictions, targets):
    """
    Smooth L1 loss between predicted and target representations.

    Args:
        predictions: (B, N_mask, D) -- predictor output at masked positions
        targets:     (B, N_mask, D) -- encoder output at masked positions
                     (computed by running full input through encoder, detached)

    Returns:
        scalar loss
    """
    return F.smooth_l1_loss(predictions, targets)


# ── SIGReg ──────────────────────────────────────────────────────────────────

def sigreg_loss(embeddings, var_weight=1.0, cov_weight=1.0, var_target=1.0):
    """
    SIGReg: variance + covariance regularization on representations.

    Variance term: pushes the standard deviation of each feature dimension
    toward `var_target` using a hinge loss. Prevents dimension collapse.

    Covariance term: pushes off-diagonal entries of the feature covariance
    matrix toward zero. Prevents redundancy between dimensions.

    Args:
        embeddings: (B, D) -- pooled representations for a batch
                    or (B, N, D) which will be reshaped to (B*N, D)
        var_weight: weight for variance term
        cov_weight: weight for covariance term
        var_target: target standard deviation per dimension

    Returns:
        total_loss: scalar
        var_loss:   scalar (for logging)
        cov_loss:   scalar (for logging)
    """
    # Handle token-level representations by flattening
    if embeddings.dim() == 3:
        embeddings = embeddings.reshape(-1, embeddings.shape[-1])

    # Cast to float32: covariance matmul in bfloat16 loses too much precision
    embeddings = embeddings.float()

    B, D = embeddings.shape

    # Center the embeddings
    embeddings = embeddings - embeddings.mean(dim=0, keepdim=True)

    # Variance loss: hinge on per-dimension std
    std = embeddings.std(dim=0)
    var_loss = F.relu(var_target - std).mean()

    # Covariance loss: penalize off-diagonal correlations
    cov = (embeddings.T @ embeddings) / (B - 1)  # (D, D)

    # Zero out diagonal (we only penalize off-diagonal)
    off_diag = cov - torch.diag(cov.diag())
    cov_loss = (off_diag ** 2).sum() / D

    total = var_weight * var_loss + cov_weight * cov_loss

    return total, var_loss.detach(), cov_loss.detach()


def sigreg_per_group(encoder_output, field_indices, var_weight=1.0, cov_weight=1.0):
    """
    Apply SIGReg separately to each field group's tokens, plus globally.

    This ensures that each field group individually uses its representation
    space well, not just the overall concatenation.

    Args:
        encoder_output: (B, N_total, D) -- full encoder output (all groups)
        field_indices:  dict mapping group name to (start_idx, end_idx)
        var_weight:     weight for variance term
        cov_weight:     weight for covariance term

    Returns:
        total_loss: scalar
        loss_dict:  dict with per-group and global losses for logging
    """
    total = 0.0
    loss_dict = {}

    # Per-group SIGReg
    for name, (start, end) in field_indices.items():
        group_tokens = encoder_output[:, start:end, :]  # (B, N_group, D)
        group_loss, var_l, cov_l = sigreg_loss(
            group_tokens, var_weight=var_weight, cov_weight=cov_weight,
        )
        total = total + group_loss
        loss_dict[f"sigreg_{name}_var"] = var_l
        loss_dict[f"sigreg_{name}_cov"] = cov_l

    # Global SIGReg on all tokens
    global_loss, var_l, cov_l = sigreg_loss(
        encoder_output, var_weight=var_weight, cov_weight=cov_weight,
    )
    total = total + global_loss
    loss_dict["sigreg_global_var"] = var_l
    loss_dict["sigreg_global_cov"] = cov_l

    loss_dict["sigreg_total"] = total.detach()

    return total, loss_dict


# ── Combined loss ───────────────────────────────────────────────────────────

class CFJEPALoss(nn.Module):
    """
    Combined loss for Channel-Factored JEPA training.

    L_total = lambda_within * L_within
            + lambda_cross  * L_cross
            + lambda_sigreg * L_sigreg
    """

    def __init__(
        self,
        lambda_within=1.0,
        lambda_cross=1.0,
        lambda_sigreg=0.1,
        sigreg_var_weight=1.0,
        sigreg_cov_weight=1.0,
    ):
        super().__init__()
        self.lambda_within = lambda_within
        self.lambda_cross = lambda_cross
        self.lambda_sigreg = lambda_sigreg
        self.sigreg_var_weight = sigreg_var_weight
        self.sigreg_cov_weight = sigreg_cov_weight

    def forward(
        self,
        within_preds,
        within_targets,
        cross_preds,
        cross_targets,
        encoder_output,
        field_indices,
    ):
        """
        Args:
            within_preds:    (B, N_within_mask, D) -- predictor output for within-field
            within_targets:  (B, N_within_mask, D) -- encoder targets for within-field
            cross_preds:     (B, N_cross_mask, D) -- predictor output for cross-field
            cross_targets:   (B, N_cross_mask, D) -- encoder targets for cross-field
            encoder_output:  (B, N_total, D) -- full encoder output for SIGReg
            field_indices:   dict mapping group name to (start, end)

        Returns:
            total_loss: scalar
            loss_dict:  dict with all loss components for logging
        """
        # Prediction losses
        l_within = prediction_loss(within_preds, within_targets)
        l_cross = prediction_loss(cross_preds, cross_targets)

        # SIGReg (per-group + global)
        l_sigreg, sigreg_dict = sigreg_per_group(
            encoder_output, field_indices,
            var_weight=self.sigreg_var_weight,
            cov_weight=self.sigreg_cov_weight,
        )

        # Weighted sum
        total = (
            self.lambda_within * l_within
            + self.lambda_cross * l_cross
            + self.lambda_sigreg * l_sigreg
        )

        # Logging dict
        loss_dict = {
            "loss_total": total.detach(),
            "loss_within": l_within.detach(),
            "loss_cross": l_cross.detach(),
            **sigreg_dict,
        }

        return total, loss_dict


# ── Quick test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, D = 4, 384

    # Test SIGReg
    emb = torch.randn(B, 100, D)
    loss, var_l, cov_l = sigreg_loss(emb)
    print(f"SIGReg: total={loss:.4f}, var={var_l:.4f}, cov={cov_l:.4f}")

    # Test prediction loss
    pred = torch.randn(B, 50, D)
    tgt = torch.randn(B, 50, D)
    ploss = prediction_loss(pred, tgt)
    print(f"Prediction loss: {ploss:.4f}")

    # Test combined loss
    criterion = CFJEPALoss()
    field_indices = {
        "concentration": (0, 25),
        "velocity": (25, 50),
        "orientation": (50, 75),
        "strain_rate": (75, 100),
    }
    total, loss_dict = criterion(
        within_preds=torch.randn(B, 75, D),
        within_targets=torch.randn(B, 75, D),
        cross_preds=torch.randn(B, 25, D),
        cross_targets=torch.randn(B, 25, D),
        encoder_output=torch.randn(B, 100, D),
        field_indices=field_indices,
    )
    print(f"\nCombined loss: {total:.4f}")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")