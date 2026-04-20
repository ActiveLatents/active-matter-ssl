# src/model/cfjepa.py

"""
Channel-Factored JEPA (CF-JEPA).

Top-level model that orchestrates:
  - Channel-factored patch embedding
  - Shared ViT encoder
  - Lightweight predictor
  - Within-field and cross-field masking
  - Combined loss (prediction + SIGReg)

Training flow:
  1. Embed each field group into tokens (separate projections)
  2. Add positional + field-type encodings
  3. Generate within-field and cross-field masks
  4. Get prediction targets: run ALL tokens through encoder (detached)
  5. Get encoder output: run only VISIBLE tokens through encoder
  6. Predictor takes visible representations + mask tokens, predicts masked
  7. Compute prediction loss + SIGReg

Evaluation flow (linear probe / kNN):
  1. Embed + encode ALL tokens (no masking)
  2. Pool to a single representation vector
  3. Use pooled representation for downstream regression
"""

import torch
import torch.nn as nn

from .patch_embed import ChannelFactoredPatchEmbed
from .encoder import ViTEncoder
from .predictor import Predictor
from .masking import generate_masks, batch_mask_indices
from .losses import CFJEPALoss


class CFJEPA(nn.Module):

    def __init__(
        self,
        # Patch embedding
        embed_dim=384,
        tube_t=2,
        patch_h=16,
        patch_w=16,
        n_frames=16,
        spatial_size=256,
        # Encoder
        encoder_depth=12,
        encoder_heads=6,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        # Predictor
        predictor_dim=192,
        predictor_depth=4,
        predictor_heads=6,
        # Masking
        within_mask_ratio=0.75,
        # Loss weights
        lambda_within=1.0,
        lambda_cross=1.0,
        lambda_sigreg=0.1,
        sigreg_var_weight=1.0,
        sigreg_cov_weight=1.0,
    ):
        super().__init__()

        self.within_mask_ratio = within_mask_ratio

        # Modules
        self.patch_embed = ChannelFactoredPatchEmbed(
            embed_dim=embed_dim,
            tube_t=tube_t,
            patch_h=patch_h,
            patch_w=patch_w,
            n_frames=n_frames,
            spatial_size=spatial_size,
        )

        self.encoder = ViTEncoder(
            embed_dim=embed_dim,
            depth=encoder_depth,
            n_heads=encoder_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
        )

        self.predictor = Predictor(
            encoder_dim=embed_dim,
            predictor_dim=predictor_dim,
            depth=predictor_depth,
            n_heads=predictor_heads,
            mlp_ratio=mlp_ratio,
        )

        self.criterion = CFJEPALoss(
            lambda_within=lambda_within,
            lambda_cross=lambda_cross,
            lambda_sigreg=lambda_sigreg,
            sigreg_var_weight=sigreg_var_weight,
            sigreg_cov_weight=sigreg_cov_weight,
        )

    def _gather_tokens(self, tokens, indices):
        """
        Gather tokens at given indices from the sequence dimension.

        Args:
            tokens:  (B, N_total, D)
            indices: (B, N_select)
        Returns:
            (B, N_select, D)
        """
        D = tokens.shape[-1]
        idx = indices.unsqueeze(-1).expand(-1, -1, D)
        return torch.gather(tokens, 1, idx)

    def forward_ssl(self, field_dict):
        """
        Full SSL training forward pass.

        Three encoder passes:
          A. All tokens (with gradients) -- used for SIGReg and as
             prediction targets (detached when used as targets)
          B. Within-field visible tokens -- fed to predictor for
             within-field prediction
          C. Cross-field visible tokens -- fed to predictor for
             cross-field prediction

        Args:
            field_dict: dict with keys (concentration, velocity, orientation,
                        strain_rate), each (B, T, C_field, H, W)

        Returns:
            total_loss: scalar
            loss_dict:  dict of all loss components
        """
        device = next(self.parameters()).device

        # 1. Embed all field groups into tokens
        all_tokens, field_indices, grid_shape = self.patch_embed(field_dict)
        B, N_total, D = all_tokens.shape

        # 2. Generate masks
        within_masks, cross_masks = generate_masks(
            field_indices,
            within_mask_ratio=self.within_mask_ratio,
            device=device,
        )

        # 3. Pass A: encode ALL tokens (with gradients)
        #    Serves dual purpose: SIGReg regularization + prediction targets
        full_encoder_out = self.encoder(all_tokens)

        # ── Within-field path ───────────────────────────────────────────
        w_vis_ids = batch_mask_indices(within_masks["visible_ids"], B)
        w_mask_ids = batch_mask_indices(within_masks["masked_ids"], B)

        # Pass B: encode only within-field visible tokens
        w_visible_tokens = self._gather_tokens(all_tokens, w_vis_ids)
        w_encoder_out = self.encoder(w_visible_tokens)

        # Predict masked tokens
        w_preds = self.predictor(
            visible_tokens=w_encoder_out,
            visible_indices=w_vis_ids,
            masked_indices=w_mask_ids,
            total_tokens=N_total,
        )

        # Targets: detach from pass A
        w_targets = self._gather_tokens(full_encoder_out, w_mask_ids).detach()

        # ── Cross-field path ────────────────────────────────────────────
        c_vis_ids = batch_mask_indices(cross_masks["visible_ids"], B)
        c_mask_ids = batch_mask_indices(cross_masks["masked_ids"], B)

        # Pass C: encode only cross-field visible tokens
        c_visible_tokens = self._gather_tokens(all_tokens, c_vis_ids)
        c_encoder_out = self.encoder(c_visible_tokens)

        # Predict the masked group
        c_preds = self.predictor(
            visible_tokens=c_encoder_out,
            visible_indices=c_vis_ids,
            masked_indices=c_mask_ids,
            total_tokens=N_total,
        )

        # Targets: detach from pass A
        c_targets = self._gather_tokens(full_encoder_out, c_mask_ids).detach()

        # ── Compute combined loss ───────────────────────────────────────
        total_loss, loss_dict = self.criterion(
            within_preds=w_preds,
            within_targets=w_targets,
            cross_preds=c_preds,
            cross_targets=c_targets,
            encoder_output=full_encoder_out,  # SIGReg uses this (has gradients)
            field_indices=field_indices,
        )

        return total_loss, loss_dict

    @torch.no_grad()
    def encode(self, field_dict, pool="mean"):
        """
        Extract frozen representations for evaluation (linear probe / kNN).

        Args:
            field_dict: dict with field group tensors
            pool:       "mean" for mean pooling over all tokens

        Returns:
            features: (B, embed_dim) -- pooled representation
        """
        all_tokens, field_indices, _ = self.patch_embed(field_dict)
        encoder_out = self.encoder(all_tokens)

        if pool == "mean":
            features = encoder_out.mean(dim=1)  # (B, D)
        else:
            raise ValueError(f"Unknown pooling: {pool}")

        return features

    def param_count(self):
        """Print parameter counts per module."""
        counts = {}
        counts["patch_embed"] = sum(p.numel() for p in self.patch_embed.parameters())
        counts["encoder"] = sum(p.numel() for p in self.encoder.parameters())
        counts["predictor"] = sum(p.numel() for p in self.predictor.parameters())
        counts["total"] = sum(p.numel() for p in self.parameters())

        for name, count in counts.items():
            print(f"  {name:15s}: {count / 1e6:.2f}M")

        return counts


# ── Quick test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = CFJEPA(
        embed_dim=384,
        tube_t=2,
        patch_h=16,
        patch_w=16,
        n_frames=16,
        spatial_size=256,
        encoder_depth=12,
        encoder_heads=6,
        predictor_dim=192,
        predictor_depth=4,
        predictor_heads=6,
        within_mask_ratio=0.75,
    )

    print("Parameter counts:")
    model.param_count()

    # Fake SSL batch
    B = 2
    field_dict = {
        "concentration": torch.randn(B, 16, 1, 256, 256),
        "velocity":      torch.randn(B, 16, 2, 256, 256),
        "orientation":   torch.randn(B, 16, 4, 256, 256),
        "strain_rate":   torch.randn(B, 16, 4, 256, 256),
    }

    print("\nSSL forward pass...")
    loss, loss_dict = model.forward_ssl(field_dict)
    print(f"Total loss: {loss:.4f}")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")

    print("\nEvaluation encode...")
    features = model.encode(field_dict)
    print(f"Features: {features.shape}")  # (2, 384)