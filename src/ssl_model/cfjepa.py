# src/ssl_model/cfjepa.py

"""
Channel-Factored JEPA (CF-JEPA).

Training flow:
  1. Patch embedding → all tokens
  2. Encoder(all tokens) → prediction targets  (no stop-gradient, no EMA)
  3. Generate within-field and cross-field spatiotemporal block masks
  4. Two encoder passes on VISIBLE tokens
  5. Predictor reconstructs masked target representations
  6. Loss = prediction loss (normalized MSE) + per-sample SIGReg

SIGReg is the sole anti-collapse mechanism. It forces each sample's token
representations to be Gaussian-distributed, making collapse impossible
regardless of how gradients flow — no stop-gradient or EMA heuristics needed.

Evaluation flow (linear probe / kNN):
  Encoder encodes ALL tokens, mean-pool to a single representation vector.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .patch_embed import ChannelFactoredPatchEmbed
from .encoder import ViTEncoder
from .predictor import Predictor
from .masking import generate_masks, batch_mask_indices
from .losses import prediction_loss, sigreg_loss


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
    ):
        super().__init__()

        self.within_mask_ratio = within_mask_ratio
        self.lambda_within = lambda_within
        self.lambda_cross = lambda_cross
        self.lambda_sigreg = lambda_sigreg

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

    def _gather_tokens(self, tokens, indices):
        """
        tokens:  (B, N, D)
        indices: (B, K)
        returns: (B, K, D)
        """
        D = tokens.shape[-1]
        idx = indices.unsqueeze(-1).expand(-1, -1, D)
        return torch.gather(tokens, 1, idx)

    def forward_ssl(self, field_dict):
        """
        SSL training forward pass.

        The encoder runs on all tokens to produce prediction targets, and also
        on visible-only subsets for the online passes. Gradients flow freely
        through all paths. SIGReg on the online encoder outputs is the only
        anti-collapse mechanism.

        Args:
            field_dict: dict with keys (concentration, velocity, orientation,
                        strain_rate), each (B, T, C_field, H, W)

        Returns:
            total_loss: scalar
            loss_dict:  dict of all loss components
        """
        device = next(self.parameters()).device

        # 1. Patch embedding
        all_tokens, field_indices, grid_shape, pos_ids = self.patch_embed(field_dict)
        B, N_total, D = all_tokens.shape

        # 2. Generate spatiotemporal block masks
        within_masks, cross_masks = generate_masks(
            field_indices,
            grid_shape=grid_shape,
            within_mask_ratio=self.within_mask_ratio,
            device=device,
        )

        # 3. Target pass: full encoder on all tokens, no stop-gradient.
        # SIGReg is the sole anti-collapse mechanism — no heuristics needed.
        # pos_ids: (N_total, 3) → expand to (B, N_total, 3) for encoder
        full_pos_ids = pos_ids.unsqueeze(0).expand(B, -1, -1)
        full_target_out = self.encoder(all_tokens, pos_ids=full_pos_ids)

        # ── Within-field online pass ────────────────────────────────────
        w_vis_ids  = batch_mask_indices(within_masks["visible_ids"], B)
        w_mask_ids = batch_mask_indices(within_masks["masked_ids"],  B)

        w_visible_tokens = self._gather_tokens(all_tokens, w_vis_ids)
        # Gather pos_ids for visible token subset: (B, N_vis, 3)
        w_pos_ids = pos_ids[w_vis_ids]
        w_encoder_out    = self.encoder(w_visible_tokens, pos_ids=w_pos_ids)

        w_preds   = self.predictor(
            visible_tokens=w_encoder_out,
            visible_indices=w_vis_ids,
            masked_indices=w_mask_ids,
            total_tokens=N_total,
            pos_ids=pos_ids,
        )
        w_targets = F.normalize(
            self._gather_tokens(full_target_out, w_mask_ids).float(), dim=-1
        )

        # ── Cross-field online pass ─────────────────────────────────────
        c_vis_ids  = batch_mask_indices(cross_masks["visible_ids"], B)
        c_mask_ids = batch_mask_indices(cross_masks["masked_ids"],  B)

        c_visible_tokens = self._gather_tokens(all_tokens, c_vis_ids)
        c_pos_ids = pos_ids[c_vis_ids]
        c_encoder_out    = self.encoder(c_visible_tokens, pos_ids=c_pos_ids)

        c_preds   = self.predictor(
            visible_tokens=c_encoder_out,
            visible_indices=c_vis_ids,
            masked_indices=c_mask_ids,
            total_tokens=N_total,
            pos_ids=pos_ids,
        )
        c_targets = F.normalize(
            self._gather_tokens(full_target_out, c_mask_ids).float(), dim=-1
        )

        # ── Losses ─────────────────────────────────────────────────────
        l_within = prediction_loss(w_preds, w_targets)
        l_cross  = prediction_loss(c_preds, c_targets)

        # SIGReg on the full target pass: full_target_out sees all tokens in a
        # single coherent attention context and is the learning target, making
        # it the right place to enforce Gaussian-distributed representations.
        # Subsample tokens to keep the O(N^2) pairwise term in sigreg_loss
        # within memory bounds (1024 tokens → ~1.6 GB for B=6, sketch_dim=64).
        n_sigreg = min(1024, N_total)
        idx = torch.randperm(N_total, device=device)[:n_sigreg]
        l_sigreg = sigreg_loss(full_target_out[:, idx, :])

        total_loss = (
            self.lambda_within  * l_within
            + self.lambda_cross * l_cross
            + self.lambda_sigreg * l_sigreg
        )

        loss_dict = {
            "loss_total":   total_loss.detach(),
            "loss_within":  l_within.detach(),
            "loss_cross":   l_cross.detach(),
            "sigreg_total": l_sigreg.detach(),
        }

        return total_loss, loss_dict

    @torch.no_grad()
    def encode(self, field_dict, pool="mean"):
        """
        Extract representations for evaluation (linear probe / kNN).

        Args:
            field_dict: dict with field group tensors
            pool:       "mean" for mean pooling over all tokens

        Returns:
            features: (B, embed_dim)
        """
        all_tokens, _, _, pos_ids = self.patch_embed(field_dict)
        B = all_tokens.shape[0]
        full_pos_ids = pos_ids.unsqueeze(0).expand(B, -1, -1)
        encoder_out = self.encoder(all_tokens, pos_ids=full_pos_ids)

        if pool == "mean":
            return encoder_out.mean(dim=1)
        raise ValueError(f"Unknown pooling: {pool}")

    def param_count(self):
        """Print trainable and total parameter counts."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())

        components = {
            "patch_embed": self.patch_embed,
            "encoder":     self.encoder,
            "predictor":   self.predictor,
        }
        for name, module in components.items():
            n = sum(p.numel() for p in module.parameters())
            print(f"  {name:12s}: {n / 1e6:.2f}M")

        print(f"  {'trainable':12s}: {trainable / 1e6:.2f}M")
        print(f"  {'total':12s}: {total / 1e6:.2f}M")
        return {"trainable": trainable, "total": total}


# ── Quick test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = CFJEPA(
        embed_dim=384, tube_t=2, patch_h=16, patch_w=16,
        n_frames=16, spatial_size=256,
        encoder_depth=12, encoder_heads=6,
        predictor_dim=192, predictor_depth=4, predictor_heads=6,
        within_mask_ratio=0.75,
    )

    print("Parameter counts:")
    model.param_count()

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
