# src/ssl_model/cfjepa.py

"""
Channel-Factored JEPA (CF-JEPA) with EMA target encoder.

Training flow (I-JEPA style):
  1. Online patch embedding → tokens for visible context
  2. Target patch embedding (EMA) → full tokens → target encoder (EMA) → targets
  3. Generate within-field and cross-field masks
  4. Online encoder processes VISIBLE tokens only (2 passes, with gradients)
  5. Predictor reconstructs masked target representations
  6. Loss = prediction loss + SIGReg on online encoder outputs

The EMA target encoder provides stable, non-collapsing targets.
update_ema() must be called after every optimizer step.

Evaluation flow (linear probe / kNN):
  1. Target encoder encodes ALL tokens (most stable representations)
  2. Pool to a single representation vector
"""

import copy

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
        sigreg_var_weight=1.0,
        sigreg_cov_weight=1.0,
    ):
        super().__init__()

        self.within_mask_ratio = within_mask_ratio
        self.lambda_within = lambda_within
        self.lambda_cross = lambda_cross
        self.lambda_sigreg = lambda_sigreg
        self.sigreg_var_weight = sigreg_var_weight
        self.sigreg_cov_weight = sigreg_cov_weight

        # ── Online modules (receive gradients) ──────────────────────────
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

        # ── Target modules (EMA copy, no gradients) ─────────────────────
        self.target_patch_embed = copy.deepcopy(self.patch_embed)
        self.target_encoder = copy.deepcopy(self.encoder)

        for p in self.target_patch_embed.parameters():
            p.requires_grad_(False)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update_ema(self, decay: float):
        """
        Update target encoder via exponential moving average of online encoder.
        Call after every optimizer.step().

        target = decay * target + (1 - decay) * online
        """
        for online, target in zip(
            self.patch_embed.parameters(), self.target_patch_embed.parameters()
        ):
            target.data.mul_(decay).add_(online.data, alpha=1.0 - decay)

        for online, target in zip(
            self.encoder.parameters(), self.target_encoder.parameters()
        ):
            target.data.mul_(decay).add_(online.data, alpha=1.0 - decay)

    def _gather_tokens(self, tokens, indices):
        """
        Gather tokens at given indices.
        tokens:  (B, N, D)
        indices: (B, K)
        returns: (B, K, D)
        """
        D = tokens.shape[-1]
        idx = indices.unsqueeze(-1).expand(-1, -1, D)
        return torch.gather(tokens, 1, idx)

    def forward_ssl(self, field_dict):
        """
        SSL training forward pass with EMA target encoder.

        Two online encoder passes (with gradients):
          B. Within-field visible tokens
          C. Cross-field visible tokens

        One target encoder pass (no gradient):
          T. All tokens → stable prediction targets

        SIGReg is applied to combined online encoder outputs to prevent
        online encoder collapse.

        Args:
            field_dict: dict with keys (concentration, velocity, orientation,
                        strain_rate), each (B, T, C_field, H, W)

        Returns:
            total_loss: scalar
            loss_dict:  dict of all loss components
        """
        device = next(self.parameters()).device

        # 1. Online patch embedding
        all_tokens, field_indices, grid_shape = self.patch_embed(field_dict)
        B, N_total, D = all_tokens.shape

        # 2. Generate spatiotemporal block masks (Fix 1)
        within_masks, cross_masks = generate_masks(
            field_indices,
            grid_shape=grid_shape,
            within_mask_ratio=self.within_mask_ratio,
            device=device,
        )

        # 3. Target pass: EMA encoder on ALL tokens (no gradient)
        with torch.no_grad():
            target_tokens, _, _ = self.target_patch_embed(field_dict)
            full_target_out = self.target_encoder(target_tokens)

        # ── Within-field online pass ────────────────────────────────────
        w_vis_ids  = batch_mask_indices(within_masks["visible_ids"], B)
        w_mask_ids = batch_mask_indices(within_masks["masked_ids"],  B)

        w_visible_tokens = self._gather_tokens(all_tokens, w_vis_ids)
        w_encoder_out    = self.encoder(w_visible_tokens)

        w_preds   = self.predictor(
            visible_tokens=w_encoder_out,
            visible_indices=w_vis_ids,
            masked_indices=w_mask_ids,
            total_tokens=N_total,
        )
        # Fix 4: normalize targets to unit sphere before loss.
        # Prevents encoder from collapsing to small-magnitude outputs that
        # trivially minimize MSE; predictor must match directions, not scales.
        w_targets = F.normalize(
            self._gather_tokens(full_target_out, w_mask_ids).float(), dim=-1
        )

        # ── Cross-field online pass ─────────────────────────────────────
        c_vis_ids  = batch_mask_indices(cross_masks["visible_ids"], B)
        c_mask_ids = batch_mask_indices(cross_masks["masked_ids"],  B)

        c_visible_tokens = self._gather_tokens(all_tokens, c_vis_ids)
        c_encoder_out    = self.encoder(c_visible_tokens)

        c_preds   = self.predictor(
            visible_tokens=c_encoder_out,
            visible_indices=c_vis_ids,
            masked_indices=c_mask_ids,
            total_tokens=N_total,
        )
        c_targets = F.normalize(
            self._gather_tokens(full_target_out, c_mask_ids).float(), dim=-1
        )

        # ── Losses ─────────────────────────────────────────────────────
        l_within = prediction_loss(w_preds, w_targets)
        l_cross  = prediction_loss(c_preds, c_targets)

        # Fix 3: SIGReg on mean-pooled per-sample representations (B, D).
        # Pooling first forces diverse per-sample summaries rather than
        # trivially satisfying token-level variance within each sample.
        online_pooled = torch.cat([w_encoder_out, c_encoder_out], dim=1).mean(dim=1)
        l_sigreg, var_l, cov_l = sigreg_loss(
            online_pooled,
            var_weight=self.sigreg_var_weight,
            cov_weight=self.sigreg_cov_weight,
        )

        total_loss = (
            self.lambda_within  * l_within
            + self.lambda_cross * l_cross
            + self.lambda_sigreg * l_sigreg
        )

        loss_dict = {
            "loss_total":        total_loss.detach(),
            "loss_within":       l_within.detach(),
            "loss_cross":        l_cross.detach(),
            "sigreg_total":      l_sigreg.detach(),
            "sigreg_global_var": var_l,
            "sigreg_global_cov": cov_l,
        }

        return total_loss, loss_dict

    @torch.no_grad()
    def encode(self, field_dict, pool="mean"):
        """
        Extract representations for evaluation (linear probe / kNN).
        Uses the target encoder — its representations are more stable
        than the online encoder since it is updated via EMA.

        Args:
            field_dict: dict with field group tensors
            pool:       "mean" for mean pooling over all tokens

        Returns:
            features: (B, embed_dim)
        """
        target_tokens, _, _ = self.target_patch_embed(field_dict)
        encoder_out = self.target_encoder(target_tokens)

        if pool == "mean":
            return encoder_out.mean(dim=1)
        raise ValueError(f"Unknown pooling: {pool}")

    def param_count(self):
        """Print trainable and total parameter counts."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())

        components = {
            "patch_embed (online)":  self.patch_embed,
            "encoder (online)":      self.encoder,
            "predictor":             self.predictor,
            "patch_embed (target)":  self.target_patch_embed,
            "encoder (target)":      self.target_encoder,
        }
        counts = {}
        for name, module in components.items():
            n = sum(p.numel() for p in module.parameters())
            print(f"  {name:25s}: {n / 1e6:.2f}M")
            counts[name] = n

        print(f"  {'trainable':25s}: {trainable / 1e6:.2f}M")
        print(f"  {'total':25s}: {total / 1e6:.2f}M")
        counts["trainable"] = trainable
        counts["total"] = total
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

    # EMA update
    model.update_ema(decay=0.996)
    print("\nEMA update: OK")

    print("\nEvaluation encode (target encoder)...")
    features = model.encode(field_dict)
    print(f"Features: {features.shape}")  # (2, 384)
