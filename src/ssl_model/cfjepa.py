# src/ssl_model/cfjepa.py

"""
Channel-Factored JEPA — Temporal Frame Prediction.

Training flow:
  1. Split the T-frame input into context (first T/2 frames) and
     target (last T/2 frames).
  2. Embed each half independently with the shared patch embedding.
     Target tokens get their temporal pos_ids shifted by n_t_half so
     that 3D RoPE encodes the correct absolute time position.
  3. Online encoder processes context tokens only.
  4. Target encoder (EMA) processes target tokens under no_grad.
  5. Predictor attends over context encoder output + learnable query
     tokens at target positions, and predicts the target representations.
  6. Loss = prediction loss (normalized MSE) + SIGReg on predictor output.

Evaluation flow (linear probe / kNN):
  Encoder encodes ALL tokens from the full sequence, mean-pooled to a
  single representation vector.
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .patch_embed import ChannelFactoredPatchEmbed
from .encoder import ViTEncoder
from .predictor import Predictor
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
        # Loss weights
        lambda_sigreg=0.1,
    ):
        super().__init__()

        self.lambda_sigreg = lambda_sigreg

        self.patch_embed = ChannelFactoredPatchEmbed(
            embed_dim=embed_dim,
            tube_t=tube_t,
            patch_h=patch_h,
            patch_w=patch_w,
            n_frames=n_frames,
            spatial_size=spatial_size,
        )

        # RoPE must cover the full temporal range including the shifted target
        # pos_ids (0..n_t_half-1 for context, n_t_half..n_t_full-1 for target).
        max_t = n_frames // tube_t
        max_h = spatial_size // patch_h
        max_w = spatial_size // patch_w

        self.encoder = ViTEncoder(
            embed_dim=embed_dim,
            depth=encoder_depth,
            n_heads=encoder_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            max_t=max_t,
            max_h=max_h,
            max_w=max_w,
        )

        self.target_encoder = copy.deepcopy(self.encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.predictor = Predictor(
            encoder_dim=embed_dim,
            predictor_dim=predictor_dim,
            depth=predictor_depth,
            n_heads=predictor_heads,
            mlp_ratio=mlp_ratio,
            max_t=max_t,
            max_h=max_h,
            max_w=max_w,
        )

    def _split_field_dict(self, field_dict):
        """Split each field tensor at T//2 into context and target halves."""
        ctx, tgt = {}, {}
        for k, v in field_dict.items():
            half = v.shape[1] // 2
            ctx[k] = v[:, :half]
            tgt[k] = v[:, half:]
        return ctx, tgt

    def forward_ssl(self, field_dict):
        """
        SSL training forward pass.

        Args:
            field_dict: dict with keys (concentration, velocity, orientation,
                        strain_rate), each (B, T, C_field, H, W).
                        T must be even.

        Returns:
            total_loss: scalar
            loss_dict:  dict of loss components
        """
        device = next(self.parameters()).device

        # 1. Split into context (first half) and target (second half)
        ctx_dict, tgt_dict = self._split_field_dict(field_dict)

        # 2. Embed each half. Both halves have the same spatial grid shape.
        #    pos_ids from patch_embed always start at t=0, so shift the
        #    target pos_ids by n_t_half to encode absolute temporal position.
        ctx_tokens, _, grid_shape, pos_ids_ctx = self.patch_embed(ctx_dict)
        tgt_tokens, _, _,          pos_ids_tgt = self.patch_embed(tgt_dict)

        n_t_half = grid_shape[0]  # number of temporal tube steps per half
        pos_ids_tgt = pos_ids_tgt.clone()
        pos_ids_tgt[:, 0] += n_t_half  # shift t coords into the second half

        B = ctx_tokens.shape[0]
        N_ctx = ctx_tokens.shape[1]
        N_tgt = tgt_tokens.shape[1]
        N_total = N_ctx + N_tgt

        # Full pos_ids covering both halves (used by predictor for RoPE)
        pos_ids_full = torch.cat([pos_ids_ctx, pos_ids_tgt], dim=0)  # (N_total, 3)

        # 3. Online encoder: context frames only
        ctx_pos = pos_ids_ctx.unsqueeze(0).expand(B, -1, -1)
        ctx_enc = self.encoder(ctx_tokens, pos_ids=ctx_pos)  # (B, N_ctx, D)

        # 4. Target encoder: second-half frames (no gradient)
        tgt_pos = pos_ids_tgt.unsqueeze(0).expand(B, -1, -1)
        with torch.no_grad():
            tgt_enc = self.target_encoder(tgt_tokens, pos_ids=tgt_pos)  # (B, N_tgt, D)
        tgt_targets = F.normalize(tgt_enc.float(), dim=-1)

        # 5. Predictor: context tokens are "visible", target positions are queries
        vis_ids = torch.arange(N_ctx, device=device).unsqueeze(0).expand(B, -1)
        tgt_ids = torch.arange(N_ctx, N_total, device=device).unsqueeze(0).expand(B, -1)

        preds = self.predictor(
            visible_tokens=ctx_enc,
            visible_indices=vis_ids,
            masked_indices=tgt_ids,
            total_tokens=N_total,
            pos_ids=pos_ids_full,
        )  # (B, N_tgt, D)

        # 6. Losses
        l_pred = prediction_loss(preds, tgt_targets)

        n_sub = min(1024, preds.shape[1])
        idx = torch.randperm(preds.shape[1], device=device)[:n_sub]
        l_sigreg = sigreg_loss(preds[:, idx, :])

        total_loss = l_pred + self.lambda_sigreg * l_sigreg

        loss_dict = {
            "loss_total":  total_loss.detach(),
            "loss_pred":   l_pred.detach(),
            "sigreg":      l_sigreg.detach(),
        }

        return total_loss, loss_dict

    @torch.no_grad()
    def update_target_encoder(self, momentum):
        """EMA update of target encoder."""
        for param, target_param in zip(self.encoder.parameters(),
                                       self.target_encoder.parameters()):
            target_param.data.mul_(momentum).add_(param.data, alpha=1.0 - momentum)

    @torch.no_grad()
    def encode(self, field_dict, pool="mean"):
        """
        Extract representations for evaluation (linear probe / kNN).

        Only the context half (first T//2 frames) is encoded, matching the
        distribution the online encoder saw during training. Using all T frames
        would expose the encoder to temporal positions it never processed during
        training (the target half was only ever seen by the target encoder).

        Returns:
            features: (B, embed_dim)
        """
        ctx_dict, _ = self._split_field_dict(field_dict)
        ctx_tokens, _, _, pos_ids_ctx = self.patch_embed(ctx_dict)
        B = ctx_tokens.shape[0]
        ctx_pos = pos_ids_ctx.unsqueeze(0).expand(B, -1, -1)
        encoder_out = self.encoder(ctx_tokens, pos_ids=ctx_pos)

        if pool == "mean":
            return encoder_out.mean(dim=1)
        raise ValueError(f"Unknown pooling: {pool}")

    def param_count(self):
        """Print trainable and total parameter counts."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())

        components = {
            "patch_embed":    self.patch_embed,
            "encoder":        self.encoder,
            "predictor":      self.predictor,
            "target_encoder": self.target_encoder,
        }
        for name, module in components.items():
            n = sum(p.numel() for p in module.parameters())
            frozen = " (frozen)" if not any(p.requires_grad for p in module.parameters()) else ""
            print(f"  {name:16s}: {n / 1e6:.2f}M{frozen}")

        print(f"  {'trainable':16s}: {trainable / 1e6:.2f}M")
        print(f"  {'total':16s}: {total / 1e6:.2f}M")
        return {"trainable": trainable, "total": total}


# ── Quick test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = CFJEPA(
        embed_dim=384, tube_t=2, patch_h=16, patch_w=16,
        n_frames=16, spatial_size=256,
        encoder_depth=12, encoder_heads=6,
        predictor_dim=192, predictor_depth=4, predictor_heads=6,
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

    print("\nEvaluation encode (context half only)...")
    features = model.encode(field_dict)
    print(f"Features: {features.shape}")  # (2, 384)
