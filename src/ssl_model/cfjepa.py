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
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .patch_embed import ChannelFactoredPatchEmbed
from .encoder import ViTEncoder
from .predictor import Predictor
from .masking import generate_masks, batch_mask_indices
from .losses import (
    prediction_loss,
    scalar_aux_loss,
    sigreg_loss,
    spectral_consistency_loss,
)


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
        mask_strategy="random",
        use_checkpointing=False,
        future_mode="direct",
        num_diffusion_steps=1000,
        # Loss weights
        lambda_within=1.0,
        lambda_cross=1.0,
        lambda_future=0.0,
        lambda_sigreg=0.1,
        lambda_vorticity=0.0,
        lambda_spectral=0.0,
        lambda_latent_kl=0.0,
        lambda_action=0.0,
    ):
        super().__init__()

        self.within_mask_ratio = within_mask_ratio
        self.mask_strategy = mask_strategy
        self.future_mode = future_mode
        self.lambda_within = lambda_within
        self.lambda_cross = lambda_cross
        self.lambda_future = lambda_future
        self.lambda_sigreg = lambda_sigreg
        self.lambda_vorticity = lambda_vorticity
        self.lambda_spectral = lambda_spectral
        self.lambda_latent_kl = lambda_latent_kl
        self.lambda_action = lambda_action
        self.tube_t = tube_t
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.num_diffusion_steps = num_diffusion_steps

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
            use_checkpointing=use_checkpointing,
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
            use_checkpointing=use_checkpointing,
        )
        self.vorticity_head = nn.Linear(embed_dim, 1)
        self.future_time_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.future_prior = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim * 2),
        )
        self.future_posterior = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim * 2),
        )
        self.action_infer = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.action_transition = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        betas = torch.linspace(1e-4, 0.02, num_diffusion_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("future_alpha_bars", alpha_bars)

    def _timestep_embedding(self, timesteps, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, device=timesteps.device) / max(half, 1)
        )
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def _q_sample_future(self, x0, timesteps, noise):
        alpha_bar = self.future_alpha_bars[timesteps].view(-1, 1, 1)
        return alpha_bar.sqrt() * x0 + (1.0 - alpha_bar).sqrt() * noise

    def _reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def _temporal_chunk_summary(self, tokens, token_indices, pos_ids, split_t, is_future, n_chunks=4):
        """
        Build chunkwise temporal summaries for selected tokens.

        Args:
            tokens:         (B, N_sel, D)
            token_indices:  (B, N_sel) global token indices into pos_ids
            pos_ids:        (N_total, 3)
            split_t:        temporal split between context and future
            is_future:      whether these tokens belong to the future half
            n_chunks:       number of chunks within the half-window

        Returns:
            summaries:      (B, n_chunks, D)
        """
        B, _, D = tokens.shape
        token_t = pos_ids[token_indices][..., 0]
        if is_future:
            local_t = token_t - split_t
            max_t = max(int(pos_ids[:, 0].max().item() + 1 - split_t), 1)
        else:
            local_t = token_t
            max_t = max(int(split_t), 1)

        chunk_ids = (local_t * n_chunks) // max(max_t, 1)
        chunk_ids = chunk_ids.clamp(min=0, max=n_chunks - 1)

        summaries = []
        for chunk in range(n_chunks):
            mask = chunk_ids == chunk
            mask_f = mask.unsqueeze(-1).to(tokens.dtype)
            denom = mask_f.sum(dim=1).clamp_min(1.0)
            chunk_sum = (tokens * mask_f).sum(dim=1) / denom
            summaries.append(chunk_sum)
        return torch.stack(summaries, dim=1)

    def _integrated_action_code(self, context_tokens, context_ids, future_tokens, future_ids, pos_ids, split_t):
        """
        Infer a transition code from chunkwise temporal summaries rather than
        only pooled endpoints, making the action pathway more trajectory-aware.
        """
        context_chunks = self._temporal_chunk_summary(
            context_tokens, context_ids, pos_ids, split_t, is_future=False
        )
        future_chunks = self._temporal_chunk_summary(
            future_tokens, future_ids, pos_ids, split_t, is_future=True
        )
        pairwise_inputs = torch.cat([context_chunks, future_chunks], dim=-1)
        pairwise_actions = self.action_infer(pairwise_inputs)
        return pairwise_actions.mean(dim=1)

    def _kl_normal(self, mu_q, logvar_q, mu_p, logvar_p):
        return 0.5 * (
            logvar_p - logvar_q
            + (logvar_q.exp() + (mu_q - mu_p).square()) / logvar_p.exp().clamp_min(1e-6)
            - 1.0
        ).sum(dim=-1).mean()

    def _pool_scalar_field_to_tokens(self, field):
        """
        field: (B, T, H, W)
        returns: (B, tokens_per_group)
        """
        pooled = F.avg_pool3d(
            field.unsqueeze(1),
            kernel_size=(self.tube_t, self.patch_h, self.patch_w),
            stride=(self.tube_t, self.patch_h, self.patch_w),
        )
        return pooled.flatten(2).squeeze(1)

    def _compute_vorticity_targets(self, velocity):
        """
        velocity: (B, T, 2, H, W)
        returns pooled vorticity target: (B, tokens_per_group)
        """
        u = velocity[:, :, 0]
        v = velocity[:, :, 1]
        dv_dx = 0.5 * (torch.roll(v, shifts=-1, dims=-1) - torch.roll(v, shifts=1, dims=-1))
        du_dy = 0.5 * (torch.roll(u, shifts=-1, dims=-2) - torch.roll(u, shifts=1, dims=-2))
        vorticity = dv_dx - du_dy
        return self._pool_scalar_field_to_tokens(vorticity)

    def _select_group_predictions(self, preds, mask_ids, group_start, group_end):
        """
        preds:    (B, K, D_or_1)
        mask_ids: (B, K)
        """
        group_mask = (mask_ids[0] >= group_start) & (mask_ids[0] < group_end)
        if not torch.any(group_mask):
            return None, None
        return preds[:, group_mask], mask_ids[:, group_mask] - group_start

    def _vorticity_loss_for_predictions(self, preds, mask_ids, field_indices, vorticity_targets):
        vel_start, vel_end = field_indices["velocity"]
        vel_preds, vel_local_ids = self._select_group_predictions(
            preds, mask_ids, vel_start, vel_end
        )
        if vel_preds is None:
            return preds.new_zeros(())
        pred_vorticity = self.vorticity_head(vel_preds).squeeze(-1)
        target_vorticity = self._gather_tokens(
            vorticity_targets.unsqueeze(-1), vel_local_ids
        ).squeeze(-1)
        return scalar_aux_loss(pred_vorticity, target_vorticity)

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
        vorticity_targets = self._compute_vorticity_targets(field_dict["velocity"])

        # 2. Generate spatiotemporal block masks
        within_masks, cross_masks = generate_masks(
            field_indices,
            grid_shape=grid_shape,
            within_mask_ratio=self.within_mask_ratio,
            device=device,
            concentration=field_dict["concentration"],
            strategy=self.mask_strategy,
        )

        # 3. Target pass: full encoder on all tokens.
        # pos_ids: (N_total, 3) → expand to (B, N_total, 3) for encoder
        full_pos_ids = pos_ids.unsqueeze(0).expand(B, -1, -1)
        with torch.no_grad():
            full_target_out = self.target_encoder(all_tokens, pos_ids=full_pos_ids)


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
        l_vorticity = w_preds.new_zeros(())
        if self.lambda_vorticity > 0:
            l_vorticity = 0.5 * self._vorticity_loss_for_predictions(
                w_preds, w_mask_ids, field_indices, vorticity_targets
            ) + 0.5 * self._vorticity_loss_for_predictions(
                c_preds, c_mask_ids, field_indices, vorticity_targets
            )

        l_future = w_preds.new_zeros(())
        l_spectral = w_preds.new_zeros(())
        l_latent_kl = w_preds.new_zeros(())
        l_action = w_preds.new_zeros(())
        if self.lambda_future > 0:
            split_t = grid_shape[0] // 2
            context_ids = torch.nonzero(pos_ids[:, 0] < split_t, as_tuple=False).squeeze(-1)
            future_ids = torch.nonzero(pos_ids[:, 0] >= split_t, as_tuple=False).squeeze(-1)

            f_vis_ids = batch_mask_indices(context_ids, B)
            f_mask_ids = batch_mask_indices(future_ids, B)
            f_visible_tokens = self._gather_tokens(all_tokens, f_vis_ids)
            f_pos_ids = pos_ids[f_vis_ids]
            f_encoder_out = self.encoder(f_visible_tokens, pos_ids=f_pos_ids)
            f_targets = F.normalize(
                self._gather_tokens(full_target_out, f_mask_ids).float(), dim=-1
            )
            context_summary = f_encoder_out.mean(dim=1)
            future_summary = f_targets.mean(dim=1)
            if self.future_mode == "noisy":
                timesteps = torch.randint(
                    0, self.num_diffusion_steps, (B,), device=device, dtype=torch.long
                )
                noise = torch.randn_like(f_targets)
                noisy_future = self._q_sample_future(f_targets, timesteps, noise)
                cond = self.future_time_mlp(self._timestep_embedding(timesteps, D))
                f_preds = self.predictor(
                    visible_tokens=f_encoder_out,
                    visible_indices=f_vis_ids,
                    masked_indices=f_mask_ids,
                    total_tokens=N_total,
                    pos_ids=pos_ids,
                    masked_token_inputs=noisy_future,
                    condition=cond,
                )
                l_future = prediction_loss(f_preds, f_targets)
            elif self.future_mode == "latent":
                prior_stats = self.future_prior(context_summary)
                mu_p, logvar_p = prior_stats.chunk(2, dim=-1)
                post_stats = self.future_posterior(
                    torch.cat([context_summary, future_summary], dim=-1)
                )
                mu_q, logvar_q = post_stats.chunk(2, dim=-1)
                z = self._reparameterize(mu_q, logvar_q)
                f_preds = self.predictor(
                    visible_tokens=f_encoder_out,
                    visible_indices=f_vis_ids,
                    masked_indices=f_mask_ids,
                    total_tokens=N_total,
                    pos_ids=pos_ids,
                    condition=z,
                )
                l_future = prediction_loss(f_preds, f_targets)
                l_latent_kl = self._kl_normal(mu_q, logvar_q, mu_p, logvar_p)
            elif self.future_mode == "action":
                action_code = self._integrated_action_code(
                    f_encoder_out, f_vis_ids, f_targets, f_mask_ids, pos_ids, split_t
                )
                f_preds = self.predictor(
                    visible_tokens=f_encoder_out,
                    visible_indices=f_vis_ids,
                    masked_indices=f_mask_ids,
                    total_tokens=N_total,
                    pos_ids=pos_ids,
                    condition=action_code,
                )
                l_future = prediction_loss(f_preds, f_targets)
                pooled_future_pred = self.action_transition(
                    torch.cat([context_summary, action_code], dim=-1)
                )
                l_action = F.mse_loss(pooled_future_pred, future_summary)
            elif self.future_mode == "hybrid":
                action_code = self._integrated_action_code(
                    f_encoder_out, f_vis_ids, f_targets, f_mask_ids, pos_ids, split_t
                )
                timesteps = torch.randint(
                    0, self.num_diffusion_steps, (B,), device=device, dtype=torch.long
                )
                noise = torch.randn_like(f_targets)
                noisy_future = self._q_sample_future(f_targets, timesteps, noise)
                time_cond = self.future_time_mlp(self._timestep_embedding(timesteps, D))
                hybrid_cond = time_cond + action_code
                f_preds = self.predictor(
                    visible_tokens=f_encoder_out,
                    visible_indices=f_vis_ids,
                    masked_indices=f_mask_ids,
                    total_tokens=N_total,
                    pos_ids=pos_ids,
                    masked_token_inputs=noisy_future,
                    condition=hybrid_cond,
                )
                l_future = prediction_loss(f_preds, f_targets)
                pooled_future_pred = self.action_transition(
                    torch.cat([context_summary, action_code], dim=-1)
                )
                l_action = F.mse_loss(pooled_future_pred, future_summary)
            else:
                f_preds = self.predictor(
                    visible_tokens=f_encoder_out,
                    visible_indices=f_vis_ids,
                    masked_indices=f_mask_ids,
                    total_tokens=N_total,
                    pos_ids=pos_ids,
                )
                l_future = prediction_loss(f_preds, f_targets)

            if self.lambda_vorticity > 0:
                l_vorticity = (2.0 * l_vorticity + self._vorticity_loss_for_predictions(
                    f_preds, f_mask_ids, field_indices, vorticity_targets
                )) / 3.0

            if self.lambda_spectral > 0:
                vel_start, vel_end = field_indices["velocity"]
                vel_preds, vel_local_ids = self._select_group_predictions(
                    f_preds, f_mask_ids, vel_start, vel_end
                )
                if vel_preds is not None:
                    pred_vorticity = self.vorticity_head(vel_preds).squeeze(-1)
                    target_vorticity = self._gather_tokens(
                        vorticity_targets.unsqueeze(-1), vel_local_ids
                    ).squeeze(-1)
                    n_t_future = grid_shape[0] - split_t
                    pred_maps = pred_vorticity.view(B, n_t_future, grid_shape[1], grid_shape[2])
                    target_maps = target_vorticity.view(B, n_t_future, grid_shape[1], grid_shape[2])
                    l_spectral = spectral_consistency_loss(pred_maps, target_maps)

        # SIGReg equally weighted over within-field and cross-field predictor
        # outputs. Subsample each independently to keep O(N^2) within memory.
        n_w = min(1024, w_preds.shape[1])
        n_c = min(1024, c_preds.shape[1])
        idx_w = torch.randperm(w_preds.shape[1], device=device)[:n_w]
        idx_c = torch.randperm(c_preds.shape[1], device=device)[:n_c]
        l_sigreg = 0.5 * sigreg_loss(w_preds[:, idx_w, :]) \
                 + 0.5 * sigreg_loss(c_preds[:, idx_c, :])

        total_loss = (
            self.lambda_within  * l_within
            + self.lambda_cross * l_cross
            + self.lambda_future * l_future
            + self.lambda_sigreg * l_sigreg
            + self.lambda_vorticity * l_vorticity
            + self.lambda_spectral * l_spectral
            + self.lambda_latent_kl * l_latent_kl
            + self.lambda_action * l_action
        )

        loss_dict = {
            "loss_total":   total_loss.detach(),
            "loss_within":  l_within.detach(),
            "loss_cross":   l_cross.detach(),
            "loss_future":  l_future.detach(),
            "sigreg_total": l_sigreg.detach(),
            "loss_vorticity": l_vorticity.detach(),
            "loss_spectral": l_spectral.detach(),
            "loss_latent_kl": l_latent_kl.detach(),
            "loss_action": l_action.detach(),
        }

        return total_loss, loss_dict

    @torch.no_grad()
    def update_target_encoder(self, momentum):
        """Updates the target encoder via exponential moving average."""
        for param, target_param in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            target_param.data.mul_(momentum).add_(param.data, alpha=1.0 - momentum)

    @torch.no_grad()
    def encode(self, field_dict, pool="mean", use_target_encoder=False):
        """
        Extract representations for evaluation (linear probe / kNN).

        Args:
            field_dict:          dict with field group tensors
            pool:                "mean" for mean pooling over all tokens
            use_target_encoder:  if True, use EMA target encoder instead of online encoder

        Returns:
            features: (B, embed_dim)
        """
        encoder = self.target_encoder if use_target_encoder else self.encoder
        all_tokens, _, _, pos_ids = self.patch_embed(field_dict)
        B = all_tokens.shape[0]
        full_pos_ids = pos_ids.unsqueeze(0).expand(B, -1, -1)
        encoder_out = encoder(all_tokens, pos_ids=full_pos_ids)

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
