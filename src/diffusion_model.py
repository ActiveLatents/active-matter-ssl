import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ssl_model.encoder import RoPE3D, TransformerBlock, ViTEncoder
from src.ssl_model.patch_embed import ChannelFactoredPatchEmbed


def timestep_embedding(timesteps, dim, max_period=10000):
    """Sinusoidal timestep embeddings."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=timesteps.device) / max(half, 1)
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class FutureLatentDenoiser(nn.Module):
    """
    Transformer denoiser over future latent tokens, conditioned on context and time.
    """

    def __init__(
        self,
        embed_dim=384,
        depth=4,
        n_heads=6,
        mlp_ratio=4.0,
        max_t=8,
        max_h=16,
        max_w=16,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.context_proj = nn.Linear(embed_dim, embed_dim)
        self.input_proj = nn.Linear(embed_dim, embed_dim)
        self.rope = RoPE3D(embed_dim // n_heads, max_t=max_t, max_h=max_h, max_w=max_w)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, noisy_future, context_latents, timesteps, future_pos_ids):
        t_emb = self.time_mlp(timestep_embedding(timesteps, self.embed_dim)).unsqueeze(1)
        context_cond = self.context_proj(context_latents.mean(dim=1, keepdim=True))

        x = self.input_proj(noisy_future) + t_emb + context_cond
        for block in self.blocks:
            x = block(x, rope=self.rope, pos_ids=future_pos_ids)
        x = self.norm(x)
        return self.out_proj(x)


class FutureLatentDiffusion(nn.Module):
    """
    Diffuse future latent tokens conditioned on clean context latent tokens.
    """

    def __init__(
        self,
        embed_dim=384,
        tube_t=2,
        patch_h=16,
        patch_w=16,
        n_frames=16,
        spatial_size=256,
        encoder_depth=12,
        encoder_heads=6,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        denoiser_depth=4,
        denoiser_heads=6,
        num_diffusion_steps=1000,
    ):
        super().__init__()
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
        self.target_encoder = copy.deepcopy(self.encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.denoiser = FutureLatentDenoiser(
            embed_dim=embed_dim,
            depth=denoiser_depth,
            n_heads=denoiser_heads,
            mlp_ratio=mlp_ratio,
            max_t=n_frames // tube_t,
            max_h=spatial_size // patch_h,
            max_w=spatial_size // patch_w,
        )

        betas = torch.linspace(1e-4, 0.02, num_diffusion_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.num_diffusion_steps = num_diffusion_steps

    def _gather_tokens(self, tokens, indices):
        d = tokens.shape[-1]
        idx = indices.unsqueeze(-1).expand(-1, -1, d)
        return torch.gather(tokens, 1, idx)

    def _split_context_future(self, pos_ids, batch_size):
        split_t = int(pos_ids[:, 0].max().item() + 1) // 2
        context_ids = torch.nonzero(pos_ids[:, 0] < split_t, as_tuple=False).squeeze(-1)
        future_ids = torch.nonzero(pos_ids[:, 0] >= split_t, as_tuple=False).squeeze(-1)
        return (
            context_ids.unsqueeze(0).expand(batch_size, -1).contiguous(),
            future_ids.unsqueeze(0).expand(batch_size, -1).contiguous(),
        )

    def _q_sample(self, x0, timesteps, noise):
        alpha_bar = self.alpha_bars[timesteps].view(-1, 1, 1)
        return alpha_bar.sqrt() * x0 + (1.0 - alpha_bar).sqrt() * noise

    def forward_diffusion(self, field_dict):
        device = next(self.parameters()).device
        all_tokens, _, _, pos_ids = self.patch_embed(field_dict)
        bsz = all_tokens.shape[0]

        context_ids, future_ids = self._split_context_future(pos_ids, bsz)
        context_tokens = self._gather_tokens(all_tokens, context_ids)
        future_tokens = self._gather_tokens(all_tokens, future_ids)

        context_pos_ids = pos_ids[context_ids]
        future_pos_ids = pos_ids[future_ids]

        context_latents = self.encoder(context_tokens, pos_ids=context_pos_ids)
        with torch.no_grad():
            target_future = self.target_encoder(future_tokens, pos_ids=future_pos_ids)
            target_future = F.layer_norm(target_future, (target_future.shape[-1],))

        timesteps = torch.randint(
            0, self.num_diffusion_steps, (bsz,), device=device, dtype=torch.long
        )
        noise = torch.randn_like(target_future)
        noisy_future = self._q_sample(target_future, timesteps, noise)
        pred_noise = self.denoiser(
            noisy_future, context_latents, timesteps, future_pos_ids
        )
        loss = F.mse_loss(pred_noise, noise)
        return loss, {
            "loss_total": loss.detach(),
            "loss_noise": loss.detach(),
        }

    @torch.no_grad()
    def update_target_encoder(self, momentum):
        for param, target_param in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            target_param.data.mul_(momentum).add_(param.data, alpha=1.0 - momentum)

    @torch.no_grad()
    def encode(self, field_dict, pool="mean", use_target_encoder=False):
        encoder = self.target_encoder if use_target_encoder else self.encoder
        all_tokens, _, _, pos_ids = self.patch_embed(field_dict)
        bsz = all_tokens.shape[0]
        full_pos_ids = pos_ids.unsqueeze(0).expand(bsz, -1, -1)
        latents = encoder(all_tokens, pos_ids=full_pos_ids)
        if pool == "mean":
            return latents.mean(dim=1)
        raise ValueError(f"Unknown pooling: {pool}")

    def param_count(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  {'trainable':16s}: {trainable / 1e6:.2f}M")
        print(f"  {'total':16s}: {total / 1e6:.2f}M")
        return {"trainable": trainable, "total": total}
