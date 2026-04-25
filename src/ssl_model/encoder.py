# src/model/encoder.py

"""
Shared Vision Transformer encoder with 3D RoPE.

Processes tokens from all field groups together. Positional information is
encoded via 3D Rotary Position Embeddings (RoPE) applied to Q and K in every
attention layer — no additive positional embeddings are used.

The encoder only sees unmasked tokens during training (masked tokens are
excluded before being fed in). This follows the MAE/JEPA efficiency trick:
the heavy encoder only processes visible tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class RoPE3D(nn.Module):
    """
    3D Rotary Position Embedding for spatiotemporal transformer tokens.

    Splits head_dim into [dim_t, dim_h, dim_w] portions and applies
    independent sinusoidal rotations to each coordinate.  No learnable
    parameters — frequencies are fixed buffers.

    head_dim must be divisible by 4 so each part stays even.
    """

    def __init__(self, head_dim, max_t=8, max_h=16, max_w=16, base=10000):
        super().__init__()
        assert head_dim % 4 == 0, "head_dim must be divisible by 4"
        # Allocate 1/4 of head_dim to temporal, 3/8 each to h and w.
        self.dim_t = head_dim // 4
        self.dim_h = (head_dim - self.dim_t) // 2
        self.dim_w = head_dim - self.dim_t - self.dim_h

        self.register_buffer("cos_t", self._make_cos(self.dim_t, max_t, base))
        self.register_buffer("sin_t", self._make_sin(self.dim_t, max_t, base))
        self.register_buffer("cos_h", self._make_cos(self.dim_h, max_h, base))
        self.register_buffer("sin_h", self._make_sin(self.dim_h, max_h, base))
        self.register_buffer("cos_w", self._make_cos(self.dim_w, max_w, base))
        self.register_buffer("sin_w", self._make_sin(self.dim_w, max_w, base))

    @staticmethod
    def _angles(dim, max_pos, base):
        half = dim // 2
        theta = 1.0 / (base ** (torch.arange(half, dtype=torch.float32) / half))
        pos = torch.arange(max_pos, dtype=torch.float32)
        # Interleave each angle for adjacent pairs: [θ_0,θ_0, θ_1,θ_1, ...]
        return torch.outer(pos, theta).repeat_interleave(2, dim=-1)  # (max_pos, dim)

    def _make_cos(self, dim, max_pos, base):
        return self._angles(dim, max_pos, base).cos()

    def _make_sin(self, dim, max_pos, base):
        return self._angles(dim, max_pos, base).sin()

    @staticmethod
    def _rotate_half(x):
        # Pairs (x_{2i}, x_{2i+1}) → (-x_{2i+1}, x_{2i})
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        return torch.stack([-x2, x1], dim=-1).flatten(-2)

    def apply(self, x, pos_ids):
        """
        Apply 3D RoPE in-place to Q or K.

        x:       (B, n_heads, N, head_dim)
        pos_ids: (B, N, 3)  [t_idx, h_idx, w_idx] per token

        Returns tensor of same shape.
        """
        t_idx, h_idx, w_idx = pos_ids[..., 0], pos_ids[..., 1], pos_ids[..., 2]

        # (B, N, dim_x) → unsqueeze head dim → (B, 1, N, dim_x)
        cos_t = self.cos_t[t_idx].unsqueeze(1)
        sin_t = self.sin_t[t_idx].unsqueeze(1)
        cos_h = self.cos_h[h_idx].unsqueeze(1)
        sin_h = self.sin_h[h_idx].unsqueeze(1)
        cos_w = self.cos_w[w_idx].unsqueeze(1)
        sin_w = self.sin_w[w_idx].unsqueeze(1)

        # Concatenate along head_dim: (B, 1, N, head_dim)
        cos = torch.cat([cos_t, cos_h, cos_w], dim=-1)
        sin = torch.cat([sin_t, sin_h, sin_w], dim=-1)

        return x * cos + self._rotate_half(x) * sin


class Attention(nn.Module):
    """Multi-head self-attention with Flash Attention (O(N) memory)."""

    def __init__(self, dim, n_heads=6, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.attn_drop_p = attn_drop

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope=None, pos_ids=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.n_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        if rope is not None and pos_ids is not None:
            q = rope.apply(q, pos_ids)
            k = rope.apply(k, pos_ids)

        # Flash Attention: O(N) memory, never materializes the N×N matrix
        dropout_p = self.attn_drop_p if self.training else 0.0
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LN -> Attn -> residual -> LN -> MLP -> residual."""

    def __init__(
        self, dim, n_heads=6, mlp_ratio=4.0, qkv_bias=True,
        drop=0.0, attn_drop=0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim, n_heads=n_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, x, rope=None, pos_ids=None):
        x = x + self.attn(self.norm1(x), rope=rope, pos_ids=pos_ids)
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """
    Standard ViT encoder (no patch embedding, no CLS token).

    Positional information is injected via 3D RoPE inside every attention
    layer rather than as additive input embeddings.

    Input:  (B, N, embed_dim)   -- embedded tokens (no position added yet)
            (B, N, 3)           -- per-token (t, h, w) grid indices
    Output: (B, N, embed_dim)   -- contextualized representations
    """

    def __init__(
        self,
        embed_dim=384,
        depth=12,
        n_heads=6,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        max_t=8,
        max_h=16,
        max_w=16,
        use_checkpointing=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_checkpointing = use_checkpointing
        head_dim = embed_dim // n_heads

        self.rope = RoPE3D(head_dim, max_t=max_t, max_h=max_h, max_w=max_w)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

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

    def forward(self, x, pos_ids=None):
        """
        Args:
            x:       (B, N, embed_dim) -- unmasked tokens only
            pos_ids: (B, N, 3)         -- (t, h, w) grid index per token
        Returns:
            (B, N, embed_dim) -- encoder representations
        """
        rope = self.rope if pos_ids is not None else None
        for block in self.blocks:
            if self.use_checkpointing and self.training:
                x = checkpoint(block, x, rope, pos_ids, use_reentrant=False)
            else:
                x = block(x, rope=rope, pos_ids=pos_ids)
        x = self.norm(x)
        return x


# ── Quick test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    encoder = ViTEncoder(embed_dim=384, depth=12, n_heads=6)

    # Simulating ~820 unmasked tokens (10% of 8192) with random (t,h,w) indices
    x = torch.randn(2, 820, 384)
    pos_ids = torch.stack([
        torch.randint(0, 8,  (2, 820)),
        torch.randint(0, 16, (2, 820)),
        torch.randint(0, 16, (2, 820)),
    ], dim=-1)
    out = encoder(x, pos_ids=pos_ids)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")

    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"Parameters: {n_params / 1e6:.2f}M")
