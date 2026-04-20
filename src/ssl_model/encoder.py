# src/model/encoder.py

"""
Shared Vision Transformer encoder.

Processes tokens from all field groups together. This is a standard ViT
with pre-norm (LayerNorm before attention and MLP), which is the default
in modern ViT implementations.

The encoder only sees unmasked tokens during training (masked tokens are
excluded before being fed in). This follows the MAE/JEPA efficiency trick:
the heavy encoder only processes visible tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.n_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

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

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """
    Standard ViT encoder (no patch embedding, no CLS token).

    Input:  (B, N, embed_dim)   -- already embedded + position-encoded tokens
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
    ):
        super().__init__()
        self.embed_dim = embed_dim

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

    def forward(self, x):
        """
        Args:
            x: (B, N, embed_dim) -- unmasked tokens only
        Returns:
            (B, N, embed_dim) -- encoder representations
        """
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x


# ── Quick test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    encoder = ViTEncoder(embed_dim=384, depth=12, n_heads=6)

    # Simulating ~820 unmasked tokens (10% of 8192)
    x = torch.randn(2, 820, 384)
    out = encoder(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")

    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"Parameters: {n_params / 1e6:.2f}M")