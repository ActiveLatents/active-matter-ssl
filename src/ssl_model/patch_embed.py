# src/model/patch_embed.py

"""
Channel-factored tubelet patch embeddings.

Each physical field group (concentration, velocity, orientation, strain-rate)
gets its own 3D convolutional projection. All projections output the same
embedding dimension so tokens can be concatenated and fed into a shared
transformer.

Positional information is NOT added here as additive embeddings.  Instead,
each token's (t, h, w) grid index is returned as `pos_ids` so that 3D RoPE
can be applied inside every attention layer.

A learnable field-group identity vector (one per group) is still added here
to let the model distinguish physical field types.
"""

import torch
import torch.nn as nn

from src.dataset import FIELD_GROUPS


# ── Single field group embedder ─────────────────────────────────────────────

class TubeletEmbedding(nn.Module):
    """
    3D conv projection for one field group.

    Input:  (B, T, C_field, H, W)
    Output: (B, n_t * n_h * n_w, embed_dim)

    where:
      n_t = T // tube_t
      n_h = H // patch_h
      n_w = W // patch_w
    """

    def __init__(self, in_channels, embed_dim, tube_t=2, patch_h=16, patch_w=16):
        super().__init__()
        self.tube_t = tube_t
        self.patch_h = patch_h
        self.patch_w = patch_w

        # 3D conv: (B, C_field, T, H, W) -> (B, embed_dim, n_t, n_h, n_w)
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(tube_t, patch_h, patch_w),
            stride=(tube_t, patch_h, patch_w),
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, C_field, H, W)
        Returns:
            tokens: (B, n_tokens, embed_dim)
            grid_shape: (n_t, n_h, n_w)
        """
        # Rearrange to (B, C_field, T, H, W) for Conv3d
        x = x.permute(0, 2, 1, 3, 4)

        tokens = self.proj(x)  # (B, embed_dim, n_t, n_h, n_w)
        n_t, n_h, n_w = tokens.shape[2], tokens.shape[3], tokens.shape[4]

        # Flatten spatial-temporal grid to sequence
        tokens = tokens.flatten(2).transpose(1, 2)  # (B, n_tokens, embed_dim)

        return tokens, (n_t, n_h, n_w)


# ── Multi-field patch embedding ─────────────────────────────────────────────

class ChannelFactoredPatchEmbed(nn.Module):
    """
    Creates separate tubelet embeddings for each field group,
    then adds positional and field-type encodings.

    Input:  dict with keys from FIELD_GROUPS, each (B, T, C_field, H, W)
    Output: (B, total_tokens, embed_dim), field_indices dict, grid_shape
    """

    # Ordered list of field group names (order matters for token concatenation)
    GROUP_NAMES = ["concentration", "velocity", "orientation", "strain_rate"]

    def __init__(
        self,
        embed_dim=384,
        tube_t=2,
        patch_h=16,
        patch_w=16,
        n_frames=16,
        spatial_size=256,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Grid dimensions after patching
        self.n_t = n_frames // tube_t
        self.n_h = spatial_size // patch_h
        self.n_w = spatial_size // patch_w
        self.tokens_per_group = self.n_t * self.n_h * self.n_w

        # One tubelet embedding per field group
        field_channels = {
            name: (end - start) for name, (start, end) in FIELD_GROUPS.items()
        }
        self.embeds = nn.ModuleDict({
            name: TubeletEmbedding(
                in_channels=field_channels[name],
                embed_dim=embed_dim,
                tube_t=tube_t,
                patch_h=patch_h,
                patch_w=patch_w,
            )
            for name in self.GROUP_NAMES
        })

        # Field group identity embeddings (semantic type, not spatial position)
        self.field_embed = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1, 1, embed_dim))
            for name in self.GROUP_NAMES
        })

        self._init_weights()

    def _init_weights(self):
        for name in self.GROUP_NAMES:
            nn.init.trunc_normal_(self.field_embed[name], std=0.02)

    def _make_pos_ids(self, n_t, n_h, n_w, device):
        """
        Return (t, h, w) grid indices for every token in one field group.

        Takes actual grid dimensions from the conv output so this works
        correctly for any input length (full sequence or half-sequence).

        Returns: (n_t * n_h * n_w, 3)
        """
        t = torch.arange(n_t, device=device)
        h = torch.arange(n_h, device=device)
        w = torch.arange(n_w, device=device)
        grid_t, grid_h, grid_w = torch.meshgrid(t, h, w, indexing="ij")
        return torch.stack([grid_t.flatten(), grid_h.flatten(), grid_w.flatten()], dim=-1)

    def forward(self, field_dict):
        """
        Args:
            field_dict: dict with keys matching GROUP_NAMES,
                        each value is (B, T, C_field, H, W)
        Returns:
            all_tokens:    (B, total_tokens, embed_dim)
            field_indices: dict mapping group name to (start_idx, end_idx)
            grid_shape:    (n_t, n_h, n_w) tuple — actual dims from conv output
            pos_ids:       (total_tokens, 3) — (t, h, w) index per token,
                           shared across all batches and field groups
        """
        device = next(iter(field_dict.values())).device
        all_tokens = []
        field_indices = {}
        offset = 0
        grid_shape = None

        for name in self.GROUP_NAMES:
            x = field_dict[name]
            tokens, grid_shape = self.embeds[name](x)

            tokens = tokens + self.field_embed[name]

            n_tokens = tokens.shape[1]
            field_indices[name] = (offset, offset + n_tokens)
            offset += n_tokens

            all_tokens.append(tokens)

        all_tokens = torch.cat(all_tokens, dim=1)  # (B, total_tokens, embed_dim)

        # Build pos_ids from the actual conv output shape, not self.n_t.
        # This handles both full-sequence (eval) and half-sequence (training) calls.
        n_t, n_h, n_w = grid_shape
        group_pos = self._make_pos_ids(n_t, n_h, n_w, device)  # (tokens_per_group, 3)
        n_groups = len(self.GROUP_NAMES)
        pos_ids = group_pos.repeat(n_groups, 1)  # (total_tokens, 3)

        return all_tokens, field_indices, grid_shape, pos_ids


# ── Quick test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    embed = ChannelFactoredPatchEmbed(
        embed_dim=384, tube_t=2, patch_h=16, patch_w=16,
        n_frames=16, spatial_size=256,
    )

    # Fake input matching SSL dataset output
    B = 2
    field_dict = {
        "concentration": torch.randn(B, 16, 1, 256, 256),
        "velocity":      torch.randn(B, 16, 2, 256, 256),
        "orientation":   torch.randn(B, 16, 4, 256, 256),
        "strain_rate":   torch.randn(B, 16, 4, 256, 256),
    }

    tokens, field_indices, grid_shape, pos_ids = embed(field_dict)

    print(f"Total tokens: {tokens.shape}")   # (2, 8192, 384)
    print(f"Grid shape:   {grid_shape}")     # (8, 16, 16)
    print(f"pos_ids:      {pos_ids.shape}")  # (8192, 3)
    print(f"Field indices:")
    for name, (s, e) in field_indices.items():
        print(f"  {name:15s}: [{s}, {e})  ({e - s} tokens)")

    n_params = sum(p.numel() for p in embed.parameters())
    print(f"Parameters:   {n_params / 1e6:.2f}M")