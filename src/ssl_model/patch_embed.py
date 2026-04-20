# src/model/patch_embed.py

"""
Channel-factored tubelet patch embeddings.

Each physical field group (concentration, velocity, orientation, strain-rate)
gets its own 3D convolutional projection. All projections output the same
embedding dimension so tokens can be concatenated and fed into a shared
transformer.

Positional information is added via three summed embeddings:
  - spatial position  (shared across groups and time)
  - temporal position (shared across groups and space)
  - field group identity (one learnable vector per group)
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

        # Positional embeddings (shared across field groups)
        self.spatial_embed = nn.Parameter(
            torch.zeros(1, self.n_h * self.n_w, embed_dim)
        )
        self.temporal_embed = nn.Parameter(
            torch.zeros(1, self.n_t, embed_dim)
        )

        # Field group identity embeddings
        self.field_embed = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1, 1, embed_dim))
            for name in self.GROUP_NAMES
        })

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.spatial_embed, std=0.02)
        nn.init.trunc_normal_(self.temporal_embed, std=0.02)
        for name in self.GROUP_NAMES:
            nn.init.trunc_normal_(self.field_embed[name], std=0.02)

    def _add_position_encoding(self, tokens, field_name):
        """
        Add spatial + temporal + field-type position encoding to tokens.

        tokens: (B, n_t * n_h * n_w, embed_dim)
        """
        B = tokens.shape[0]
        n_t, n_hw = self.n_t, self.n_h * self.n_w

        # Reshape to (B, n_t, n_hw, D) to add positional embeddings
        tokens = tokens.view(B, n_t, n_hw, -1)

        # Add spatial embedding: broadcast over batch and time
        tokens = tokens + self.spatial_embed.unsqueeze(1)  # (1, 1, n_hw, D)

        # Add temporal embedding: broadcast over batch and space
        tokens = tokens + self.temporal_embed.unsqueeze(2)  # (1, n_t, 1, D)

        # Flatten back to (B, n_t * n_hw, D)
        tokens = tokens.view(B, n_t * n_hw, -1)

        # Add field group identity
        tokens = tokens + self.field_embed[field_name]

        return tokens

    def forward(self, field_dict):
        """
        Args:
            field_dict: dict with keys matching GROUP_NAMES,
                        each value is (B, T, C_field, H, W)
        Returns:
            all_tokens:    (B, total_tokens, embed_dim)
            field_indices: dict mapping group name to (start_idx, end_idx)
                           into the token sequence
            grid_shape:    (n_t, n_h, n_w) tuple
        """
        all_tokens = []
        field_indices = {}
        offset = 0

        for name in self.GROUP_NAMES:
            x = field_dict[name]
            tokens, grid_shape = self.embeds[name](x)
            tokens = self._add_position_encoding(tokens, name)

            n_tokens = tokens.shape[1]
            field_indices[name] = (offset, offset + n_tokens)
            offset += n_tokens

            all_tokens.append(tokens)

        all_tokens = torch.cat(all_tokens, dim=1)  # (B, total_tokens, embed_dim)

        return all_tokens, field_indices, grid_shape


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

    tokens, field_indices, grid_shape = embed(field_dict)

    print(f"Total tokens: {tokens.shape}")  # (2, 8192, 384)
    print(f"Grid shape:   {grid_shape}")    # (8, 16, 16)
    print(f"Field indices:")
    for name, (s, e) in field_indices.items():
        print(f"  {name:15s}: [{s}, {e})  ({e - s} tokens)")

    n_params = sum(p.numel() for p in embed.parameters())
    print(f"Parameters:   {n_params / 1e6:.2f}M")