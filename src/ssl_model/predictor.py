# src/model/predictor.py

"""
Lightweight prediction transformer.

Takes encoder output for visible (unmasked) tokens along with learnable mask
tokens placed at masked positions, and predicts the encoder representations
at those masked positions.

The predictor is intentionally small (2-4 layers) to force the encoder to
learn good representations rather than offloading that work to the predictor.
"""

import torch
import torch.nn as nn

from .encoder import TransformerBlock


class Predictor(nn.Module):
    """
    Predicts encoder representations for masked tokens.

    The predictor operates in a shared embedding space with the encoder
    but can optionally use a different (narrower) dimension internally
    to save parameters.

    Flow:
      1. Project visible encoder tokens to predictor dim
      2. Create learnable mask tokens at masked positions
      3. Concatenate visible + mask tokens (in original order)
      4. Run through lightweight transformer blocks
      5. Extract and return predictions at masked positions only
    """

    def __init__(
        self,
        encoder_dim=384,
        predictor_dim=192,
        depth=4,
        n_heads=6,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.predictor_dim = predictor_dim

        # Project from encoder dim to predictor dim
        self.input_proj = nn.Linear(encoder_dim, predictor_dim)

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=predictor_dim,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(predictor_dim)

        # Project back to encoder dim for loss computation
        self.output_proj = nn.Linear(predictor_dim, encoder_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, visible_tokens, visible_indices, masked_indices, total_tokens):
        """
        Args:
            visible_tokens:  (B, N_vis, encoder_dim) -- encoder output for visible tokens
            visible_indices: (B, N_vis) -- original positions of visible tokens
            masked_indices:  (B, N_mask) -- original positions of masked tokens
            total_tokens:    int -- total number of tokens (visible + masked)

        Returns:
            predictions: (B, N_mask, encoder_dim) -- predicted representations
                         for the masked positions
        """
        B, N_vis, _ = visible_tokens.shape
        N_mask = masked_indices.shape[1]

        # Project visible tokens to predictor dimension
        visible = self.input_proj(visible_tokens)  # (B, N_vis, predictor_dim)

        # Create mask tokens
        mask_tokens = self.mask_token.expand(B, N_mask, -1)  # (B, N_mask, predictor_dim)

        # Combine visible + mask tokens and restore original ordering
        all_tokens = torch.zeros(
            B, total_tokens, self.predictor_dim,
            device=visible.device, dtype=visible.dtype,
        )

        # Scatter visible tokens to their original positions
        vis_idx = visible_indices.unsqueeze(-1).expand(-1, -1, self.predictor_dim)
        all_tokens.scatter_(1, vis_idx, visible)

        # Scatter mask tokens to their original positions
        mask_idx = masked_indices.unsqueeze(-1).expand(-1, -1, self.predictor_dim)
        all_tokens.scatter_(1, mask_idx, mask_tokens)

        # Run through transformer
        for block in self.blocks:
            all_tokens = block(all_tokens)
        all_tokens = self.norm(all_tokens)

        # Extract predictions at masked positions only
        predictions = torch.gather(all_tokens, 1, mask_idx)

        # Project back to encoder dim
        predictions = self.output_proj(predictions)  # (B, N_mask, encoder_dim)

        return predictions


# ── Quick test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pred = Predictor(encoder_dim=384, predictor_dim=192, depth=4, n_heads=6)

    B, N_vis, N_mask, total = 2, 820, 7372, 8192
    visible_tokens = torch.randn(B, N_vis, 384)
    visible_indices = torch.randperm(total)[:N_vis].unsqueeze(0).expand(B, -1)
    masked_indices = torch.randperm(total)[:N_mask].unsqueeze(0).expand(B, -1)

    out = pred(visible_tokens, visible_indices, masked_indices, total)
    print(f"Predictions: {out.shape}")  # (2, 7372, 384)

    n_params = sum(p.numel() for p in pred.parameters())
    print(f"Parameters: {n_params / 1e6:.2f}M")