import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """Simple CNN encoder: (C, H, W) -> embed_dim. AdaptiveAvgPool handles any H/W."""

    def __init__(self, in_channels=11, embed_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 7, stride=2, padding=3),   # -> 32 x 112 x 112
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),            # -> 64 x 56 x 56
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),           # -> 128 x 28 x 28
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),          # -> 256 x 14 x 14
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),                               # -> 256 x 1 x 1
            nn.Flatten(),                                          # -> 256
        )
        self.embed_dim = embed_dim

    def forward(self, x):
        return self.net(x)


class SupervisedBaseline(nn.Module):
    """
    Per-frame CNN encoder + temporal mean pooling + linear regression head.

    Input: (B, T, 11, H, W)  -- T frames, 11 channels, any spatial size
    Output: (B, 2)            -- predicted [zeta, alpha]
    """

    def __init__(self, in_channels=11, embed_dim=256, n_targets=2):
        super().__init__()
        self.encoder = CNNEncoder(in_channels=in_channels, embed_dim=embed_dim)
        self.head = nn.Linear(embed_dim, n_targets)

    def forward(self, x):
        B, T, C, H, W = x.shape

        # Encode each frame independently
        x = x.reshape(B * T, C, H, W)          # (B*T, 11, 224, 224)
        features = self.encoder(x)               # (B*T, 256)
        features = features.reshape(B, T, -1)    # (B, T, 256)

        # Temporal mean pooling
        pooled = features.mean(dim=1)            # (B, 256)

        # Regression head
        out = self.head(pooled)                  # (B, 2)
        return out


if __name__ == "__main__":
    model = SupervisedBaseline()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    head_params = sum(p.numel() for p in model.head.parameters())
    print(f"Encoder params: {encoder_params / 1e3:.1f}K")
    print(f"Head params:    {head_params / 1e3:.1f}K")
    print(f"Total params:   {total_params / 1e3:.1f}K")

    # Test forward pass
    dummy = torch.randn(2, 32, 11, 224, 224)
    out = model(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}")