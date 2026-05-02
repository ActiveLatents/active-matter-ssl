import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNetFrameEncoder(nn.Module):
    """ResNet18 backbone adapted for 11-channel simulation frames."""

    def __init__(self, in_channels=11):
        super().__init__()
        backbone = resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.embed_dim = 512

    def forward(self, x):
        return self.backbone(x)


class SupervisedBaseline(nn.Module):
    """
    Per-frame ResNet18 encoder + temporal mean pooling + linear regression head.

    Input:  (B, T, 11, H, W)
    Output: (B, 2) predicted [zeta, alpha]
    """

    def __init__(self, in_channels=11, n_targets=2):
        super().__init__()
        self.encoder = ResNetFrameEncoder(in_channels=in_channels)
        self.head = nn.Linear(self.encoder.embed_dim, n_targets)

    def forward(self, x):
        bsz, n_frames, n_channels, height, width = x.shape
        x = x.reshape(bsz * n_frames, n_channels, height, width)
        features = self.encoder(x)
        features = features.reshape(bsz, n_frames, -1)
        pooled = features.mean(dim=1)
        return self.head(pooled)


if __name__ == "__main__":
    model = SupervisedBaseline()
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    head_params = sum(p.numel() for p in model.head.parameters())
    print(f"Encoder params: {encoder_params / 1e6:.2f}M")
    print(f"Head params:    {head_params / 1e3:.1f}K")
    print(f"Total params:   {total_params / 1e6:.2f}M")

    dummy = torch.randn(2, 16, 11, 256, 256)
    out = model(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}")
