import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual block with two convolutions."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual, inplace=True)


class Reconstructor(nn.Module):
    """CNN decoder: maps CLIP embedding (512-dim) to 224x224 RGB image.

    Architecture:
        FC → reshape 512x7x7
        → [ConvTranspose2d + BN + ReLU + ResBlock] × 5
        → 3x224x224 via Sigmoid
    """

    def __init__(self, embed_dim=512, base_channels=512):
        super().__init__()
        self.base_channels = base_channels

        # Project embedding to spatial feature map
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, base_channels * 7 * 7),
            nn.ReLU(inplace=True),
        )

        # Upsample decoder: 512x7x7 → 3x224x224
        # 7→14→28→56→112→224
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ResBlock(256),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResBlock(128),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResBlock(64),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResBlock(32),

            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

    def forward(self, z):
        """
        Args:
            z: (B, embed_dim) embedding tensor
        Returns:
            image: (B, 3, 224, 224) in [0, 1]
        """
        x = self.fc(z)
        x = x.view(-1, self.base_channels, 7, 7)
        x = self.decoder(x)
        return x
