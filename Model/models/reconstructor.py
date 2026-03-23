import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================== Basic Reconstructor (backward compat) ========================

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
    """Basic CNN decoder (kept for loading old checkpoints).

    Architecture:
        FC → reshape 512x7x7
        → [ConvTranspose2d + BN + ReLU + ResBlock] × 5
        → 3x224x224 via Sigmoid
    """

    def __init__(self, embed_dim=512, base_channels=512):
        super().__init__()
        self.base_channels = base_channels

        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, base_channels * 7 * 7),
            nn.ReLU(inplace=True),
        )

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
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, self.base_channels, 7, 7)
        x = self.decoder(x)
        return x


# ======================== Improved Reconstructor ========================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.SiLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        B, C, _, _ = x.shape
        scale = self.pool(x).view(B, C)
        scale = self.fc(scale).view(B, C, 1, 1)
        return x * scale


class ResBlockV2(nn.Module):
    """Improved residual block: GroupNorm + SiLU + SE attention.

    Supports channel change via 1x1 skip convolution.
    """

    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.se = SEBlock(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = F.silu(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        h = self.se(h)
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    """Multi-head self-attention for 2D feature maps.

    Used at low spatial resolutions (7x7, 14x14) for global context.
    """

    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv.unbind(dim=1)  # each: (B, heads, head_dim, N)

        attn = torch.einsum('bhdn,bhdm->bhnm', q, k) * self.scale
        attn = attn.softmax(dim=-1)

        out = torch.einsum('bhnm,bhdm->bhdn', attn, v)
        out = out.reshape(B, C, H, W)
        return x + self.proj(out)


class ImprovedReconstructor(nn.Module):
    """Enhanced embedding-to-image decoder with attention and modern design.

    Improvements over basic Reconstructor:
    - Bilinear upsampling + Conv (avoids checkerboard artifacts)
    - GroupNorm instead of BatchNorm (more stable for generation)
    - Self-attention at low resolutions (7x7, 14x14) for global coherence
    - SE channel attention in all residual blocks
    - Two residual blocks per resolution level
    - SiLU activation (smoother gradients than ReLU)

    Architecture:
        FC → reshape base_ch × 7 × 7
        → [SelfAttn + ResBlockV2 × 2 + Upsample] × 5
        → GroupNorm + SiLU + Conv → 3 × 224 × 224 via Sigmoid
    """

    def __init__(self, embed_dim=768, base_channels=512,
                 num_res_blocks=2, dropout=0.0,
                 attention_resolutions=(7, 14)):
        super().__init__()
        self.base_channels = base_channels
        self.attention_resolutions = set(attention_resolutions)

        # Projection: embedding → spatial feature map
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, base_channels * 7 * 7),
        )

        # Channel schedule for each resolution level
        # 7×7(512) → 14×14(256) → 28×28(128) → 56×56(64) → 112×112(32) → 224×224(3)
        ch_list = [base_channels, 256, 128, 64, 32]
        resolutions = [7, 14, 28, 56, 112]

        self.stages = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        for i, (ch, res) in enumerate(zip(ch_list, resolutions)):
            blocks = nn.ModuleList()
            # Self-attention at specified low resolutions
            if res in self.attention_resolutions:
                blocks.append(SelfAttention2d(ch, num_heads=min(8, ch // 32)))
            # Residual blocks
            for _ in range(num_res_blocks):
                blocks.append(ResBlockV2(ch, ch, dropout))
            self.stages.append(blocks)

            # Upsampling: bilinear interpolation + conv
            next_ch = ch_list[i + 1] if i + 1 < len(ch_list) else 32
            self.upsamplers.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(ch, next_ch, 3, 1, 1),
            ))

        # Final output head at 224×224
        self.head = nn.Sequential(
            nn.GroupNorm(32, 32),
            nn.SiLU(),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        """
        Args:
            z: (B, embed_dim) embedding tensor
        Returns:
            image: (B, 3, 224, 224) in [0, 1]
        """
        B = z.shape[0]
        x = self.proj(z).view(B, self.base_channels, 7, 7)

        for stage, up in zip(self.stages, self.upsamplers):
            for block in stage:
                x = block(x)
            x = up(x)

        return self.head(x)
