"""Parametrik U-Net (Ronneberger et al., 2015).

Instructor'ın sabit/basit örneğinden farkı: derinlik (depth), taban kanal
sayısı (base_channels) ve dropout config'den ayarlanabilir; encoder/decoder
simetrik olarak otomatik kurulur, skip connection boyut uyuşmazlıkları
(tek sayı H/W'de pooling sonrası) interpolasyonla güvenli şekilde ele alınır.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv3x3 -> BN -> ReLU) x 2, opsiyonel Dropout2d ile."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
        base_channels: int = 32,
        depth: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        channels = [base_channels * (2**i) for i in range(depth + 1)]  # örn. depth=4 -> [32,64,128,256,512]

        # Encoder
        self.downs = nn.ModuleList()
        prev_ch = in_channels
        for ch in channels[:-1]:
            self.downs.append(DoubleConv(prev_ch, ch, dropout))
            prev_ch = ch
        self.pool = nn.MaxPool2d(kernel_size=2)

        # Bottleneck
        self.bottleneck = DoubleConv(channels[-2], channels[-1], dropout)

        # Decoder
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for ch in reversed(channels[:-1]):
            self.ups.append(nn.ConvTranspose2d(ch * 2, ch, kernel_size=2, stride=2))
            self.up_convs.append(DoubleConv(ch * 2, ch, dropout))

        self.out_conv = nn.Conv2d(channels[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for up, up_conv, skip in zip(self.ups, self.up_convs, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = up_conv(x)

        return self.out_conv(x)  # (B, num_classes, H, W) - logits, softmax yok (CrossEntropyLoss içinde uygulanır)


if __name__ == "__main__":
    model = UNet(in_channels=3, num_classes=6, base_channels=32, depth=4)
    dummy = torch.randn(2, 3, 256, 256)
    out = model(dummy)
    print("output shape:", out.shape)  # beklenen: (2, 6, 256, 256)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parametre sayısı: {n_params:,}")
