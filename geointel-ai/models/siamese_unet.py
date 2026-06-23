import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import DoubleConv, Down


class UpDiff(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class SharedEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        return x1, x2, x3, x4


class SiameseUNet(nn.Module):
    """Siamese U-Net with shared encoder and absolute feature difference decoder."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 32):
        super().__init__()
        self.encoder = SharedEncoder(in_channels, base_channels)
        self.up1 = UpDiff(base_channels * 12, base_channels * 4)
        self.up2 = UpDiff(base_channels * 6, base_channels * 2)
        self.up3 = UpDiff(base_channels * 3, base_channels)
        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, before, after):
        b1, b2, b3, b4 = self.encoder(before)
        a1, a2, a3, a4 = self.encoder(after)
        d1 = torch.abs(a1 - b1)
        d2 = torch.abs(a2 - b2)
        d3 = torch.abs(a3 - b3)
        d4 = torch.abs(a4 - b4)
        x = self.up1(d4, d3)
        x = self.up2(x, d2)
        x = self.up3(x, d1)
        return torch.sigmoid(self.outc(x))
