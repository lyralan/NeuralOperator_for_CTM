"""Minimal U-Net baseline."""
from __future__ import annotations

import torch
import torch.nn as nn


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.ReLU(),
    )


class UNet2d(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, base=32):
        super().__init__()
        self.down1 = conv_block(in_channels, base)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = conv_block(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.mid = conv_block(base * 2, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = conv_block(base * 2, base)

        self.out = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        m = self.mid(self.pool2(d2))
        u2 = self.up2(m)
        x = self.dec2(torch.cat([u2, d2], dim=1))
        u1 = self.up1(x)
        x = self.dec1(torch.cat([u1, d1], dim=1))
        return self.out(x)
