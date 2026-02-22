"""Fourier Neural Operator (2D) baseline."""
from __future__ import annotations

import torch
import torch.nn as nn

from .layers import SpectralConv2d


class FNO2d(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, width=32, modes1=12, modes2=12, depth=4):
        super().__init__()
        self.width = width
        self.fc0 = nn.Linear(in_channels, width)

        self.convs = nn.ModuleList()
        self.ws = nn.ModuleList()
        for _ in range(depth):
            self.convs.append(SpectralConv2d(width, width, modes1, modes2))
            self.ws.append(nn.Conv2d(width, width, kernel_size=1))

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        # x: [B, C, H, W]
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  # [B, width, H, W]

        for conv, w in zip(self.convs, self.ws):
            x = conv(x) + w(x)
            x = torch.relu(x)

        x = x.permute(0, 2, 3, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)
        return x
