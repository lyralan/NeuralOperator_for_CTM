"""Simple CNN baseline."""
from __future__ import annotations

import torch.nn as nn


class CNN2d(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, width=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, out_channels, 1),
        )

    def forward(self, x):
        return self.net(x)
