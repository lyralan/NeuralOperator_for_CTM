"""Loss functions."""
from __future__ import annotations

import torch


def mse(pred, target):
    return torch.mean((pred - target) ** 2)
