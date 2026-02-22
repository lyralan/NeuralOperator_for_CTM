"""Optimizer helpers."""
from __future__ import annotations

import torch


def make_optimizer(params, lr=1e-3, weight_decay=0.0):
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
