"""Inverse-problem objectives (placeholder)."""
from __future__ import annotations

import torch


def objective_mse(pred, obs):
    return torch.mean((pred - obs) ** 2)
