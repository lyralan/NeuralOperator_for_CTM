"""Dataset transforms."""
from __future__ import annotations

import numpy as np


def normalize(x, mean, std, eps=1e-8):
    return (x - mean) / (std + eps)


def denormalize(x, mean, std):
    return x * std + mean


def compute_stats(x):
    return np.mean(x), np.std(x)
