"""Gradient utilities for comparing true vs surrogate gradients (placeholder)."""
from __future__ import annotations

import torch


def cosine_similarity(a, b, eps=1e-8):
    a = a.flatten()
    b = b.flatten()
    return torch.dot(a, b) / (torch.norm(a) * torch.norm(b) + eps)
