"""Compare gradient fidelity (placeholder)."""
from __future__ import annotations

import torch
from .gradients import cosine_similarity


def compare_gradients(g_true, g_sur):
    rel_l2 = torch.norm(g_true - g_sur) / (torch.norm(g_true) + 1e-8)
    cos = cosine_similarity(g_true, g_sur)
    return {"rel_l2": rel_l2.item(), "cosine": cos.item()}
