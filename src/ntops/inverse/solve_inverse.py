"""Inverse solver scaffold (placeholder)."""
from __future__ import annotations

import torch


def solve(model, init, obs, steps=50, lr=1e-2):
    x = init.clone().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)
    for _ in range(steps):
        pred = model(x)
        loss = torch.mean((pred - obs) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return x.detach()
