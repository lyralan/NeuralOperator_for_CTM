"""Long-horizon rollout utilities (placeholder)."""
from __future__ import annotations

import torch


def rollout(model, x0, steps):
    """Iteratively apply a model to produce a rollout."""
    model.eval()
    traj = [x0]
    x = x0
    with torch.no_grad():
        for _ in range(steps):
            x = model(x)
            traj.append(x)
    return torch.stack(traj, dim=0)
