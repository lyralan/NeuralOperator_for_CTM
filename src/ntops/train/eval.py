"""Evaluation utilities."""
from __future__ import annotations

import torch
from ..pde.metrics import l2


def eval_one_step(model, batch, device):
    model.eval()
    with torch.no_grad():
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        pred = model(x)
    return l2(pred.cpu().numpy(), y.cpu().numpy())
