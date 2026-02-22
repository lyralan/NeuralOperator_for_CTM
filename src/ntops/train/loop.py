"""Training loop utilities."""
from __future__ import annotations

import torch

from .losses import mse


def make_batch_inputs(batch, device):
    # Stack c0, u, v, S into channels. D is broadcast.
    c0 = batch["c0"].unsqueeze(1)
    u = batch["u"].unsqueeze(1)
    v = batch["v"].unsqueeze(1)
    S = batch["S"].unsqueeze(1)
    D = batch["D"].view(-1, 1, 1, 1).expand_as(c0)
    x = torch.cat([c0, u, v, S, D], dim=1)
    x = x.to(device)
    D = D.to(device)
    y = batch["y"].unsqueeze(1).to(device)
    return x, D, y


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total = 0.0
    n = 0
    for batch in dataloader:
        x, D, y = make_batch_inputs(batch, device)
        # optional: concatenate D as channel if desired
        pred = model(x)
        loss = mse(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * y.size(0)
        n += y.size(0)
    return total / max(n, 1)


def eval_epoch(model, dataloader, device):
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in dataloader:
            x, D, y = make_batch_inputs(batch, device)
            pred = model(x)
            loss = mse(pred, y)
            total += loss.item() * y.size(0)
            n += y.size(0)
    return total / max(n, 1)
