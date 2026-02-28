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


def train_epoch(model, dataloader, optimizer, device, unroll_steps=1, step_stride=1,
                jacobian_weight=0.0, jacobian_eps=0.01):
    model.train()
    total = 0.0
    n = 0
    for batch in dataloader:
        x, D, y = make_batch_inputs(batch, device)
        if unroll_steps <= 1 or "traj" not in batch:
            pred = model(x)
            loss = mse(pred, y)
            last_input = x
        else:
            traj = batch["traj"].to(device)  # [B, T, H, W]
            max_steps = min(unroll_steps, (traj.size(1) - 1) // step_stride)
            c = batch["c0"].unsqueeze(1).to(device)
            loss = 0.0
            for k in range(1, max_steps + 1):
                xk = torch.cat(
                    [
                        c,
                        batch["u"].unsqueeze(1).to(device),
                        batch["v"].unsqueeze(1).to(device),
                        batch["S"].unsqueeze(1).to(device),
                        batch["D"].view(-1, 1, 1, 1).expand_as(c).to(device),
                    ],
                    dim=1,
                )
                pred = model(xk)
                target = traj[:, k * step_stride].unsqueeze(1)
                loss = loss + mse(pred, target)
                c = pred
            loss = loss / max_steps
            last_input = xk
        # Jacobian sensitivity loss: penalize the model for ignoring S
        if jacobian_weight > 0:
            delta = jacobian_eps * torch.randn_like(last_input[:, 3:4])
            x_pert = last_input.clone()
            x_pert[:, 3:4] = last_input[:, 3:4] + delta
            pred_pert = model(x_pert)
            diff_sq = torch.mean((pred_pert - pred.detach()) ** 2)
            loss = loss + jacobian_weight * (-torch.log(diff_sq + 1e-8))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * y.size(0)
        n += y.size(0)
    return total / max(n, 1)


def eval_epoch(model, dataloader, device, unroll_steps=1, step_stride=1):
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in dataloader:
            x, D, y = make_batch_inputs(batch, device)
            if unroll_steps <= 1 or "traj" not in batch:
                pred = model(x)
                loss = mse(pred, y)
            else:
                traj = batch["traj"].to(device)
                max_steps = min(unroll_steps, (traj.size(1) - 1) // step_stride)
                c = batch["c0"].unsqueeze(1).to(device)
                loss = 0.0
                for k in range(1, max_steps + 1):
                    xk = torch.cat(
                        [
                            c,
                            batch["u"].unsqueeze(1).to(device),
                            batch["v"].unsqueeze(1).to(device),
                            batch["S"].unsqueeze(1).to(device),
                            batch["D"].view(-1, 1, 1, 1).expand_as(c).to(device),
                        ],
                        dim=1,
                    )
                    pred = model(xk)
                    target = traj[:, k * step_stride].unsqueeze(1)
                    loss = loss + mse(pred, target)
                    c = pred
                loss = loss / max_steps
            total += loss.item() * y.size(0)
            n += y.size(0)
    return total / max(n, 1)
