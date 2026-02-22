"""Training entrypoint."""
from __future__ import annotations

import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from ntops.data.dataset import NPZTrajectoryDataset
from ntops.models import FNO2d, UNet2d, CNN2d, HybridDiffusion
from ntops.train.loop import train_epoch, eval_epoch
from ntops.train.optim import make_optimizer


def build_model(cfg):
    mcfg = cfg["model"]
    mtype = mcfg.get("type", "fno")
    if mtype == "fno":
        return FNO2d(
            in_channels=mcfg.get("in_channels", 4),
            out_channels=mcfg.get("out_channels", 1),
            width=mcfg.get("width", 32),
            modes1=mcfg.get("modes1", 12),
            modes2=mcfg.get("modes2", 12),
            depth=mcfg.get("depth", 4),
        )
    if mtype == "unet":
        return UNet2d(
            in_channels=mcfg.get("in_channels", 4),
            out_channels=mcfg.get("out_channels", 1),
            base=mcfg.get("base", 32),
        )
    if mtype == "cnn":
        return CNN2d(
            in_channels=mcfg.get("in_channels", 4),
            out_channels=mcfg.get("out_channels", 1),
            width=mcfg.get("width", 32),
        )
    if mtype == "hybrid":
        return HybridDiffusion(
            in_channels=mcfg.get("in_channels", 4),
            hidden=mcfg.get("hidden", 32),
        )
    raise ValueError(f"Unknown model type: {mtype}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    dataset = NPZTrajectoryDataset(cfg["data_path"], normalize=False)
    val_split = cfg["train"].get("val_split", 0.1)
    seed = cfg["train"].get("seed", 0)
    n_total = len(dataset)
    gen = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_total, generator=gen).numpy()
    if val_split > 0:
        n_val = int(n_total * val_split)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        train_ds = NPZTrajectoryDataset(cfg["data_path"], normalize=False, indices=train_indices)
        val_ds = NPZTrajectoryDataset(cfg["data_path"], normalize=False, indices=val_indices)
    else:
        train_ds, val_ds = NPZTrajectoryDataset(cfg["data_path"], normalize=False), None

    if cfg["train"].get("normalize", True):
        stats = train_ds.compute_stats()
        train_ds.set_stats(stats)
        train_ds.set_normalize(True)
        if val_ds is not None:
            val_ds.set_stats(stats)
            val_ds.set_normalize(True)

    num_workers = cfg["train"].get("num_workers", 0)
    pin_memory = cfg["train"].get("pin_memory", False)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg["train"]["batch_size"],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    model = build_model(cfg)
    device = torch.device(cfg["train"].get("device", "cpu"))
    model.to(device)

    # Baselines (on current dataset scale)
    with torch.no_grad():
        zeros_loss = 0.0
        c0_loss = 0.0
        n = 0
        for batch in train_loader:
            y = batch["y"].unsqueeze(1)
            zeros = torch.zeros_like(y)
            c0 = batch["c0"].unsqueeze(1)
            zeros_loss += torch.mean((zeros - y) ** 2).item() * y.size(0)
            c0_loss += torch.mean((c0 - y) ** 2).item() * y.size(0)
            n += y.size(0)
        zeros_loss /= max(n, 1)
        c0_loss /= max(n, 1)
        print(f"Baseline | zeros_mse={zeros_loss:.6f} | c0_mse={c0_loss:.6f}")

    optimizer = make_optimizer(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"].get("weight_decay", 0.0))

    scheduler = None
    scfg = cfg["train"].get("scheduler")
    if scfg:
        stype = scfg.get("type", "step")
        if stype == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scfg.get("step_size", 10),
                gamma=scfg.get("gamma", 0.5),
            )
        elif stype == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cfg["train"]["epochs"],
                eta_min=scfg.get("eta_min", 0.0),
            )
        else:
            raise ValueError(f"Unknown scheduler type: {stype}")

    epochs = cfg["train"]["epochs"]
    best_val = float("inf")
    patience = cfg["train"].get("early_stopping", {}).get("patience", 0)
    min_delta = cfg["train"].get("early_stopping", {}).get("min_delta", 0.0)
    bad_epochs = 0

    for ep in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, device)
        if val_loader is not None:
            val_loss = eval_epoch(model, val_loader, device)
            print(f"Epoch {ep:03d} | train_loss={loss:.6f} | val_loss={val_loss:.6f}")
            if patience > 0:
                if val_loss < best_val - min_delta:
                    best_val = val_loss
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= patience:
                        print(f"Early stopping at epoch {ep} (best_val={best_val:.6f})")
                        break
        else:
            print(f"Epoch {ep:03d} | train_loss={loss:.6f}")
        if scheduler is not None:
            scheduler.step()

    # Plot a few predictions vs targets from validation set
    if val_loader is not None and cfg["train"].get("plot_predictions", True):
        plot_dir = cfg["train"].get("plot_dir", "outputs")
        os.makedirs(plot_dir, exist_ok=True)
        batch = next(iter(val_loader))
        x = batch["c0"].unsqueeze(1)
        u = batch["u"].unsqueeze(1)
        v = batch["v"].unsqueeze(1)
        S = batch["S"].unsqueeze(1)
        D = batch["D"].view(-1, 1, 1, 1).expand_as(x)
        x = torch.cat([x, u, v, S, D], dim=1).to(device)
        y = batch["y"].unsqueeze(1).to(device)
        with torch.no_grad():
            pred = model(x)

        # denormalize if needed
        if train_ds.normalize:
            m, s = train_ds.stats["y"]
            pred = pred * s + m
            y = y * s + m

        pred = pred.cpu()
        y = y.cpu()
        nsamples = min(3, pred.size(0))
        for i in range(nsamples):
            fig, axes = plt.subplots(1, 2, figsize=(6, 3))
            axes[0].imshow(y[i, 0], origin="lower")
            axes[0].set_title("Target")
            axes[0].axis("off")
            axes[1].imshow(pred[i, 0], origin="lower")
            axes[1].set_title("Prediction")
            axes[1].axis("off")
            fig.tight_layout()
            fig.savefig(os.path.join(plot_dir, f"pred_vs_true_{i}.png"), dpi=150)
            plt.close(fig)


if __name__ == "__main__":
    main()
