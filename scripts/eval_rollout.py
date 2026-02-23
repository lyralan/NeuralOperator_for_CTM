"""Rollout evaluation against PDE solver."""
from __future__ import annotations

import argparse
import time
import yaml
import numpy as np
import torch

from ntops.models import FNO2d, UNet2d, CNN2d, HybridDiffusion
from ntops.pde.grid import make_grid
from ntops.pde.solver_fd import solve
from ntops.pde.metrics import l2, rel_l2


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


def normalize_field(x, mean, std):
    return (x - mean) / (std + 1e-8)


def denormalize_field(x, mean, std):
    return x * std + mean


def main():
    t0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    with open(cfg["model_config"], "r") as f:
        mcfg = yaml.safe_load(f)

    with open(cfg["pde_config"], "r") as f:
        pcfg = yaml.safe_load(f)

    data = np.load(pcfg["output_path"])
    c0 = data["c0"]
    u = data["u"]
    v = data["v"]
    D = data["D"]
    S = data["S"]

    model = build_model(mcfg)
    device = torch.device(cfg.get("device", "cpu"))
    model.to(device)

    ckpt = torch.load(cfg["checkpoint"], map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    stats = ckpt.get("stats", None)

    nx = int(pcfg.get("nx", 64))
    ny = int(pcfg.get("ny", 64))
    lx = float(pcfg.get("lx", 1.0))
    ly = float(pcfg.get("ly", 1.0))
    dt = float(pcfg.get("dt", 1e-3))
    nsteps = int(pcfg.get("nsteps", 200))
    rollout_steps = int(cfg.get("rollout_steps", 5))
    step_stride = int(cfg.get("step_stride", nsteps))

    _, _, dx, dy = make_grid(nx, ny, lx, ly)

    sample_indices = cfg.get("sample_indices")
    if sample_indices is None:
        sample_indices = [0]

    per_step_l2 = []
    per_step_rel = []

    model.eval()
    with torch.no_grad():
        for idx in sample_indices:
            c = c0[idx]
            ui = u[idx]
            vi = v[idx]
            Si = S[idx]
            Di = D[idx]

            # PDE truth with larger horizon
            traj = solve(c, ui, vi, Di, Si, dx, dy, dt, step_stride * rollout_steps, save_every=step_stride)
            # Prepend initial condition
            truth = [c] + [traj[k] for k in range(traj.shape[0])]

            preds = [c]
            for _ in range(rollout_steps):
                c_in = c
                if stats is not None:
                    c_in = normalize_field(c_in, *stats["c"])
                    u_in = normalize_field(ui, *stats["u"])
                    v_in = normalize_field(vi, *stats["v"])
                    S_in = normalize_field(Si, *stats["S"])
                    D_in = normalize_field(Di, *stats["D"])
                else:
                    u_in, v_in, S_in, D_in = ui, vi, Si, Di

                x = np.stack([c_in, u_in, v_in, S_in, np.ones_like(c_in) * D_in], axis=0)
                x = torch.from_numpy(x[None]).float().to(device)

                pred = model(x)
                pred = pred[0, 0].cpu().numpy()

                if stats is not None:
                    pred = denormalize_field(pred, *stats["c"])

                c = pred
                preds.append(c)

            # compute errors per step (skip step 0)
            step_l2 = []
            step_rel = []
            for k in range(1, rollout_steps + 1):
                step_l2.append(l2(preds[k], truth[k]))
                step_rel.append(rel_l2(preds[k], truth[k]))

            per_step_l2.append(step_l2)
            per_step_rel.append(step_rel)

    per_step_l2 = np.array(per_step_l2)
    per_step_rel = np.array(per_step_rel)

    print("Rollout L2 per step (mean):", per_step_l2.mean(axis=0))
    print("Rollout rel-L2 per step (mean):", per_step_rel.mean(axis=0))
    print(f"runtime_sec: {round(time.time() - t0, 3)}")


if __name__ == "__main__":
    main()
