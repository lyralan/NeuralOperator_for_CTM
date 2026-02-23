"""Compare PDE finite-difference gradients vs surrogate gradients w.r.t. source S."""
from __future__ import annotations

import argparse
import time
import yaml
import numpy as np
import torch

from ntops.models import FNO2d, UNet2d, CNN2d, HybridDiffusion
from ntops.pde.grid import make_grid
from ntops.pde.solver_fd import solve
from ntops.inverse.gradients import cosine_similarity


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


def sse(a, b):
    return np.sum((a - b) ** 2)


def normalize(x, mean, std):
    return (x - mean) / (std + 1e-8)


def denormalize(x, mean, std):
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
    idx = int(cfg.get("sample_index", 0))
    obs_idx = int(cfg.get("obs_index", idx))

    c0 = data["c0"][idx]
    u = data["u"][idx]
    v = data["v"][idx]
    D = data["D"][idx]
    S_true = data["S"][idx]

    S_obs = data["S"][obs_idx]

    nx = int(pcfg.get("nx", 64))
    ny = int(pcfg.get("ny", 64))
    lx = float(pcfg.get("lx", 1.0))
    ly = float(pcfg.get("ly", 1.0))
    dt = float(pcfg.get("dt", 1e-3))
    nsteps = int(pcfg.get("nsteps", 200))
    nsteps_multiplier = int(cfg.get("nsteps_multiplier", 1))
    _, _, dx, dy = make_grid(nx, ny, lx, ly)

    s_scale = float(cfg.get("S_scale", 1.0))
    S_true = S_true * s_scale

    # Observations generated from true source
    obs_times = cfg.get("obs_times", [nsteps])
    obs_times = [int(t) for t in obs_times]
    max_t = max(obs_times)
    total_steps = max_t * nsteps_multiplier
    obs_traj = solve(
        c0,
        u,
        v,
        D,
        S_obs * s_scale,
        dx,
        dy,
        dt,
        total_steps,
        save_every=1,
    )
    obs_idx = [t - 1 for t in obs_times]
    if min(obs_idx) < 0 or max(obs_idx) >= obs_traj.shape[0]:
        raise ValueError("obs_times must be between 1 and total_steps.")
    obs_list = [obs_traj[i] for i in obs_idx]

    # Load model + stats
    model = build_model(mcfg)
    device = torch.device(cfg.get("device", "cpu"))
    model.to(device)
    ckpt = torch.load(cfg["checkpoint"], map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    stats = ckpt.get("stats", None)

    # Prepare normalized inputs for surrogate
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    init_noise_std = float(cfg.get("init_noise_std", 0.0))
    S_init = S_true + rng.standard_normal(S_true.shape) * init_noise_std

    # Scale FD epsilon by source std for numerical stability
    if stats is not None:
        s_std = stats["S"][1]
    else:
        s_std = np.std(S_true) + 1e-8
    fd_eps_list = cfg.get("fd_eps_list", [cfg.get("fd_eps", 1e-3)])

    if stats is not None:
        c0n = normalize(c0, *stats["c"])
        un = normalize(u, *stats["u"])
        vn = normalize(v, *stats["v"])
        Sn = normalize(S_init, *stats["S"])
        Dn = normalize(D, *stats["D"])
        c_mean, c_std = stats["c"]
        s_mean, s_std = stats["S"]
    else:
        c0n, un, vn, Sn, Dn = c0, u, v, S_init, D
        c_std = 1.0
        s_std = 1.0

    # Surrogate gradient via autograd (rollout)
    model.eval()
    Sn_t = torch.tensor(Sn, dtype=torch.float32, device=device, requires_grad=True)
    step_stride = int(cfg.get("sur_step_stride", 10))
    x0 = np.stack([c0n, un, vn, Sn, np.ones_like(Sn) * Dn], axis=0)
    x0 = torch.from_numpy(x0[None]).float().to(device)
    x0 = x0.clone()
    x0[:, 3] = Sn_t

    with torch.enable_grad():
        if any(t % step_stride != 0 for t in obs_times):
            raise ValueError("obs_times must be multiples of sur_step_stride for surrogate rollout.")
        loss = 0.0
        c = x0[:, 0:1]
        obs_pairs = sorted(zip(obs_times, obs_list), key=lambda x: x[0])
        last_step = 0
        for t, o in obs_pairs:
            steps = int(t / step_stride)
            for _ in range(steps - last_step):
                xk = torch.cat(
                    [
                        c,
                        x0[:, 1:2],
                        x0[:, 2:3],
                        x0[:, 3:4],
                        x0[:, 4:5],
                    ],
                    dim=1,
                )
                c = model(xk)
            last_step = steps
            pred = c[0, 0]
            if stats is not None:
                pred = denormalize(pred, c_mean, c_std)
            loss = loss + torch.sum((pred - torch.tensor(o, device=device)) ** 2)
        loss.backward()
        grad_sur = Sn_t.grad.detach().cpu().numpy() / s_std  # dL/dS (physical units)
        sur_loss = loss.detach().cpu().item()

    # Finite-difference gradient on PDE for random subset of pixels
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    n_points = int(cfg.get("fd_points", 50))
    flat_idx = rng.choice(S_true.size, size=n_points, replace=False)
    grad_sur_sample = grad_sur.reshape(-1)[flat_idx]
    grad_sur_t = torch.tensor(grad_sur_sample, dtype=torch.float32)

    print("Gradient fidelity (subset):")
    print("  sur_loss_sse:", float(sur_loss))

    for eps_scale in fd_eps_list:
        eps = float(eps_scale) * s_std
        grad_fd = np.zeros(n_points, dtype=np.float64)
        for i, fi in enumerate(flat_idx):
            S_perturb = S_init.copy().reshape(-1)
            S_perturb[fi] += eps
            S_perturb = S_perturb.reshape(S_init.shape)
            obs_p = solve(
                c0, u, v, D, S_perturb, dx, dy, dt, total_steps, save_every=1
            )
            loss_p = 0.0
            for idx_t, o_true in zip(obs_idx, obs_list):
                loss_p = loss_p + sse(obs_p[idx_t], o_true)
            S_perturb = S_init.copy().reshape(-1)
            S_perturb[fi] -= eps
            S_perturb = S_perturb.reshape(S_init.shape)
            obs_m = solve(
                c0, u, v, D, S_perturb, dx, dy, dt, total_steps, save_every=1
            )
            loss_m = 0.0
            for idx_t, o_true in zip(obs_idx, obs_list):
                loss_m = loss_m + sse(obs_m[idx_t], o_true)
            grad_fd[i] = (loss_p - loss_m) / (2 * eps)

        grad_fd_t = torch.tensor(grad_fd, dtype=torch.float32)
        cos = cosine_similarity(grad_fd_t, grad_sur_t).item()
        rel_l2 = (torch.norm(grad_fd_t - grad_sur_t) / (torch.norm(grad_fd_t) + 1e-8)).item()

        # Scale-matched rel-L2
        alpha = (grad_fd_t @ grad_sur_t) / (grad_sur_t @ grad_sur_t + 1e-8)
        rel_l2_scaled = (torch.norm(grad_fd_t - alpha * grad_sur_t) / (torch.norm(grad_fd_t) + 1e-8)).item()

        print(f"  fd_eps_scale: {eps_scale}")
        print("    cosine_similarity:", cos)
        print("    rel_l2:", rel_l2)
        print("    rel_l2_scaled:", rel_l2_scaled)
        print("    fd_mean_abs:", float(np.mean(np.abs(grad_fd))))
        print("    sur_mean_abs:", float(np.mean(np.abs(grad_sur_sample))))

    print("  runtime_sec:", round(time.time() - t0, 3))


if __name__ == "__main__":
    main()
