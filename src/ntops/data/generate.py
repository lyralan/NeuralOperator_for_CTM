"""Dataset generation entrypoints."""
from __future__ import annotations

import numpy as np

from ..pde.grid import make_grid
from ..pde.fields import random_wind_field, random_source, random_initial_condition
from ..pde.solver_fd import solve
from .io import save_npz


def generate_dataset(
    path: str,
    nx: int = 64,
    ny: int = 64,
    lx: float = 1.0,
    ly: float = 1.0,
    dt: float = 1e-3,
    nsteps: int = 200,
    nsamples: int = 100,
    D_range=(1e-4, 1e-2),
    wind_scale: float = 1.0,
    seed: int = 0,
    save_every: int = 1,
):
    X, Y, dx, dy = make_grid(nx, ny, lx, ly)
    rng = np.random.default_rng(seed)

    c0s = np.zeros((nsamples, nx, ny), dtype=np.float32)
    us = np.zeros((nsamples, nx, ny), dtype=np.float32)
    vs = np.zeros((nsamples, nx, ny), dtype=np.float32)
    Ds = np.zeros((nsamples,), dtype=np.float32)
    Ss = np.zeros((nsamples, nx, ny), dtype=np.float32)
    trajs = []

    for i in range(nsamples):
        u, v = random_wind_field(nx, ny, seed=rng.integers(1e9), scale=wind_scale)
        S = random_source(X, Y, seed=rng.integers(1e9))
        c0 = random_initial_condition(X, Y, seed=rng.integers(1e9))
        D = rng.uniform(D_range[0], D_range[1])

        traj = solve(c0, u, v, D, S, dx, dy, dt, nsteps, save_every=save_every)

        c0s[i] = c0
        us[i] = u
        vs[i] = v
        Ds[i] = D
        Ss[i] = S
        trajs.append(traj.astype(np.float32))

    trajs = np.stack(trajs, axis=0)

    save_npz(path, c0=c0s, u=us, v=vs, D=Ds, S=Ss, traj=trajs)
    return path
