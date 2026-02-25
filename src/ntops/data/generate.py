"""Dataset generation entrypoints."""
from __future__ import annotations

import multiprocessing as mp
from functools import partial

import numpy as np

from ..pde.grid import make_grid
from ..pde.fields import random_wind_field, random_source, random_initial_condition
from ..pde.solver_fd import solve
from .io import save_npz


def _generate_one(i, seeds, X, Y, nx, ny, dx, dy, dt, nsteps, D_range, wind_scale, save_every):
    """Generate a single sample. Designed for use with multiprocessing."""
    seed_wind, seed_src, seed_ic, seed_D = seeds[i]
    rng_D = np.random.default_rng(seed_D)

    u, v = random_wind_field(nx, ny, seed=seed_wind, scale=wind_scale)
    S = random_source(X, Y, seed=seed_src)
    c0 = random_initial_condition(X, Y, seed=seed_ic)
    D = rng_D.uniform(D_range[0], D_range[1])

    traj = solve(c0, u, v, D, S, dx, dy, dt, nsteps, save_every=save_every)

    return i, c0, u, v, D, S, traj.astype(np.float32)


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
    num_workers: int = 0,
):
    X, Y, dx, dy = make_grid(nx, ny, lx, ly)

    # Pre-generate all random seeds so results are reproducible regardless of num_workers
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31, size=(nsamples, 4))

    if num_workers <= 0:
        num_workers = mp.cpu_count() or 1

    worker_fn = partial(
        _generate_one,
        seeds=seeds,
        X=X, Y=Y, nx=nx, ny=ny, dx=dx, dy=dy,
        dt=dt, nsteps=nsteps, D_range=D_range,
        wind_scale=wind_scale, save_every=save_every,
    )

    c0s = np.zeros((nsamples, nx, ny), dtype=np.float32)
    us = np.zeros((nsamples, nx, ny), dtype=np.float32)
    vs = np.zeros((nsamples, nx, ny), dtype=np.float32)
    Ds = np.zeros((nsamples,), dtype=np.float32)
    Ss = np.zeros((nsamples, nx, ny), dtype=np.float32)
    trajs = [None] * nsamples

    if num_workers == 1:
        for i in range(nsamples):
            idx, c0, u, v, D, S, traj = worker_fn(i)
            c0s[idx] = c0
            us[idx] = u
            vs[idx] = v
            Ds[idx] = D
            Ss[idx] = S
            trajs[idx] = traj
            if (i + 1) % 100 == 0 or i == nsamples - 1:
                print(f"  [{i + 1}/{nsamples}]")
    else:
        with mp.Pool(num_workers) as pool:
            for idx, c0, u, v, D, S, traj in pool.imap_unordered(worker_fn, range(nsamples), chunksize=4):
                c0s[idx] = c0
                us[idx] = u
                vs[idx] = v
                Ds[idx] = D
                Ss[idx] = S
                trajs[idx] = traj
                done = sum(t is not None for t in trajs)
                if done % 100 == 0 or done == nsamples:
                    print(f"  [{done}/{nsamples}]")

    trajs = np.stack(trajs, axis=0)

    save_npz(path, c0=c0s, u=us, v=vs, D=Ds, S=Ss, traj=trajs)
    return path
