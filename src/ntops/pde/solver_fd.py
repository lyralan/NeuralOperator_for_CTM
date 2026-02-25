"""Finite-difference solver for 2D advection–diffusion."""
from __future__ import annotations

import numpy as np

from .integrators import rk4_step


def _grad_x(c, dx):
    return (np.roll(c, -1, axis=0) - np.roll(c, 1, axis=0)) / (2.0 * dx)


def _grad_y(c, dy):
    return (np.roll(c, -1, axis=1) - np.roll(c, 1, axis=1)) / (2.0 * dy)


def _laplacian(c, dx, dy):
    return (
        (np.roll(c, -1, 0) - 2.0 * c + np.roll(c, 1, 0)) / (dx ** 2)
        + (np.roll(c, -1, 1) - 2.0 * c + np.roll(c, 1, 1)) / (dy ** 2)
    )


def make_rhs(u, v, D, S, dx, dy):
    """Return RHS function for advection–diffusion with periodic BCs."""
    def rhs(c, t):
        adv = u * _grad_x(c, dx) + v * _grad_y(c, dy)
        diff = D * _laplacian(c, dx, dy)
        return -adv + diff + S
    return rhs


def solve(c0, u, v, D, S, dx, dy, dt, nsteps, save_every=1):
    """Integrate forward with RK4. Returns trajectory [T, nx, ny]."""
    rhs = make_rhs(u, v, D, S, dx, dy)
    c = c0.copy()
    traj = []
    t = 0.0
    for step in range(nsteps):
        c = rk4_step(c, t, dt, rhs)
        t += dt
        if (step + 1) % save_every == 0:
            traj.append(c.copy())
    return np.stack(traj, axis=0)
