"""Field generators for wind, sources, and initial conditions."""
from __future__ import annotations

import numpy as np


def gaussian_blob(X, Y, x0=0.5, y0=0.5, sigma=0.1, amp=1.0):
    r2 = (X - x0) ** 2 + (Y - y0) ** 2
    return amp * np.exp(-0.5 * r2 / (sigma ** 2))


def random_wind_field(nx: int, ny: int, seed: int | None = None, scale=1.0):
    """Generate a smooth random wind field (u, v) on a grid."""
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((nx, ny))
    v = rng.standard_normal((nx, ny))
    for _ in range(2):
        u = 0.25 * (np.roll(u, 1, 0) + np.roll(u, -1, 0) + np.roll(u, 1, 1) + np.roll(u, -1, 1))
        v = 0.25 * (np.roll(v, 1, 0) + np.roll(v, -1, 0) + np.roll(v, 1, 1) + np.roll(v, -1, 1))
    return scale * u, scale * v


def random_source(X, Y, seed: int | None = None):
    rng = np.random.default_rng(seed)
    x0, y0 = rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)
    sigma = rng.uniform(0.05, 0.2)
    amp = rng.uniform(0.5, 1.5)
    return gaussian_blob(X, Y, x0=x0, y0=y0, sigma=sigma, amp=amp)


def random_initial_condition(X, Y, seed: int | None = None):
    rng = np.random.default_rng(seed)
    x0, y0 = rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)
    sigma = rng.uniform(0.05, 0.2)
    amp = rng.uniform(0.5, 1.5)
    return gaussian_blob(X, Y, x0=x0, y0=y0, sigma=sigma, amp=amp)
