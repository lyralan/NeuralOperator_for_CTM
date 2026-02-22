"""Metrics for PDE rollouts and physical consistency."""
from __future__ import annotations

import numpy as np


def l2(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


def rel_l2(a, b, eps=1e-8):
    return l2(a, b) / (np.sqrt(np.mean(b ** 2)) + eps)


def mass(c, dx, dy):
    return np.sum(c) * dx * dy


def mass_error(traj, dx, dy):
    m0 = mass(traj[0], dx, dy)
    ms = np.array([mass(c, dx, dy) for c in traj])
    return np.abs(ms - m0) / (np.abs(m0) + 1e-8)


def spectral_error(a, b):
    fa = np.fft.rfftn(a)
    fb = np.fft.rfftn(b)
    return l2(np.abs(fa), np.abs(fb))
