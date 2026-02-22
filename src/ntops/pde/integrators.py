"""Time integrators for PDE solvers."""
from __future__ import annotations

from typing import Callable
import numpy as np


RHS = Callable[[np.ndarray, float], np.ndarray]


def rk2_step(c: np.ndarray, t: float, dt: float, rhs: RHS) -> np.ndarray:
    k1 = rhs(c, t)
    k2 = rhs(c + 0.5 * dt * k1, t + 0.5 * dt)
    return c + dt * k2


def rk4_step(c: np.ndarray, t: float, dt: float, rhs: RHS) -> np.ndarray:
    k1 = rhs(c, t)
    k2 = rhs(c + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = rhs(c + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = rhs(c + dt * k3, t + dt)
    return c + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
