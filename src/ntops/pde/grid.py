"""Grid utilities for 2D periodic domains."""
from __future__ import annotations

import numpy as np


def make_grid(nx: int, ny: int, lx: float = 1.0, ly: float = 1.0):
    """Return meshgrid X, Y and spacing for a 2D periodic domain."""
    x = np.linspace(0.0, lx, nx, endpoint=False)
    y = np.linspace(0.0, ly, ny, endpoint=False)
    dx = lx / nx
    dy = ly / ny
    X, Y = np.meshgrid(x, y, indexing="ij")
    return X, Y, dx, dy


def wavenumbers(nx: int, ny: int, lx: float = 1.0, ly: float = 1.0):
    """Return spectral wavenumbers for FFT-based operators."""
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=lx / nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=ly / ny)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    return KX, KY
