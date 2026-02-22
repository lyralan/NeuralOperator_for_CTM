import numpy as np

from ntops.pde.grid import make_grid
from ntops.pde.solver_fd import solve


def test_advection_translation_smoke():
    nx = ny = 32
    X, Y, dx, dy = make_grid(nx, ny)
    c0 = np.exp(-((X - 0.25) ** 2 + (Y - 0.25) ** 2) / 0.01)
    u = np.ones_like(c0) * 0.1
    v = np.zeros_like(c0)
    S = np.zeros_like(c0)
    traj = solve(c0, u, v, D=0.0, S=S, dx=dx, dy=dy, dt=1e-3, nsteps=5)
    assert traj.shape[0] == 5
