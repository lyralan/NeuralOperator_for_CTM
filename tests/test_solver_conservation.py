import numpy as np

from ntops.pde.grid import make_grid
from ntops.pde.fields import random_wind_field, random_source, random_initial_condition
from ntops.pde.solver_fd import solve
from ntops.pde.metrics import mass_error


def test_mass_error_small():
    nx = ny = 32
    X, Y, dx, dy = make_grid(nx, ny)
    u, v = random_wind_field(nx, ny, seed=0, scale=0.1)
    S = random_source(X, Y, seed=1)
    c0 = random_initial_condition(X, Y, seed=2)
    traj = solve(c0, u, v, D=1e-4, S=S, dx=dx, dy=dy, dt=1e-3, nsteps=20)
    me = mass_error(traj, dx, dy)
    assert np.max(me) < 1e-1
