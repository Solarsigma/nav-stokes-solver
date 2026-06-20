import boundary_conditions
import numpy as np
import pytest
from tests.utils import init_grid

def test_common_boundary_conditions():
    grid = init_grid()
    Q0 = np.asarray([1,1,1,1])
    Q = np.zeros(Q0.shape + grid.cell_volumes.shape)

    boundary_conditions.set_common_boundary_conditions(Q, Q0, grid)

    assert np.all(Q[:, 1:-1,1:-1] == 0)
    assert np.all(Q[:, 0,:] == Q0[:, None])
    assert np.all(Q[:, :,0] == Q[:, :,1])
    assert np.all(Q[:, :,-1] == Q[:, :,-2])

def test_euler_boundary_conditions():
    grid = init_grid((3,2), (
        [0, 1, 2, 0, 1, 2],
        [0, 0, 0, 1, 1, 1]
    ))
    airfoil_start = grid.airfoil_start
    airfoil_end = grid.airfoil_end
    Q0 = np.asarray([1,1,1,1])
    Q = np.zeros(Q0.shape + grid.cell_volumes.shape)

    boundary_conditions.set_euler_boundary_conditions(Q, Q0, grid)

    assert np.all(Q[:, airfoil_start:airfoil_end+1,0] + Q[:, airfoil_start:airfoil_end+1,1] == 2 * Q[:, airfoil_start:airfoil_end+1,1])

def test_nav_stokes_boundary_conditions():
    grid = init_grid((3,2), (
        [0, 1, 2, 0, 1, 2],
        [0, 0, 0, 1, 1, 1]
    ))
    airfoil_start = grid.airfoil_start
    airfoil_end = grid.airfoil_end
    Q0 = np.asarray([1,1,1,1])
    Q = np.zeros(Q0.shape + grid.cell_volumes.shape)

    boundary_conditions.set_nav_stokes_boundary_conditions(Q, Q0, grid)

    assert np.all(Q[1:3, airfoil_start:airfoil_end+1,0] + Q[1:3, airfoil_start:airfoil_end+1,1] == 0)
    assert np.all(Q[0, airfoil_start:airfoil_end+1,0] + Q[0, airfoil_start:airfoil_end+1,1] == 2 * Q[0, airfoil_start:airfoil_end+1,1])
    assert np.all(Q[3, airfoil_start:airfoil_end+1,0] + Q[3, airfoil_start:airfoil_end+1,1] == 2 * Q[3, airfoil_start:airfoil_end+1,1])