import pytest
import numpy as np
from grid import Grid2D
from fluxes import viscous
import physics
from math import sqrt
from tests.utils import init_grid

def test_calculate_Ev_xi():
    grid = init_grid()
    
    Q0 = np.asarray([1,1,0,1])
    Q = np.zeros(Q0.shape + grid.cell_volumes.shape)

    Q[:, :,:] = Q0[:,None,None]

    Ev_xi = viscous.calculate_Ev_xi(Q, grid)

    assert Ev_xi == pytest.approx(np.zeros_like(Ev_xi))

def test_calculate_Fv_eta():
    grid = init_grid()
    
    Q0 = np.asarray([1,1,0,1])
    Q = np.zeros(Q0.shape + grid.cell_volumes.shape)

    Q[:, :,:] = Q0[:,None,None]

    Fv_eta = viscous.calculate_Fv_eta(Q, grid)

    assert Fv_eta == pytest.approx(np.zeros_like(Fv_eta))

def test_get_R():
    grid = init_grid()
    s_xi,s_eta = grid.cell_surface_areas
    vol = grid.cell_volumes
    Q = [1,1,1,1]

    mu = physics.get_mu(Q)
    k = physics.get_k(mu)

    R = viscous.get_R(s_xi[:, 1,1], s_xi[:, 1,1], Q, vol[1,1])

    R_actual = np.asarray([
        [0, 0, 0, 0],
        [0, mu * sqrt(8) / 3, 0, 0],
        [0, 0, mu / (3 * sqrt(2)), 0],
        [0, mu * sqrt(8) / 3, mu / (3 * sqrt(2)), k / sqrt(2)]
    ])

    assert R == pytest.approx(R_actual)