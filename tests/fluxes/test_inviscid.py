import pytest
import numpy as np
from grid import Grid2D
from fluxes import inviscid
from tests.utils import init_grid

def test_get_flux_dissipation():
    grid = init_grid()
    nx,_ = grid.n
    Q0 = np.asarray([1,1,0,1])
    Q = np.zeros(Q0.shape + grid.cell_volumes.shape)

    Q[:, :,:] = Q0[:,None,None]

    flux_dissipation = inviscid.get_flux_dissipation(Q[:, 1:nx,1], Q[:, 1:nx,1], grid.cell_surface_areas[0][:, 1:nx,1])

    assert flux_dissipation == pytest.approx(np.zeros_like(flux_dissipation))

def test_calculate_E_xi():
    grid = init_grid()
    nx,ny = grid.n
    Q0 = np.asarray([1,1,0,1])
    Q = np.zeros(Q0.shape + grid.cell_volumes.shape)

    Q[:, :,:] = Q0[:,None,None]

    E_xi = inviscid.calculate_E_xi(Q, grid)

    E_xi_actual = np.zeros_like(E_xi)
    E_xi_actual[:, 1:nx+1,1:ny+1] = np.asarray([1,1.2,0,1.2])[:,None,None]

    assert E_xi == pytest.approx(E_xi_actual)

def test_calculate_F_eta():
    grid = init_grid()
    nx,ny = grid.n
    Q0 = np.asarray([1,0,1,1])
    Q = np.zeros(Q0.shape + grid.cell_volumes.shape)

    Q[:, :,:] = Q0[:,None,None]

    F_eta = inviscid.calculate_F_eta(Q, grid)

    F_eta_actual = np.zeros_like(F_eta)
    F_eta_actual[:, 1:nx+1,1:ny+1] = np.asarray([1,0,1.2,1.2])[:,None,None]

    assert F_eta == pytest.approx(F_eta_actual)