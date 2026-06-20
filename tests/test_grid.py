from grid import Grid2D
import numpy as np
from tests.utils import init_grid

def test_grid_volumes():
    grid = init_grid()
    vol = grid.cell_volumes

    assert np.all(vol[:-1,:-1] > 0)

def test_grid_areas():
    grid = init_grid()
    s_xi,s_eta = grid.cell_surface_areas

    div_s = s_xi[:-1, 1:-1,1:] - s_xi[:-1, 2:,1:] + s_eta[:-1, 1:,1:-1] - s_eta[:-1, 1:,2:]

    assert np.all(div_s == 0)