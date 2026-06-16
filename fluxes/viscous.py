import numpy as np
import state
from physics import *

def get_R(S_m, S_n, Q, vol_cell):
    mu = get_mu(Q)
    k = get_k(mu)
    r22 = mu / (vol_cell * S_m[2]) * ((S_m[0] * S_n[0] * 4/3) + (S_m[1] * S_n[1]))
    r23 = mu / (vol_cell * S_m[2]) * (-(S_m[0] * S_n[1] * 2/3) + (S_m[1] * S_n[0]))
    r32 = mu / (vol_cell * S_m[2]) * (-(S_m[1] * S_n[0] * 2/3) + (S_m[0] * S_n[1]))
    r33 = mu / (vol_cell * S_m[2]) * ((S_m[1] * S_n[1] * 4/3) + (S_m[0] * S_n[0]))
    r42 = (Q[1] / Q[0]) * r22 + (Q[2] / Q[0]) * r32
    r43 = (Q[1] / Q[0]) * r23 + (Q[2] / Q[0]) * r33
    r44 = k / (vol_cell * S_m[2]) * (S_m[0] * S_n[0] + S_m[1] * S_n[1])
    zeros = np.zeros_like(r22)
    return np.asarray([
        [zeros,     zeros,      zeros,      zeros],
        [zeros,     r22,    r23,    zeros],
        [zeros,     r32,    r33,    zeros],
        [zeros,     r42,    r43,    r44]
    ])

def calculate_Ev_xi(Q, grid : Grid2D):
    s_xi,s_eta = grid.cell_surface_areas
    nx,ny = grid.n
    vol = grid.cell_volumes
    Qv = state.get_Qv(Q)

    Ev_xi_curr = np.zeros((4,) + s_xi.shape[1:])

    delta_Qv_xi = Qv[:, nx,1:ny] - Qv[:, nx-1,1:ny]
    delta_Qv_eta = 0.25 * (Qv[:, nx,2:ny+1] - Qv[:, nx,:ny-1]
                            + Qv[:, nx-1,2:ny+1] - Qv[:, nx-1,:ny-1])
    Ev_xi_curr[:, nx,1:ny] = (
        np.matvec(get_R(s_xi[:, nx,1:ny], s_xi[:, nx,1:ny], 0.5 * (Q[:, nx-1,1:ny] + Q[:, nx,1:ny]), 0.5 * (vol[nx-1,1:ny] + vol[nx,1:ny])),
                    delta_Qv_xi,
                    axes=[(0,1),(0,),(0,)])
        + np.matvec(get_R(s_xi[:, nx,1:ny], 0.5*(s_eta[:, nx-1,1:ny]+s_eta[:, nx-1,2:ny+1]), 0.5 * (Q[:, nx-1,1:ny] + Q[:, nx,1:ny]), 0.5 * (vol[nx-1,1:ny] + vol[nx,1:ny])),
                    delta_Qv_eta,
                    axes=[(0,1),(0,),(0,)])
    )

    delta_Qv_xi = Qv[:, 1,1:ny] - Qv[:, 0,1:ny]
    delta_Qv_eta = 0.25 * (Qv[:, 1,2:ny+1] - Qv[:, 1,:ny-1]
                            + Qv[:, 0,2:ny+1] - Qv[:, 0,:ny-1])

    Ev_xi_curr[:, 1,1:ny] = (
        np.matvec(get_R(s_xi[:, 1,1:ny], s_xi[:, 1,1:ny], 0.5 * (Q[:, 0,1:ny] + Q[:, 1,1:ny]), 0.5 * (vol[0,1:ny] + vol[1,1:ny])),
                    delta_Qv_xi,
                    axes=[(0,1),(0,),(0,)])
        + np.matvec(get_R(s_xi[:, 1,1:ny], 0.5*(s_eta[:, 1,1:ny]+s_eta[:, 1,2:ny+1]), 0.5 * (Q[:, 0,1:ny] + Q[:, 1,1:ny]), 0.5 * (vol[0,1:ny] + vol[1,1:ny])),
                    delta_Qv_eta,
                    axes=[(0,1),(0,),(0,)])
    )

    delta_Qv_xi = Qv[:, 2:nx,1:ny] - Qv[:, 1:nx-1,1:ny]
    delta_Qv_eta = 0.25 * (Qv[:, 2:nx,2:ny+1] - Qv[:, 2:nx,:ny-1]
                            + Qv[:, 1:nx-1,2:ny+1] - Qv[:, 1:nx-1,:ny-1])
    s_eta_curr = 0.25*(s_eta[:, 2:nx,1:ny] + s_eta[:, 2:nx,2:ny+1] + s_eta[:, 1:nx-1,1:ny] + s_eta[:, 1:nx-1,2:ny+1])

    Ev_xi_curr[:, 2:nx,1:ny] = (
        np.matvec(get_R(s_xi[:, 2:nx,1:ny], s_xi[:, 2:nx,1:ny], 0.5 * (Q[:, 1:nx-1,1:ny] + Q[:, 2:nx,1:ny]), 0.5 * (vol[1:nx-1,1:ny] + vol[2:nx,1:ny])),
                    delta_Qv_xi,
                    axes=[(0,1),(0,),(0,)])
        + np.matvec(get_R(s_xi[:, 2:nx,1:ny], s_eta_curr, 0.5 * (Q[:, 1:nx-1,1:ny] + Q[:, 2:nx,1:ny]), 0.5 * (vol[1:nx-1,1:ny] + vol[2:nx,1:ny])),
                    delta_Qv_eta,
                    axes=[(0,1),(0,),(0,)])
    )

    return Ev_xi_curr

def calculate_Fv_eta(Q, grid : Grid2D):
    s_xi,s_eta = grid.cell_surface_areas
    nx,ny = grid.n
    vol = grid.cell_volumes
    Qv = state.get_Qv(Q)

    Fv_eta_curr = np.zeros((4,) + s_eta.shape[1:])
    delta_Qv_eta = Qv[:, 1:nx,ny] - Qv[:, 1:nx,ny-1]
    delta_Qv_xi = 0.25 * (Qv[:, 2:nx+1,ny] - Qv[:, :nx-1,ny]
                            + Qv[:, 2:nx+1,ny-1] - Qv[:, :nx-1,ny-1])
    Fv_eta_curr[:, 1:nx,ny] = (
        np.matvec(get_R(s_eta[:, 1:nx,ny], 0.5*(s_xi[:, 1:nx,ny-1] + s_xi[:, 2:nx+1,ny-1]), 0.5 * (Q[:, 1:nx,ny-1] + Q[:, 1:nx,ny]), 0.5 * (vol[1:nx,ny-1] + vol[1:nx,ny])),
                    delta_Qv_xi,
                    axes=[(0,1),(0,),(0,)])
        + np.matvec(get_R(s_eta[:, 1:nx,ny], s_eta[:, 1:nx,ny], 0.5 * (Q[:, 1:nx,ny-1] + Q[:, 1:nx,ny]), 0.5 * (vol[1:nx,ny-1] + vol[1:nx,ny])),
                    delta_Qv_eta,
                    axes=[(0,1),(0,),(0,)])
    )

    delta_Qv_eta = Qv[:, 1:nx,1] - Qv[:, 1:nx,0]
    delta_Qv_xi = 0.25 * (Qv[:, 2:nx+1,1] - Qv[:, :nx-1,1]
                            + Qv[:, 2:nx+1,0] - Qv[:, :nx-1,0])
    Fv_eta_curr[:, 1:nx,1] = (
        np.matvec(get_R(s_eta[:, 1:nx,1], 0.5*(s_xi[:, 1:nx,1] + s_xi[:, 2:nx+1,1]), 0.5 * (Q[:, 1:nx,0] + Q[:, 1:nx,1]), 0.5 * (vol[1:nx,0] + vol[1:nx,1])),
                    delta_Qv_xi,
                    axes=[(0,1),(0,),(0,)])
        + np.matvec(get_R(s_eta[:, 1:nx,1], s_eta[:, 1:nx,1], 0.5 * (Q[:, 1:nx,0] + Q[:, 1:nx,1]), 0.5 * (vol[1:nx,0] + vol[1:nx,1])),
                    delta_Qv_eta,
                    axes=[(0,1),(0,),(0,)])
    )

    delta_Qv_eta = Qv[:, 1:nx,2:ny] - Qv[:, 1:nx,1:ny-1]
    delta_Qv_xi = 0.25 * (Qv[:, 2:nx+1,2:ny] - Qv[:, 0:nx-1,2:ny]
                            + Qv[:, 2:nx+1,1:ny-1] - Qv[:, :nx-1,1:ny-1])
    s_xi_curr = 0.25*(s_xi[:, 1:nx,2:ny] + s_xi[:, 2:nx+1,2:ny] + s_xi[:, 1:nx,1:ny-1] + s_xi[:, 2:nx+1,1:ny-1])
    Fv_eta_curr[:, 1:nx,2:ny] = (
        np.matvec(get_R(s_eta[:, 1:nx,2:ny], s_xi_curr, 0.5 * (Q[:, 1:nx,1:ny-1] + Q[:, 1:nx,2:ny]), 0.5 * (vol[1:nx,1:ny-1] + vol[1:nx,2:ny])),
                    delta_Qv_xi,
                    axes=[(0,1),(0,),(0,)])
        + np.matvec(get_R(s_eta[:, 1:nx,2:ny], s_eta[:, 1:nx,2:ny], 0.5 * (Q[:, 1:nx,1:ny-1] + Q[:, 1:nx,2:ny]), 0.5 * (vol[1:nx,1:ny-1] + vol[1:nx,2:ny])),
                    delta_Qv_eta,
                    axes=[(0,1),(0,),(0,)])
    )

    return Fv_eta_curr