import numpy as np
from fluxes.limiter import FluxLimiterFactory
import config
from grid import Grid2D
from constants.fluid_properties import gamma, R

if config.ORDER == 1:
    MUSCL_EPS = 0
    MUSCL_K = 1
elif config.ORDER == 2:
    MUSCL_EPS = 1
    MUSCL_K = -1
elif config.ORDER == 3:
    MUSCL_EPS = 1
    MUSCL_K = 1/2
else:
    MUSCL_EPS = 0
    MUSCL_K = 1

limiter = FluxLimiterFactory().get_limiter(config.LIMITER)

def get_MUSCL_Q_LR(Q_n2, Q_n1, Q_0, Q_1):
    
    r_L = (Q_0 - Q_n1) / (Q_n1 - Q_n2 + 1e-10)
    r_R = (Q_0 - Q_n1) / (Q_1 - Q_0 + 1e-10)
    
    Q_L = Q_n1 + 0.25 * MUSCL_EPS * (Q_n1 - Q_n2) * ((1 - MUSCL_K) * limiter.apply(r_L) + (1 + MUSCL_K) * r_L * limiter.apply(1/(r_L + 1e-10)))
    Q_R = Q_0 - 0.25 * MUSCL_EPS * (Q_1 - Q_0) * ((1 - MUSCL_K) * limiter.apply(r_R) + (1 + MUSCL_K) * r_R * limiter.apply(1/(r_R + 1e-10)))
    return Q_L, Q_R

def get_flux_dissipation(Q_L, Q_R, s):
    rho_L = Q_L[0]
    u_L = Q_L[1] / rho_L
    v_L = Q_L[2] / rho_L
    p_L = (gamma - 1) * (Q_L[3] - 0.5*rho_L*(u_L**2 + v_L**2))
    if np.any(rho_L < 0) or np.any(p_L < 0):
        print("NEGATIVE DENSITY OR PRESSURE ENCOUNTERED!")
        print(f"RHO: {rho_L[rho_L < 0]}")
        print(f"p: {p_L[p_L < 0]}")
        input("Continue?")

    ht_L = (Q_L[3] + p_L) / rho_L

    rho_R = Q_R[0]
    u_R = Q_R[1] / rho_R
    v_R = Q_R[2] / rho_R
    p_R = (gamma - 1) * (Q_R[3] - 0.5*rho_R*(u_R**2 + v_R**2))
    ht_R = (Q_R[3] + p_R) / rho_R

    rho_L_rt = np.sqrt(rho_L)
    rho_R_rt = np.sqrt(rho_R)

    nx = s[0] / s[2]
    ny = s[1] / s[2]

    u_f = (rho_L_rt * u_L + rho_R_rt * u_R) / (rho_L_rt + rho_R_rt)
    v_f = (rho_L_rt * v_L + rho_R_rt * v_R) / (rho_L_rt + rho_R_rt)
    ht_f = (rho_L_rt * ht_L + rho_R_rt * ht_R) / (rho_L_rt + rho_R_rt)
    ek_f = 0.5*(u_f**2 + v_f**2)
    c_f = np.sqrt((gamma - 1) * (ht_f - ek_f))
    U = u_f*nx + v_f*ny

    ones_arr = np.ones_like(u_f)


    R = np.asarray([
        [ones_arr,              ones_arr,               ones_arr,               ones_arr*0],
        [u_f - c_f*nx,          u_f,                    u_f + c_f*nx,           ones_arr*ny],
        [v_f - c_f*ny,          v_f,                    v_f + c_f*ny,           ones_arr*(-nx)],
        [ht_f - U*c_f,          ek_f,                   ht_f + U*c_f,           u_f*ny - v_f*nx]
    ])

    c_f_sq = c_f**2

    L = np.asarray([
        [((gamma - 1)*ek_f + U*c_f)/(2*c_f_sq),    ((1 - gamma)*u_f - c_f*nx)/(2*c_f_sq),      ((1 - gamma)*v_f - c_f*ny)/(2*c_f_sq),  (gamma - 1)/(2*c_f_sq) ],
        [1 - (gamma - 1)*ek_f/c_f_sq,                   (gamma - 1)*u_f/c_f_sq,                     (gamma - 1)*v_f/c_f_sq,                 (1 - gamma)/c_f_sq],
        [((gamma - 1)*ek_f - U*c_f)/(2*c_f_sq),    ((1 - gamma)*u_f + c_f*nx)/(2*c_f_sq),      ((1 - gamma)*v_f + c_f*ny)/(2*c_f_sq),  (gamma - 1)/(2*c_f_sq)],
        [v_f*nx - u_f*ny,                               ones_arr*ny,                            ones_arr*(-nx),                         ones_arr*0]
    ])

    Lambda = entropy_corrected_roe_vel(np.asarray([U - c_f, U, U + c_f, U]), np.abs(U) + c_f)

    delta_Q = Q_R - Q_L
    flux_dissipation = np.vecdot(L[0], delta_Q, axis=0)*Lambda[0]*R[:,0]
    flux_dissipation += np.vecdot(L[1], delta_Q, axis=0)*Lambda[1]*R[:,1]
    flux_dissipation += np.vecdot(L[2], delta_Q, axis=0)*Lambda[2]*R[:,2]
    flux_dissipation += np.vecdot(L[3], delta_Q, axis=0)*Lambda[3]*R[:,3]

    return 0.5*flux_dissipation

def entropy_corrected_roe_vel(v, max_eigen):
    eps = 0.2*max_eigen
    return np.where(np.abs(v) > eps, np.abs(v), (v**2 + eps**2) / (2*eps))

def get_roe_jacobian(Q_L, Q_R, s):
    rho_L = Q_L[0]
    u_L = Q_L[1] / rho_L
    v_L = Q_L[2] / rho_L
    p_L = (gamma - 1) * (Q_L[3] - 0.5*rho_L*(u_L**2 + v_L**2))
    if rho_L < 0 or p_L < 0:
        print("NEGATIVE DENSITY OR PRESSURE ENCOUNTERED!")
        print(f"RHO: {rho_L}")
        print(f"p: {p_L}")
        input("Continue?")

    ht_L = (Q_L[3] + p_L) / rho_L

    rho_R = Q_R[0]
    u_R = Q_R[1] / rho_R
    v_R = Q_R[2] / rho_R
    p_R = (gamma - 1) * (Q_R[3] - 0.5*rho_R*(u_R**2 + v_R**2))
    ht_R = (Q_R[3] + p_R) / rho_R

    rho_L_rt = np.sqrt(rho_L)
    rho_R_rt = np.sqrt(rho_R)

    nx = s[0] / s[2]
    ny = s[1] / s[2]

    u_f = (rho_L_rt * u_L + rho_R_rt * u_R) / (rho_L_rt + rho_R_rt)
    v_f = (rho_L_rt * v_L + rho_R_rt * v_R) / (rho_L_rt + rho_R_rt)
    ht_f = (rho_L_rt * ht_L + rho_R_rt * ht_R) / (rho_L_rt + rho_R_rt)
    ek_f = 0.5*(u_f**2 + v_f**2)
    c_f = np.sqrt((gamma - 1) * (ht_f - ek_f))
    U = u_f*nx + v_f*ny

    R = np.asarray([
        [1,                     1,          1,                  0],
        [u_f - c_f*nx,          u_f,        u_f + c_f*nx,       ny],
        [v_f - c_f*ny,          v_f,        v_f + c_f*ny,       -nx],
        [ht_f - U*c_f,          ek_f,       ht_f + U*c_f,       u_f*ny - v_f*nx]
    ])

    c_f_sq = c_f**2

    L = np.asarray([
        [((gamma - 1)*ek_f + U*c_f)/(2*c_f_sq),    ((1 - gamma)*u_f - c_f*nx)/(2*c_f_sq),      ((1 - gamma)*v_f - c_f*ny)/(2*c_f_sq),  (gamma - 1)/(2*c_f_sq) ],
        [1 - (gamma - 1)*ek_f/c_f_sq,                   (gamma - 1)*u_f/c_f_sq,                     (gamma - 1)*v_f/c_f_sq,                 (1 - gamma)/c_f_sq],
        [((gamma - 1)*ek_f - U*c_f)/(2*c_f_sq),    ((1 - gamma)*u_f + c_f*nx)/(2*c_f_sq),      ((1 - gamma)*v_f + c_f*ny)/(2*c_f_sq),  (gamma - 1)/(2*c_f_sq)],
        [v_f*nx - u_f*ny,                               ny,                                         -nx,                                    0]
    ])

    Lambda = np.asarray([
        [entropy_corrected_roe_vel(U - c_f, abs(U) + c_f),        0,          0,                  0],
        [0,                     entropy_corrected_roe_vel(U, abs(U) + c_f),   0,                  0],
        [0,                     0,          entropy_corrected_roe_vel(U + c_f, abs(U) + c_f),     0],
        [0,                     0,          0,                  entropy_corrected_roe_vel(U, abs(U) + c_f)]
    ])

    return R @ Lambda @ L

def get_E_xi(Q, s_xi):
    u = Q[1] / Q[0]
    v = Q[2] / Q[0]
    p = (gamma - 1) * (Q[3] - 0.5*Q[0]*(u**2 + v**2))

    U_xi = (u * s_xi[0] + v * s_xi[1]) / s_xi[2]

    return np.asarray([
        Q[0] * U_xi,
        Q[1] * U_xi + p * s_xi[0] / s_xi[2],
        Q[2] * U_xi + p * s_xi[1] / s_xi[2],
        (Q[3] + p) * U_xi
    ])

def get_F_eta(Q, s_eta):
    u = Q[1] / Q[0]
    v = Q[2] / Q[0]
    p = (gamma - 1) * (Q[3] - 0.5*Q[0]*(u**2 + v**2))

    V_eta = (u * s_eta[0] + v * s_eta[1]) / s_eta[2]

    return np.asarray([
        Q[0] * V_eta,
        Q[1] * V_eta + p * s_eta[0] / s_eta[2],
        Q[2] * V_eta + p * s_eta[1] / s_eta[2],
        (Q[3] + p) * V_eta
    ])


def calculate_E_xi(Q, grid : Grid2D):
    s_xi, _ = grid.cell_surface_areas
    nx,ny = grid.n
    E_xi_curr = np.zeros((4,) + s_xi.shape[1:])
    
    Q_L, Q_R = get_MUSCL_Q_LR(Q[:, 0,1:ny], Q[:, 0,1:ny], Q[:, 1,1:ny], Q[:, 2,1:ny])
    E_xi_curr[:, 1,1:ny] = 0.5*(get_E_xi(Q_L, s_xi[:, 1,1:ny]) + get_E_xi(Q_R, s_xi[:, 1,1:ny]))
    E_xi_curr[:, 1,1:ny] -= get_flux_dissipation(Q_L, Q_R, s_xi[:, 1,1:ny])

    Q_L, Q_R = get_MUSCL_Q_LR(Q[:, nx-2,1:ny], Q[:, nx-1,1:ny], Q[:, nx,1:ny], Q[:, nx,1:ny])
    E_xi_curr[:, nx,1:ny] = 0.5*(get_E_xi(Q_L, s_xi[:, nx,1:ny]) + get_E_xi(Q_R, s_xi[:, nx,1:ny]))
    E_xi_curr[:, nx,1:ny] -= get_flux_dissipation(Q_L, Q_R, s_xi[:, nx,1:ny])

    Q_L, Q_R = get_MUSCL_Q_LR(Q[:, :nx-2,1:ny], Q[:, 1:nx-1,1:ny], Q[:, 2:nx,1:ny], Q[:, 3:nx+1,1:ny])
    E_xi_curr[:, 2:nx,1:ny] = 0.5*(get_E_xi(Q_L, s_xi[:, 2:nx,1:ny]) + get_E_xi(Q_R, s_xi[:, 2:nx,1:ny]))
    E_xi_curr[:, 2:nx,1:ny] -= get_flux_dissipation(Q_L, Q_R, s_xi[:, 2:nx,1:ny])

    return E_xi_curr

def calculate_F_eta(Q, grid : Grid2D):
    _,s_eta = grid.cell_surface_areas
    nx,ny = grid.n
    F_eta_curr = np.zeros((4,) + s_eta.shape[1:])

    Q_L, Q_R = get_MUSCL_Q_LR(Q[:, 1:nx,0], Q[:, 1:nx,0], Q[:, 1:nx,1], Q[:, 1:nx,2])
    F_eta_curr[:, 1:nx,1] = 0.5*(get_F_eta(Q_L, s_eta[:, 1:nx,1]) + get_F_eta(Q_R, s_eta[:, 1:nx,1]))
    F_eta_curr[:, 1:nx,1] -= get_flux_dissipation(Q_L, Q_R, s_eta[:, 1:nx,1])

    Q_L, Q_R = get_MUSCL_Q_LR(Q[:, 1:nx,ny-2], Q[:, 1:nx,ny-1], Q[:, 1:nx,ny], Q[:, 1:nx,ny])
    F_eta_curr[:, 1:nx,ny] = 0.5*(get_F_eta(Q_L, s_eta[:, 1:nx,ny]) + get_F_eta(Q_R, s_eta[:, 1:nx,ny]))
    F_eta_curr[:, 1:nx,ny] -= get_flux_dissipation(Q_L, Q_R, s_eta[:, 1:nx,ny])

    Q_L, Q_R = get_MUSCL_Q_LR(Q[:, 1:nx,:ny-2], Q[:, 1:nx,1:ny-1], Q[:, 1:nx,2:ny], Q[:, 1:nx,3:ny+1])
    F_eta_curr[:, 1:nx,2:ny] = 0.5*(get_F_eta(Q_L, s_eta[:, 1:nx,2:ny]) + get_F_eta(Q_R, s_eta[:, 1:nx,2:ny]))
    F_eta_curr[:, 1:nx,2:ny] -= get_flux_dissipation(Q_L, Q_R, s_eta[:, 1:nx,2:ny])

    return F_eta_curr