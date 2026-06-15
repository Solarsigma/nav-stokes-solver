from readgrid import readgrid
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from pprint import pprint
from math import sqrt
from os.path import join as ospathjoin


## CONSTANTS
ORDER = 1
CFL_MAX = 1
LIMITER = 'none'
# LIMITER = 'minmod'
# LIMITER = 'superbee'
ENV = 'dev'
GRID_FILENAME = 'g65x49u.dat' if ENV == 'prod' else 'g33x25u.dat'
FILE_NAME_SUFFIX = f'{ENV}_euler_inviscid_order_{ORDER}_limiter_{LIMITER}'

if ORDER == 1:
    MUSCL_EPS = 0
    MUSCL_K = 1
elif ORDER == 2:
    MUSCL_EPS = 1
    MUSCL_K = -1
elif ORDER == 3:
    MUSCL_EPS = 1
    MUSCL_K = 1/2
else:
    MUSCL_EPS = 0
    MUSCL_K = 1

max_iter = 100
convergence_tol = 1e-16



## COMPUTATIONAL DOMAIN

(nx,ny), (x_inp, y_inp) = readgrid(filepath=GRID_FILENAME)

x = np.zeros((nx+2, ny+2))
y = np.zeros((nx+2, ny+2))

airfoil_start = None
airfoil_end = None

for i in range(1, nx+1):
    for j in range(1, ny+1):
        x[i,j] = x_inp[i-1 + (j-1)*nx]
        y[i,j] = y_inp[i-1 + (j-1)*nx]

airfoil_start = np.argwhere(x[1:,1] == 0)[0,0] + 1
airfoil_end = np.argwhere(x[1:,1] == 1)[0,0] + 1

for i in range(1, nx+1):
    y[i,0] = 2*y[i,1] - y[i,2]
    x[i,0] = x[i,1]

    y[i,ny+1] = 2*y[i,ny] - y[i,ny-1]
    x[i,ny+1] = x[i,ny]

for j in range(0, ny+2):
    y[0,j] = y[1,j]
    x[0,j] = 2*x[1,j] - x[2,j]

    y[nx+1,j] = y[nx,j]
    x[nx+1,j] = 2*x[nx,j] - x[nx-1,j]


vol = np.zeros((nx+1, ny+1))

for i in range(0, nx+1):
    for j in range(0, ny+1):
        vol[i,j] = 0.5 * ((x[i+1,j+1] - x[i,j]) * (y[i,j+1] - y[i+1,j]) \
                            - (x[i,j+1] - x[i+1,j]) * (y[i+1,j+1] - y[i,j]))


s_xi = np.zeros((3, nx+1, ny))
for i in range(1, nx+1):
    for j in range(1, ny):
        s_xi[0,i,j] = y[i,j+1] - y[i,j]
        s_xi[1,i,j] = x[i,j] - x[i,j+1]
        s_xi[2,i,j] = (s_xi[0,i,j]**2 + s_xi[1,i,j]**2)**0.5

# Setting None so that program fails early if these are used in computations.
# TODO: Clean up
for i in range(nx+1):
    s_xi[0,i,0] = None
    s_xi[1,i,0] = None
    s_xi[2,i,0] = None
for j in range(1, ny):
    s_xi[0,0,j] = None
    s_xi[1,0,j] = None
    s_xi[2,0,j] = None


s_eta = np.zeros((3, nx, ny+1))
for i in range(1, nx):
    for j in range(1, ny+1):
        s_eta[0,i,j] = y[i,j] - y[i+1,j]
        s_eta[1,i,j] = x[i+1,j] - x[i,j]
        s_eta[2,i,j] = (s_eta[0,i,j]**2 + s_eta[1,i,j]**2)**0.5

# Setting None so that program fails early if these are used in computations.
# TODO: Clean up
for j in range(0, ny+1):
    s_eta[0,0,j] = None
    s_eta[1,0,j] = None
    s_eta[2,0,j] = None
for i in range(1, nx):
    s_eta[0,i,0] = None
    s_eta[1,i,0] = None
    s_eta[2,i,0] = None


xc = np.empty((nx+1, ny+1))
yc = np.empty((nx+1, ny+1))

for i in range(nx+1):
    for j in range(ny+1):
        xc[i,j] = 0.25*(x[i,j] + x[i+1,j]
                   + x[i,j+1] + x[i+1,j+1])
        yc[i,j] = 0.25*(y[i,j] + y[i+1,j]
                   + y[i,j+1] + y[i+1,j+1])


## PROPERTIES OF FLUID/SYSTEM
R = 287 # J/(kg-K)
Cp = 1005 # J/(kg-K)
gamma = 1.4
M = 2
c = 347.2 # m/s
u_ref = 694.4 # m/s
p_ref = 101325 # Pa
T_ref = 300 # K


## SETTING UP Q, E_ksi, F_eta VECTORS
def get_phi():
    if LIMITER == 'none':
        return lambda r : 1
    elif LIMITER == 'minmod':
        return lambda r : np.fmin(np.fmax(r, 0), 1)
    elif LIMITER == 'superbee':
        return lambda r : np.fmax(0, np.fmax(np.fmin(2*r, 1), np.fmin(r, 2)))


phi = get_phi()


def get_MUSCL_Q_LR(Q_n2, Q_n1, Q_0, Q_1):
    
    r_L = (Q_0 - Q_n1) / (Q_n1 - Q_n2)
    r_R = (Q_0 - Q_n1) / (Q_1 - Q_0)

    # for avoiding issues related to division by 0 or 0/0
    # note - the value assigned here doesn't matter as in such cases it will fall-back to (Q_n1, Q_0) anyways
    r_L[np.isnan(r_L)] = 1
    r_L[np.isinf(r_L)] = 1
    r_R[np.isnan(r_R)] = 1
    r_R[np.isinf(r_R)] = 1
    
    Q_L = Q_n1 + 0.25 * MUSCL_EPS * (Q_n1 - Q_n2) * ((1 - MUSCL_K) * phi(r_L) + (1 + MUSCL_K) * r_L * phi(1/r_L))
    Q_R = Q_0 - 0.25 * MUSCL_EPS * (Q_1 - Q_0) * ((1 - MUSCL_K) * phi(r_R) + (1 + MUSCL_K) * r_R * phi(1/r_R))
    return (Q_L, Q_R)

def entropy_corrected_roe_vel(v, max_eigen):
    eps = 0.2*max_eigen
    return abs(v) if abs(v) > eps else (v**2 + eps**2) / (2*eps)

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

    rho_L_rt = sqrt(rho_L)
    rho_R_rt = sqrt(rho_R)

    nx = s[0] / s[2]
    ny = s[1] / s[2]

    u_f = (rho_L_rt * u_L + rho_R_rt * u_R) / (rho_L_rt + rho_R_rt)
    v_f = (rho_L_rt * v_L + rho_R_rt * v_R) / (rho_L_rt + rho_R_rt)
    ht_f = (rho_L_rt * ht_L + rho_R_rt * ht_R) / (rho_L_rt + rho_R_rt)
    ek_f = 0.5*(u_f**2 + v_f**2)
    c_f = sqrt((gamma - 1) * (ht_f - ek_f))
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

def get_Q(p, u, v, T):
    rho = p / (R * T)
    return np.asarray([
        rho,
        rho * u,
        rho * v,
        (p / (gamma - 1)) + 0.5 * rho * (u**2 + v**2)
    ])

Q0 = get_Q(p_ref, u_ref, 0, T_ref)

Q_ref = np.asarray([Q0[0], Q0[1], Q0[1], Q0[3]])

Q = [np.zeros((4,) + vol.shape)]

## INITIAL CONDITION
for i in range(1, nx):
    for j in range(1, ny):
        Q[0][:,i,j] = Q0


## BOUNDARY CONDITIONS
def set_boundary_conditions(Q):
    ## INLET - free-stream
    for j in range(0, ny+1):
        Q[:,0,j] = Q0
    
    ## TOP WALL - inviscid adiabatic slip
    for i in range(1, nx):

        # rho and et are const
        Q[0, i,ny] = Q[0, i,ny-1]
        Q[3, i,ny] = Q[3, i,ny-1]

        # u and v - slip conditions
        # Q[1, i,ny] =  Q[1, i,ny-1]
        # Q[2, i,ny] = - Q[2, i,ny-1]
        n_x = s_eta[0, i,ny]/s_eta[2, i,ny]
        n_y = s_eta[1, i,ny]/s_eta[2, i,ny]

        Q[1, i,ny] = (n_y**2 - n_x**2) * Q[1, i,ny-1] - 2*n_x*ny * Q[2, i,ny-1]
        Q[2, i,ny] = -2*n_x*n_y * Q[1, i,ny-1] - (n_y**2 - n_x**2) * Q[2, i,ny-1]
    
    ## BOTTOM WALL - adiabatic inviscid slip
    for i in range(1, airfoil_start):

        # rho and et are const
        Q[0, i,0] = Q[0, i,1]
        Q[3, i,0] = Q[3, i,1]

        # u and v - slip conditions
        n_x = s_eta[0, i,1]/s_eta[2, i,1]
        n_y = s_eta[1, i,1]/s_eta[2, i,1]

        Q[1, i,0] = (n_y**2 - n_x**2) * Q[1, i,1] - 2*n_x*n_y * Q[2, i,1]
        Q[2, i,0] = -2*n_x*n_y * Q[1, i,1] - (n_y**2 - n_x**2) * Q[2, i,1]
    
    ## BOTTOM WALL - adiabatic inviscid slip
    for i in range(airfoil_end+1, nx):

        # rho and et are const
        Q[0, i,0] = Q[0, i,1]
        Q[3, i,0] = Q[3, i,1]

        # u and v - slip conditions
        n_x = s_eta[0, i,1]/s_eta[2, i,1]
        n_y = s_eta[1, i,1]/s_eta[2, i,1]

        Q[1, i,0] = (n_y**2 - n_x**2) * Q[1, i,1] - 2*n_x*n_y * Q[2, i,1]
        Q[2, i,0] = -2*n_x*n_y * Q[1, i,1] - (n_y**2 - n_x**2) * Q[2, i,1]
    
    ## AIRFOIL_EDGE - adiabatic inviscid slip for Euler
    for i in range(airfoil_start, airfoil_end+1):

        # rho and et are const
        Q[0, i,0] = Q[0, i,1]
        Q[3, i,0] = Q[3, i,1]

        # u and v - slip conditions
        n_x = s_eta[0, i,1]/s_eta[2, i,1]
        n_y = s_eta[1, i,1]/s_eta[2, i,1]

        Q[1, i,0] = (n_y**2 - n_x**2) * Q[1, i,1] - 2*n_x*n_y * Q[2, i,1]
        Q[2, i,0] = -2*n_x*n_y * Q[1, i,1] - (n_y**2 - n_x**2) * Q[2, i,1]
    
    ## OUTLET - free-stream
    for j in range(0, ny+1):
        Q[:, nx,j] = Q[:, nx-1,j]


set_boundary_conditions(Q[0])

# ## OUTLET - free-stream
# for j in range(0, ny+1):
#     Q[0][:, nx,j] = Q[0][:, nx-1,j]

## TIME MARCHING

l2_err_norms = np.ones((max_iter, 4))
linf_err_norms = np.ones((max_iter, 4))

curr_iter = 1
while (np.max(l2_err_norms[curr_iter - 1]) > convergence_tol and curr_iter < max_iter):

    E_xi_curr = np.zeros((4,) + s_xi.shape[1:])
    for j in range(1, ny):
        Q_L, Q_R = get_MUSCL_Q_LR(Q[-1][:, 0,j], Q[-1][:, 0,j], Q[-1][:, 1,j], Q[-1][:, 2,j])
        E_xi_curr[:, 1,j] = 0.5*(get_E_xi(Q_L, s_xi[:, 1,j]) + get_E_xi(Q_R, s_xi[:, 1,j]))

        E_xi_curr[:, 1,j] -= 0.5 * (get_roe_jacobian(Q_L, Q_R, s_xi[:, 1,j]) @ (Q_R - Q_L))

        Q_L, Q_R = get_MUSCL_Q_LR(Q[-1][:, nx-2,j], Q[-1][:, nx-1,j], Q[-1][:, nx,j], Q[-1][:, nx,j])
        E_xi_curr[:, nx,j] = 0.5*(get_E_xi(Q_L, s_xi[:, nx,j]) + get_E_xi(Q_R, s_xi[:, nx,j]))

        E_xi_curr[:, nx,j] -= 0.5 * (get_roe_jacobian(Q_L, Q_R, s_xi[:, nx,j]) @ (Q_R - Q_L))
    for i in range(2, nx):
        for j in range(1, ny):
            Q_L, Q_R = get_MUSCL_Q_LR(Q[-1][:, i-2,j], Q[-1][:, i-1,j], Q[-1][:, i,j], Q[-1][:, i+1,j])
            E_xi_curr[:, i,j] = 0.5*(get_E_xi(Q_L, s_xi[:, i,j]) + get_E_xi(Q_R, s_xi[:, i,j]))

            E_xi_curr[:, i,j] -= 0.5 * (get_roe_jacobian(Q_L, Q_R, s_xi[:, i,j]) @ (Q_R - Q_L))
    
    # Setting None so that program fails early if these are used in computations.
    # TODO: Clean up
    for i in range(nx+1):
        E_xi_curr[0,i,0] = None
        E_xi_curr[1,i,0] = None
        E_xi_curr[2,i,0] = None
    for j in range(1, ny):
        E_xi_curr[0,0,j] = None
        E_xi_curr[1,0,j] = None
        E_xi_curr[2,0,j] = None


    F_eta_curr = np.zeros((4,) + s_eta.shape[1:])
    for i in range(1, nx):
        Q_L, Q_R = get_MUSCL_Q_LR(Q[-1][:, i,0], Q[-1][:, i,0], Q[-1][:, i,1], Q[-1][:, i,2])
        F_eta_curr[:, i,1] = 0.5*(get_F_eta(Q_L, s_eta[:, i,1]) + get_F_eta(Q_R, s_eta[:, i,1]))

        F_eta_curr[:, i,1] -= 0.5 * (get_roe_jacobian(Q_L, Q_R, s_eta[:, i,1]) @ (Q_R - Q_L))

        Q_L, Q_R = get_MUSCL_Q_LR(Q[-1][:, i,ny-2], Q[-1][:, i,ny-1], Q[-1][:, i,ny], Q[-1][:, i,ny])
        F_eta_curr[:, i,ny] = 0.5*(get_F_eta(Q_L, s_eta[:, i,ny]) + get_F_eta(Q_R, s_eta[:, i,ny]))

        F_eta_curr[:, i,ny] -= 0.5 * (get_roe_jacobian(Q_L, Q_R, s_eta[:, i,ny]) @ (Q_R - Q_L))
    for i in range(1, nx):
        for j in range(2, ny):
            Q_L, Q_R = get_MUSCL_Q_LR(Q[-1][:, i,j-2], Q[-1][:, i,j-1], Q[-1][:, i,j], Q[-1][:, i,j+1])
            F_eta_curr[:, i,j] = 0.5*(get_F_eta(Q_L, s_eta[:, i,j]) + get_F_eta(Q_R, s_eta[:, i,j]))

            F_eta_curr[:, i,j] -= 0.5 * (get_roe_jacobian(Q_L, Q_R, s_eta[:, i,j]) @ (Q_R - Q_L))
    
    # Setting None so that program fails early if these are used in computations.
    # TODO: Clean up
    for i in range(1, nx):
        F_eta_curr[0,i,0] = None
        F_eta_curr[1,i,0] = None
        F_eta_curr[2,i,0] = None
    for j in range(0, ny+1):
        F_eta_curr[0,0,j] = None
        F_eta_curr[1,0,j] = None
        F_eta_curr[2,0,j] = None
    

    Q_diff = np.zeros(Q[-1].shape)
    for i in range(1, nx):
        for j in range(1, ny):
            p_cell = (gamma - 1) * (Q[-1][3, i,j] - 0.5*(Q[-1][1, i,j]**2 + Q[-1][2, i,j]**2) / Q[-1][0, i,j])
            c_cell = sqrt(gamma * p_cell / Q[-1][0, i,j])

            # U_xi_ij = (Q[-1][1, i,j]*s_xi[0, i,j] + Q[-1][2, i,j]*s_xi[1, i,j]) \
            #             / (Q[-1][0, i,j] * (s_xi[2, i,j]))
            # U_xi_i1j = (Q[-1][1, i,j]*s_xi[0, i+1,j] + Q[-1][2, i,j]*s_xi[1, i+1,j]) \
            #             / (Q[-1][0, i,j] * (s_xi[2, i+1,j]))

            # V_eta_ij = (Q[-1][1, i,j]*s_eta[0, i,j] + Q[-1][2, i,j]*s_eta[1, i,j]) \
            #             / (Q[-1][0, i,j] * (s_eta[2, i,j]))
            # V_eta_ij1 = (Q[-1][1, i,j]*s_eta[0, i,j+1] + Q[-1][2, i,j]*s_eta[1, i,j+1]) \
            #             / (Q[-1][0, i,j] * (s_eta[2, i,j+1]))

            # denom = (abs(U_xi_ij) + c_cell) * s_xi[2, i,j] \
            #         + (abs(U_xi_i1j) + c_cell) * s_xi[2, i+1,j] \
            #         + (abs(V_eta_ij) + c_cell) * s_eta[2, i,j] \
            #         + (abs(V_eta_ij1) + c_cell) * s_eta[2, i,j+1] \
            
            # delta_tau_cap = CFL_MAX / denom

            n_xi_cell = 0.5*(s_xi[:2, i,j]/s_xi[2, i,j] + s_xi[:2, i+1,j]/s_xi[2, i+1,j])
            s_xi_cell = 0.5 * (s_xi[2, i,j] + s_xi[2, i+1,j])

            U_xi_cell = Q[-1][1, i,j]*n_xi_cell[0] + Q[-1][2, i,j]*n_xi_cell[1]

            n_eta_cell = 0.5*(s_eta[:2, i,j]/s_eta[2, i,j] + s_eta[:2, i,j+1]/s_eta[2, i,j+1])
            s_eta_cell = 0.5 * (s_eta[2, i,j] + s_eta[2, i,j+1])

            V_eta_cell = Q[-1][1, i,j]*n_eta_cell[0] + Q[-1][2, i,j]*n_eta_cell[1]

            rho_xi = (abs(U_xi_cell) + c_cell) * s_xi_cell
            rho_eta = (abs(V_eta_cell) + c_cell) * s_eta_cell

            delta_tau_cap = CFL_MAX / (rho_xi + rho_eta)
            # s_xi_cell[2] = (s_xi_cell[0]**2 + s_xi_cell[1]**2)**0.5
            # s_eta_cell = 0.5 * (s_eta[:, i,j] + s_eta[:, i,j+1])
            # s_eta_cell[2] = (s_eta_cell[0]**2 + s_eta_cell[1]**2)**0.5

            # U_xi_cell = (Q[-1][1, i,j]*s_xi_cell[0] + Q[-1][2, i,j]*s_xi_cell[1]) \
            #             / (Q[-1][0, i,j] * (s_xi_cell[2]))

            # V_eta_cell = (Q[-1][1, i,j]*s_eta_cell[0] + Q[-1][2, i,j]*s_eta_cell[1]) \
            #             / (Q[-1][0, i,j] * (s_eta_cell[2]))

            # rho_xi_cell = (abs(U_xi_cell) + c_cell) * s_xi_cell[2]
            # rho_eta_cell = (abs(V_eta_cell) + c_cell) * s_eta_cell[2]
            
            # delta_tau_cap = CFL_MAX / (rho_xi_cell + rho_eta_cell)

            Q_diff[:, i,j] = - (delta_tau_cap) * ( \
                (E_xi_curr[:, i+1,j] * s_xi[2, i+1,j] - E_xi_curr[:, i,j] * s_xi[2, i,j])
                + (F_eta_curr[:, i,j+1] * s_eta[2, i,j+1] - F_eta_curr[:, i,j] * s_eta[2, i,j])
                )
    
    Q.append(Q[-1] + Q_diff)
    
    set_boundary_conditions(Q[-1])

    ## OUTLET - first order interp
    for j in range(1, ny):
        Q[-1][:, nx,j] = Q[-2][:, nx-1,j]
    

    l2_err_norms[curr_iter] = np.linalg.norm(Q_diff, axis=(1,2)) / Q_ref
    linf_err_norms[curr_iter] = np.max(Q_diff, axis=(1,2)) / Q_ref

    print(f"Iter {curr_iter}")
    print(f"Actual Norm Error: {np.linalg.norm(Q_diff / np.expand_dims(Q_ref, (1,2)))}")
    print(f"Q Norm Error: {np.linalg.norm(Q_diff / np.expand_dims(Q_ref, (1,2)))}")
    print(f"Error norm RHO: {l2_err_norms[curr_iter][0]}")
    print(f"Error norm RHO U: {l2_err_norms[curr_iter][1]}")
    print(f"Error norm RHO V: {l2_err_norms[curr_iter][2]}")
    print(f"Error norm RHO ET: {l2_err_norms[curr_iter][3]}")
    print("\n")

    curr_iter += 1

    if curr_iter == max_iter:
        print("Convergence not achieved!")
        print("Last error norms: ")
        print(f"L2: {l2_err_norms[-1]}")
        print(f"LINF: {linf_err_norms[-1]}")
        to_continue = input("Continue?[Y/n]: ")
        if to_continue == 'y':
            new_max_iter = input(f"New max-iter (int) (Curr val: {max_iter}): ")
            max_iter = int(new_max_iter)
            print(f"new max iter = {max_iter}")
            l2_err_norms.resize((max_iter,) + l2_err_norms.shape[1:])
            linf_err_norms.resize((max_iter,) + linf_err_norms.shape[1:])


np.save(ospathjoin("data", f"Q_{FILE_NAME_SUFFIX}"), Q)
np.save(ospathjoin("data", f"l2_err_norms_{FILE_NAME_SUFFIX}"), l2_err_norms[1:curr_iter])
np.save(ospathjoin("data", f"linf_err_norms_{FILE_NAME_SUFFIX}"), linf_err_norms[1:curr_iter])


titles = ["RHO", "RHO U", "RHO V", "RHO ET"]
for i in range(4):
    plt.figure()
    plt.plot(list(range(1, curr_iter)), l2_err_norms[1:,i])
    plt.title(titles[i])
plt.show()




## DEBUGGING

# mesh = np.asarray([[[x[i,j], y[i,j]] for j in range(ny+2)] for i in range(nx+2)])

# obs_time = 2

# for obs_time in range(2,2):
#     # Qv = get_Qv(Q[obs_time])
#     Q_curr = Q[obs_time] / vol

#     plt.figure()

#     plt.pcolormesh(xc, yc, Q_curr[1,:,:])
#     plt.colorbar()
#     plt.title("rho u")

#     segs1 = mesh
#     segs2 = segs1.transpose(1,0,2)
#     plt.gca().add_collection(LineCollection(segs1, color='black', linewidths=0.2))
#     plt.gca().add_collection(LineCollection(segs2, color='black', linewidths=0.2))
#     plt.gca().autoscale()

#     plt.figure()

#     plt.pcolormesh(xc, yc, Q_curr[2,:,:])
#     plt.colorbar()
#     plt.title("rho v")

#     segs1 = mesh
#     segs2 = segs1.transpose(1,0,2)
#     plt.gca().add_collection(LineCollection(segs1, color='black', linewidths=0.2))
#     plt.gca().add_collection(LineCollection(segs2, color='black', linewidths=0.2))
#     plt.gca().autoscale()

#     plt.figure()

#     plt.pcolormesh(xc, yc, Q_curr[0,:,:])
#     plt.colorbar()
#     plt.title("rho")

#     segs1 = mesh
#     segs2 = segs1.transpose(1,0,2)
#     plt.gca().add_collection(LineCollection(segs1, color='black', linewidths=0.2))
#     plt.gca().add_collection(LineCollection(segs2, color='black', linewidths=0.2))
#     plt.gca().autoscale()

#     plt.figure()

#     plt.pcolormesh(xc, yc, Q_curr[3,:,:])
#     plt.colorbar()
#     plt.title("rho et")

#     segs1 = mesh
#     segs2 = segs1.transpose(1,0,2)
#     plt.gca().add_collection(LineCollection(segs1, color='black', linewidths=0.2))
#     plt.gca().add_collection(LineCollection(segs2, color='black', linewidths=0.2))
#     plt.gca().autoscale()

#     plt.show()

# TODO: Refactor/restructure code
# TODO: AUSM
# TODO: Navier stokes solver