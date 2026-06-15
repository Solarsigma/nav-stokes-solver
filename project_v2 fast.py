from readgrid import readgrid
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from pprint import pprint
from os.path import join as ospathjoin


## CONSTANTS
ORDER = 3
CFL_MAX = 0.4
# LIMITER = 'none'
LIMITER = 'minmod'
# LIMITER = 'superbee'
ENV = 'prod'
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

max_iter = 2200
convergence_tol = 1e-16



## COMPUTATIONAL DOMAIN

(nx,ny), (x_inp, y_inp) = readgrid(filepath=GRID_FILENAME)

x = np.zeros((nx+2, ny+2))
y = np.zeros((nx+2, ny+2))
x[1:nx+1, 1:ny+1] = np.asarray(x_inp).reshape((nx,ny))
y[1:nx+1, 1:ny+1] = np.asarray(y_inp).reshape((nx,ny))


airfoil_start = np.argwhere(x[1:,1] == 0)[0,0] + 1
airfoil_end = np.argwhere(x[1:,1] == 1)[0,0] + 1


y[1:nx+1, 0] = 2*y[1:nx+1,1] - y[1:nx+1,2]
x[1:nx+1, 0] = x[1:nx+1, 1]

y[0, :] = y[1, :]
x[0, :] = 2*x[1, :] - x[2, :]

y[nx+1, :] = y[nx, :]
x[nx+1, :] = 2*x[nx, :] - x[nx-1, :]


vol = np.zeros((nx+1, ny+1))

vol = 0.5*((x[1:,1:] - x[:-1,:-1]) * (y[:-1, 1:] - y[1:, :-1])
            - (x[:-1, 1:] - x[1:, :-1]) * (y[1:, 1:] - y[:-1, :-1]))
            
    
s_xi = np.zeros((3, nx+1, ny))
s_xi[0, 1:,1:] = y[1:nx+1, 2:ny+1] - y[1:nx+1, 1:ny]
s_xi[1, 1:,1:] = x[1:nx+1, 1:ny] - x[1:nx+1, 2:ny+1]
s_xi[2] = (s_xi[0]**2 + s_xi[1]**2)**0.5


s_eta = np.zeros((3, nx, ny+1))
s_eta[0, 1:,1:] = y[1:nx, 1:ny+1] - y[2:nx+1, 1:ny+1]
s_eta[1, 1:,1:] = x[2:nx+1, 1:ny+1] - x[1:nx, 1:ny+1]
s_eta[2] = (s_eta[0]**2 + s_eta[1]**2)**0.5


xc = 0.25*(x[:-1, :-1] + x[1:, :-1]
            + x[:-1, 1:] + x[1:, 1:])
yc = 0.25*(y[:-1, :-1] + y[1:, :-1]
            + y[:-1, 1:] + y[1:, 1:])


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
    
    r_L = (Q_0 - Q_n1) / (Q_n1 - Q_n2 + 1e-10)
    r_R = (Q_0 - Q_n1) / (Q_1 - Q_0 + 1e-10)
    
    Q_L = Q_n1 + 0.25 * MUSCL_EPS * (Q_n1 - Q_n2) * ((1 - MUSCL_K) * phi(r_L) + (1 + MUSCL_K) * r_L * phi(1/(r_L + 1e-10)))
    Q_R = Q_0 - 0.25 * MUSCL_EPS * (Q_1 - Q_0) * ((1 - MUSCL_K) * phi(r_R) + (1 + MUSCL_K) * r_R * phi(1/(r_R + 1e-10)))
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

Q = [np.zeros(Q0.shape + vol.shape)]
## INITIAL CONDITION
Q[0][:, 1:nx,1:ny] = Q0[:,None,None]


## BOUNDARY CONDITIONS
def set_boundary_conditions(Q):
    ## INLET - free-stream
    Q[:, 0,:] = Q0[:, None]
    
    ## TOP WALL - inviscid adiabatic slip
    # rho and et are const
    Q[0, 1:nx,ny] = Q[0, 1:nx,ny-1]
    Q[3, 1:nx,ny] = Q[3, 1:nx,ny-1]

    # u and v - slip conditions
    n = s_eta[0:2, 1:nx,ny] / s_eta[2, 1:nx,ny]
    Q[1, 1:nx,ny] = (n[1]**2 - n[0]**2) * Q[1, 1:nx,ny-1] - 2*n[0]*n[1] * Q[2, 1:nx,ny-1]
    Q[2, 1:nx,ny] = -2*n[0]*n[1] * Q[1, 1:nx,ny-1] - (n[1]**2 - n[0]**2) * Q[2, 1:nx,ny-1]
    
    ## BOTTOM WALL - adiabatic inviscid slip
    # rho and et are const
    Q[0, 1:airfoil_start,0] = Q[0, 1:airfoil_start,1]
    Q[3, 1:airfoil_start,0] = Q[3, 1:airfoil_start,1]

    # u and v - slip conditions
    n = s_eta[0:2, 1:airfoil_start,1] / s_eta[2, 1:airfoil_start,1]
    Q[1, 1:airfoil_start,0] = (n[1]**2 - n[0]**2) * Q[1, 1:airfoil_start,1] - 2*n[0]*n[1] * Q[2, 1:airfoil_start,1]
    Q[2, 1:airfoil_start,0] = -2*n[0]*n[1] * Q[1, 1:airfoil_start,1] - (n[1]**2 - n[0]**2) * Q[2, 1:airfoil_start,1]
    
    ## BOTTOM WALL - adiabatic inviscid slip
    # rho and et are const
    Q[0, airfoil_end+1:nx,0] = Q[0, airfoil_end+1:nx,1]
    Q[3, airfoil_end+1:nx,0] = Q[3, airfoil_end+1:nx,1]

    # u and v - slip conditions
    n = s_eta[0:2, airfoil_end+1:nx,1] / s_eta[2, airfoil_end+1:nx,1]
    Q[1, airfoil_end+1:nx,0] = (n[1]**2 - n[0]**2) * Q[1, airfoil_end+1:nx,1] - 2*n[0]*n[1] * Q[2, airfoil_end+1:nx,1]
    Q[2, airfoil_end+1:nx,0] = -2*n[0]*n[1] * Q[1, airfoil_end+1:nx,1] - (n[1]**2 - n[0]**2) * Q[2, airfoil_end+1:nx,1]
    
    ## AIRFOIL_EDGE - adiabatic inviscid slip for Euler
    # rho and et are const
    Q[0, airfoil_start:airfoil_end+1,0] = Q[0, airfoil_start:airfoil_end+1,1]
    Q[3, airfoil_start:airfoil_end+1,0] = Q[3, airfoil_start:airfoil_end+1,1]

    n = s_eta[0:2, airfoil_start:airfoil_end+1,1] / s_eta[2, airfoil_start:airfoil_end+1,1]
    Q[1, airfoil_start:airfoil_end+1,0] = (n[1]**2 - n[0]**2) * Q[1, airfoil_start:airfoil_end+1,1] - 2*n[0]*n[1] * Q[2, airfoil_start:airfoil_end+1,1]
    Q[2, airfoil_start:airfoil_end+1,0] = -2*n[0]*n[1] * Q[1, airfoil_start:airfoil_end+1,1] - (n[1]**2 - n[0]**2) * Q[2, airfoil_start:airfoil_end+1,1]
    
    ## OUTLET - free-stream
    Q[:, nx,:] = Q[:, nx-1,:]

set_boundary_conditions(Q[0])

## TIME MARCHING

l2_err_norms = np.ones((max_iter, 4))
linf_err_norms = np.ones((max_iter, 4))

curr_iter = 1
while (np.max(l2_err_norms[curr_iter - 1]) > convergence_tol and curr_iter < max_iter):

    E_xi_curr = np.zeros((4,) + s_xi.shape[1:])
    
    Q_L, Q_R = get_MUSCL_Q_LR(Q[-1][:, 0,1:ny], Q[-1][:, 0,1:ny], Q[-1][:, 1,1:ny], Q[-1][:, 2,1:ny])
    E_xi_curr[:, 1,1:ny] = 0.5*(get_E_xi(Q_L, s_xi[:, 1,1:ny]) + get_E_xi(Q_R, s_xi[:, 1,1:ny]))
    E_xi_curr[:, 1,1:ny] -= get_flux_dissipation(Q_L, Q_R, s_xi[:, 1,1:ny])

    Q_L, Q_R = get_MUSCL_Q_LR(Q[-1][:, nx-2,1:ny], Q[-1][:, nx-1,1:ny], Q[-1][:, nx,1:ny], Q[-1][:, nx,1:ny])
    E_xi_curr[:, nx,1:ny] = 0.5*(get_E_xi(Q_L, s_xi[:, nx,1:ny]) + get_E_xi(Q_R, s_xi[:, nx,1:ny]))
    E_xi_curr[:, nx,1:ny] -= get_flux_dissipation(Q_L, Q_R, s_xi[:, nx,1:ny])

    Q_L, Q_R = get_MUSCL_Q_LR(Q[-1][:, :nx-2,1:ny], Q[-1][:, 1:nx-1,1:ny], Q[-1][:, 2:nx,1:ny], Q[-1][:, 3:nx+1,1:ny])
    E_xi_curr[:, 2:nx,1:ny] = 0.5*(get_E_xi(Q_L, s_xi[:, 2:nx,1:ny]) + get_E_xi(Q_R, s_xi[:, 2:nx,1:ny]))
    E_xi_curr[:, 2:nx,1:ny] -= get_flux_dissipation(Q_L, Q_R, s_xi[:, 2:nx,1:ny])

    
    F_eta_curr = np.zeros((4,) + s_eta.shape[1:])

    Q_L, Q_R = get_MUSCL_Q_LR(Q[-1][:, 1:nx,0], Q[-1][:, 1:nx,0], Q[-1][:, 1:nx,1], Q[-1][:, 1:nx,2])
    F_eta_curr[:, 1:nx,1] = 0.5*(get_F_eta(Q_L, s_eta[:, 1:nx,1]) + get_F_eta(Q_R, s_eta[:, 1:nx,1]))
    F_eta_curr[:, 1:nx,1] -= get_flux_dissipation(Q_L, Q_R, s_eta[:, 1:nx,1])

    Q_L, Q_R = get_MUSCL_Q_LR(Q[-1][:, 1:nx,ny-2], Q[-1][:, 1:nx,ny-1], Q[-1][:, 1:nx,ny], Q[-1][:, 1:nx,ny])
    F_eta_curr[:, 1:nx,ny] = 0.5*(get_F_eta(Q_L, s_eta[:, 1:nx,ny]) + get_F_eta(Q_R, s_eta[:, 1:nx,ny]))
    F_eta_curr[:, 1:nx,ny] -= get_flux_dissipation(Q_L, Q_R, s_eta[:, 1:nx,ny])

    Q_L, Q_R = get_MUSCL_Q_LR(Q[-1][:, 1:nx,:ny-2], Q[-1][:, 1:nx,1:ny-1], Q[-1][:, 1:nx,2:ny], Q[-1][:, 1:nx,3:ny+1])
    F_eta_curr[:, 1:nx,2:ny] = 0.5*(get_F_eta(Q_L, s_eta[:, 1:nx,2:ny]) + get_F_eta(Q_R, s_eta[:, 1:nx,2:ny]))
    F_eta_curr[:, 1:nx,2:ny] -= get_flux_dissipation(Q_L, Q_R, s_eta[:, 1:nx,2:ny])


    Q_diff = np.zeros(Q[-1].shape)
    p_cell = (gamma - 1) * (Q[-1][3, 1:nx,1:ny] - 0.5*(Q[-1][1, 1:nx,1:ny]**2 + Q[-1][2, 1:nx,1:ny]**2) / Q[-1][0, 1:nx,1:ny])
    c_cell = np.sqrt(gamma * p_cell / Q[-1][0, 1:nx,1:ny])

    n_xi_cell = 0.5*(s_xi[:2, 1:nx,1:ny]/s_xi[2, 1:nx,1:ny] + s_xi[:2, 2:nx+1,1:ny]/s_xi[2, 2:nx+1,1:ny])
    s_xi_cell = 0.5 * (s_xi[2, 1:nx,1:ny] + s_xi[2, 2:nx+1,1:ny])

    U_xi_cell = Q[-1][1, 1:nx,1:ny]*n_xi_cell[0] + Q[-1][2, 1:nx,1:ny]*n_xi_cell[1]

    n_eta_cell = 0.5*(s_eta[:2, 1:nx,1:ny]/s_eta[2, 1:nx,1:ny] + s_eta[:2, 1:nx,2:ny+1]/s_eta[2, 1:nx,2:ny+1])
    s_eta_cell = 0.5 * (s_eta[2, 1:nx,1:ny] + s_eta[2, 1:nx,2:ny+1])

    V_eta_cell = Q[-1][1, 1:nx,1:ny]*n_eta_cell[0] + Q[-1][2, 1:nx,1:ny]*n_eta_cell[1]

    rho_xi = (abs(U_xi_cell) + c_cell) * s_xi_cell
    rho_eta = (abs(V_eta_cell) + c_cell) * s_eta_cell

    delta_tau_cap = CFL_MAX / (rho_xi + rho_eta)


    Q_diff[:, 1:nx,1:ny] = - (delta_tau_cap) * (
        (E_xi_curr[:, 2:nx+1,1:ny] * s_xi[2, 2:nx+1,1:ny] - E_xi_curr[:, 1:nx,1:ny] * s_xi[2, 1:nx,1:ny])
        + (F_eta_curr[:, 1:nx,2:ny+1] * s_eta[2, 1:nx,2:ny+1] - F_eta_curr[:, 1:nx,1:ny] * s_eta[2, 1:nx,1:ny])
        )
    
    Q.append(Q[-1] + Q_diff)
    
    set_boundary_conditions(Q[-1])

    ## OUTLET - first order interp
    Q[-1][:, nx,1:ny] = Q[-2][:, nx-1,1:ny]
    

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
            l2_err_norms.resize((max_iter,) + l2_err_norms.shape[1:], refcheck=False)
            linf_err_norms.resize((max_iter,) + linf_err_norms.shape[1:], refcheck=False)


np.save(ospathjoin("data", f"Q_{FILE_NAME_SUFFIX}"), Q)
np.save(ospathjoin("data", f"l2_err_norms_{FILE_NAME_SUFFIX}"), l2_err_norms[1:curr_iter])
np.save(ospathjoin("data", f"linf_err_norms_{FILE_NAME_SUFFIX}"), linf_err_norms[1:curr_iter])


titles = ["RHO", "RHO U", "RHO V", "RHO ET"]
for i in range(4):
    plt.figure()
    plt.semilogy(list(range(1, curr_iter)), l2_err_norms[1:,i])
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