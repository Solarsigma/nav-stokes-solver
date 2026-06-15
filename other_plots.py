import numpy as np
import matplotlib.pyplot as plt
from readgrid import readgrid
from matplotlib.collections import LineCollection
import os

ORDER = 1
LIMITER = 'none'
ENV = 'prod'
# GRID_FILENAME = 'g65x49u.dat' if ENV == 'prod' else 'g33x25u.dat'
# FILE_NAME_SUFFIX = f'{ENV}_euler_inviscid_order_{ORDER}_limiter_{LIMITER}'
FILE_NAME_SUFFIX = f'{ENV}_nav_stokes_order_{ORDER}_limiter_{LIMITER}'
GRID_FILENAME = 'g65x65s.dat'
data_source_path = './data'

Q = np.load(os.path.join(data_source_path, f'Q_{FILE_NAME_SUFFIX}.npy'))

l2_err_norms = np.load(os.path.join(data_source_path, f'l2_err_norms_{FILE_NAME_SUFFIX}.npy'))

linf_err_norms = np.load(os.path.join(data_source_path, f'linf_err_norms_{FILE_NAME_SUFFIX}.npy'))

R = 287 # J/(kg-K)
Cp = 1005 # J/(kg-K)
gamma = 1.4
M = 2
c = 347.2 # m/s
u_ref = 694.4 # m/s
p_ref = 101325 # Pa
T_ref = 300 # K


(nx,ny), (x_inp, y_inp) = readgrid(filepath=GRID_FILENAME)

x = np.zeros((nx+2, ny+2))
y = np.zeros((nx+2, ny+2))

airfoil_start = None
airfoil_end = None

for i in range(1, nx+1):
    for j in range(1, ny+1):
        x[i,j] = x_inp[i-1 + (j-1)*nx]
        y[i,j] = y_inp[i-1 + (j-1)*nx]

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
xc = np.empty((nx+1, ny+1))
yc = np.empty((nx+1, ny+1))

for i in range(nx+1):
    for j in range(ny+1):
        xc[i,j] = 0.25*(x[i,j] + x[i+1,j]
                   + x[i,j+1] + x[i+1,j+1])
        yc[i,j] = 0.25*(y[i,j] + y[i+1,j]
                   + y[i,j+1] + y[i+1,j+1])


curr_iter = l2_err_norms.shape[0]

def get_Qv(Q):

    rho = Q[0]

    u = Q[1] / rho
    v = Q[2] / rho

    p = (gamma - 1) * (
        Q[3] - 0.5 * (Q[1]**2 + Q[2]**2) / rho
    )

    T = p / (rho * R)

    return np.asarray([
        rho,
        p,
        u,
        v,
        T
    ])

Q_steady = Q[-1]
Qv_steady = get_Qv(Q_steady)

# ---------------------------------------------
# CLEAN UP DATA
# ---------------------------------------------
# Remove the unused first row (index 0)
iters = np.arange(1, curr_iter)

l2_plot = l2_err_norms[1:curr_iter]
linf_plot = linf_err_norms[1:curr_iter]

var_labels = [
    r"$\rho$ / $rho_0$",
    r"$\rho u$ / $(\rho u)_0$",
    r"$\rho v$ / $(\rho v)_0$",
    r"$\rho e_t$ / $(\rho e_t)_0$"
]

# ---------------------------------------------
# PLOT SETTINGS
# ---------------------------------------------
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 15,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.figsize": (9, 6),
    "lines.linewidth": 2.0,
    "grid.alpha": 0.3
})

mesh = np.asarray([[[x[i,j], y[i,j]] for j in range(1, ny+1)] for i in range(1, nx+1)])

plt.figure()

plt.pcolormesh(xc[1:-1,1:-1], yc[1:-1,1:-1], Qv_steady[3,1:-1,1:-1])
plt.colorbar()
plt.title("v")

segs1 = mesh
segs2 = segs1.transpose(1,0,2)
plt.gca().add_collection(LineCollection(segs1, color='black', linewidths=0.2))
plt.gca().add_collection(LineCollection(segs2, color='black', linewidths=0.2))
plt.gca().autoscale()

# plt.savefig(f"v_{FILE_NAME_SUFFIX}.png", dpi=300)

# plt.show()

plt.figure()

plt.pcolormesh(xc[1:-1,1:-1], yc[1:-1,1:-1], Qv_steady[2,1:-1,1:-1])
plt.colorbar()
plt.title("u")

segs1 = mesh
segs2 = segs1.transpose(1,0,2)
plt.gca().add_collection(LineCollection(segs1, color='black', linewidths=0.2))
plt.gca().add_collection(LineCollection(segs2, color='black', linewidths=0.2))
plt.gca().autoscale()

plt.figure()

plt.pcolormesh(xc[1:-1,1:-1], yc[1:-1,1:-1], Qv_steady[1,1:-1,1:-1])
plt.colorbar()
plt.title("p")

segs1 = mesh
segs2 = segs1.transpose(1,0,2)
plt.gca().add_collection(LineCollection(segs1, color='black', linewidths=0.2))
plt.gca().add_collection(LineCollection(segs2, color='black', linewidths=0.2))
plt.gca().autoscale()

plt.figure()

plt.pcolormesh(xc[1:-1,1:-1], yc[1:-1,1:-1], Qv_steady[4,1:-1,1:-1])
plt.colorbar()
plt.title("T")

segs1 = mesh
segs2 = segs1.transpose(1,0,2)
plt.gca().add_collection(LineCollection(segs1, color='black', linewidths=0.2))
plt.gca().add_collection(LineCollection(segs2, color='black', linewidths=0.2))
plt.gca().autoscale()

plt.figure()

plt.pcolormesh(xc[1:-1,1:-1], yc[1:-1,1:-1], Qv_steady[0,1:-1,1:-1])
plt.colorbar()
plt.title("rho")

segs1 = mesh
segs2 = segs1.transpose(1,0,2)
plt.gca().add_collection(LineCollection(segs1, color='black', linewidths=0.2))
plt.gca().add_collection(LineCollection(segs2, color='black', linewidths=0.2))
plt.gca().autoscale()

# plt.savefig(f"v_{FILE_NAME_SUFFIX}.png", dpi=300)

plt.show()
# plt.pcolormesh(xc, yc, Q_curr[2,:,:])
# plt.colorbar()
# plt.title("rho v")

# segs1 = mesh
# segs2 = segs1.transpose(1,0,2)
# plt.gca().add_collection(LineCollection(segs1, color='black', linewidths=0.2))
# plt.gca().add_collection(LineCollection(segs2, color='black', linewidths=0.2))
# plt.gca().autoscale()

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


# ---------------------------------------------
# INDIVIDUAL L2 CONVERGENCE PLOTS
# ---------------------------------------------
# for k in range(4):

#     fig, ax = plt.subplots()

#     ax.semilogy(
#         iters,
#         l2_plot[:, k],
#         # marker='o',
#         # markersize=4,
#         label=f"L2 Error: {var_labels[k]}"
#     )

#     ax.set_xlabel("Iteration")
#     ax.set_ylabel("L2 Residual Norm")
#     ax.set_title(f"Convergence History — {var_labels[k]}")
#     ax.grid(True, which='both')
#     ax.legend()

#     plt.tight_layout()

# ---------------------------------------------
# COMBINED L2 PLOT
# ---------------------------------------------
# fig, ax = plt.subplots()

# for k in range(4):
#     ax.semilogy(
#         iters,
#         l2_plot[:, k],
#         # marker='o',
#         # markersize=3,
#         label=var_labels[k]
#     )

# ax.set_xlabel("Iteration")
# ax.set_ylabel("L2 Residual Norm")
# ax.set_title("L2 Convergence History")
# ax.grid(True, which='both')
# ax.legend()

# plt.tight_layout()
# plt.savefig(f"l2_convergence_{FILE_NAME_SUFFIX}.png", dpi=300)

# # ---------------------------------------------
# # COMBINED LINF PLOT
# # ---------------------------------------------
# fig, ax = plt.subplots()

# for k in range(4):
#     ax.semilogy(
#         iters,
#         linf_plot[:, k],
#         # marker='s',
#         # markersize=3,
#         label=var_labels[k]
#     )

# ax.set_xlabel("Iteration")
# ax.set_ylabel(r"$L_\infty$ Residual Norm")
# ax.set_title(r"$L_\infty$ Convergence History")
# ax.grid(True, which='both')
# ax.legend()

# plt.tight_layout()

# # ---------------------------------------------
# # OPTIONAL: SAVE FIGURES
# # ---------------------------------------------
# plt.show()