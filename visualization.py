import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from os.path import join as ospathjoin
from constants.fluid_properties import *
from state import get_Qv
from grid import Grid2D

ORDER = 3
LIMITER = 'minmod'
ENV = 'prod'
FILE_NAME_SUFFIX = f'{ENV}_nav_stokes_order_{ORDER}_limiter_{LIMITER}'
GRID_FILENAME = 'g65x65s.dat'
GRID_FILEPATH = 'mesh'
FILE_PATH = 'data'

grid = Grid2D(ospathjoin(GRID_FILEPATH, GRID_FILENAME))

Q = np.load(ospathjoin(FILE_PATH, f'Q_{FILE_NAME_SUFFIX}.npy'))

l2_err_norms = np.load(ospathjoin(FILE_PATH, f'l2_err_norms_{FILE_NAME_SUFFIX}.npy'))

linf_err_norms = np.load(ospathjoin(FILE_PATH, f'linf_err_norms_{FILE_NAME_SUFFIX}.npy'))

curr_iter = l2_err_norms.shape[0]

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


fig, ax = plt.subplots()

for k in range(4):
    ax.semilogy(
        iters,
        l2_plot[:, k],
        label=var_labels[k]
    )

ax.set_xlabel("Iteration")
ax.set_ylabel("L2 Residual Norm")
ax.set_title("L2 Convergence History")
ax.grid(True, which='both')
ax.legend()

plt.tight_layout()
plt.savefig(ospathjoin("plots", f"l2_convergence_{FILE_NAME_SUFFIX}.png"), dpi=300)


fig, ax = plt.subplots()

for k in range(4):
    ax.semilogy(
        iters,
        linf_plot[:, k],
        # marker='s',
        # markersize=3,
        label=var_labels[k]
    )

ax.set_xlabel("Iteration")
ax.set_ylabel(r"$L_\infty$ Residual Norm")
ax.set_title(r"$L_\infty$ Convergence History")
ax.grid(True, which='both')
ax.legend()

plt.tight_layout()


plt.savefig(ospathjoin("plots", f"linf_convergence_{FILE_NAME_SUFFIX}.png"), dpi=300)
plt.show()


x,y = grid.coords
nx,ny = grid.n
xc,yc = grid.cell_centers

Qv_steady = get_Qv(Q[-1])

mesh = np.asarray([[[x[i,j], y[i,j]] for j in range(1, ny+1)] for i in range(1, nx+1)])

plt.figure()

plt.pcolormesh(xc[1:-1,1:-1], yc[1:-1,1:-1], Qv_steady[2,1:-1,1:-1])
plt.colorbar()
plt.title("v")

segs1 = mesh
segs2 = segs1.transpose(1,0,2)
plt.gca().add_collection(LineCollection(segs1, color='black', linewidths=0.2))
plt.gca().add_collection(LineCollection(segs2, color='black', linewidths=0.2))
plt.gca().autoscale()

plt.show()
# plt.savefig(ospathjoin('plots', f"v_{FILE_NAME_SUFFIX}.png"), dpi=300)