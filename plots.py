import numpy as np
import matplotlib.pyplot as plt
from os.path import join as ospathjoin

ORDER = 3
LIMITER = 'minmod'
ENV = 'prod'
# GRID_FILENAME = 'g65x49u.dat' if ENV == 'prod' else 'g33x25u.dat'
FILE_NAME_SUFFIX = f'{ENV}_euler_inviscid_order_{ORDER}_limiter_{LIMITER}'

Q = np.load(f'Q_{FILE_NAME_SUFFIX}.npy')

l2_err_norms = np.load(f'l2_err_norms_{FILE_NAME_SUFFIX}.npy')

linf_err_norms = np.load(f'linf_err_norms_{FILE_NAME_SUFFIX}.npy')

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
        # marker='o',
        # markersize=3,
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