from matplotlib import pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from os.path import join as ospathjoin
from constants.fluid_properties import p_ref, u_ref, T_ref
import config
from grid import Grid2D
from solver import Solver
from state import get_Q

## CONSTANTS
if config.ENV == 'prod':
    max_iter = 1000
elif config.ENV == 'dev':
    max_iter = 200
elif config.ENV == 'debug':
    max_iter = 50
else:
    max_iter = 200

convergence_tol = 1e-16

## COMPUTATIONAL DOMAIN

grid = Grid2D(filename = ospathjoin(config.GRID_FILEPATH, config.GRID_FILENAME))

Q0 = get_Q(p_ref, u_ref, 0, T_ref)
Q_ref = np.asarray([Q0[0], Q0[1], Q0[1], Q0[3]])
solver = Solver(grid=grid, Q_init=Q0)

## TIME MARCHING

l2_err_norms = np.ones((max_iter, 4))
linf_err_norms = np.ones((max_iter, 4))

curr_iter = 1
while (np.max(l2_err_norms[curr_iter - 1]) > convergence_tol and curr_iter < max_iter):
    
    solver.advance_time_step()
    Q_diff = solver.Q_diff_curr

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


np.save(ospathjoin(config.OUTP_FILEPATH, f"Q_{config.OUTP_FILE_NAME_SUFFIX}"), solver.Q)
np.save(ospathjoin(config.OUTP_FILEPATH, f"l2_err_norms_{config.OUTP_FILE_NAME_SUFFIX}"), l2_err_norms[1:curr_iter])
np.save(ospathjoin(config.OUTP_FILEPATH, f"linf_err_norms_{config.OUTP_FILE_NAME_SUFFIX}"), linf_err_norms[1:curr_iter])


titles = ["RHO", "RHO U", "RHO V", "RHO ET"]
for i in range(4):
    plt.figure()
    plt.semilogy(list(range(1, curr_iter)), l2_err_norms[1:,i])
    plt.title(titles[i])
plt.show()


# TODO: AUSM