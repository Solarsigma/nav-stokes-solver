import numpy as np
import matplotlib.pyplot as plt
from readgrid import readgrid

ORDER = 3
LIMITER = 'minmod'
ENV = 'prod'
# GRID_FILENAME = 'g65x49u.dat' if ENV == 'prod' else 'g33x25u.dat'
FILE_NAME_SUFFIX = f'{ENV}_euler_inviscid_order_{ORDER}_limiter_{LIMITER}'

Q = np.load(f'Q_{FILE_NAME_SUFFIX}.npy')

filename = 'g65x49u.dat'

(nx,ny), (x_inp, y_inp) = readgrid(filepath=filename)

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


before_cell = (airfoil_start - 12, 2)
after_cell = (airfoil_start + 12, 4)

R = 287 # J/(kg-K)
Cp = 1005 # J/(kg-K)
gamma = 1.4
M = 2
c = 347.2 # m/s
u_ref = 694.4 # m/s
p_ref = 101325 # Pa
T_ref = 300 # K

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


def get_c(T):
    return np.sqrt(gamma * R * T)


Q_steady = Q[-1]

Qv_steady = get_Qv(Q_steady)

rho = Qv_steady[0]
p = Qv_steady[1]
u = Qv_steady[2]
v = Qv_steady[3]
T = Qv_steady[4]

print("rho")
print(rho[after_cell])
print(rho[before_cell])
print(rho[after_cell] / rho[before_cell])
print("\n")


print("p")
print(p[after_cell])
print(p[before_cell])
print(p[after_cell] / p[before_cell])
print("\n")


print("T")
print(T[after_cell])
print(T[before_cell])
print(T[after_cell] / T[before_cell])
print("\n")


a2 = get_c(T[after_cell])
a1 = get_c(T[before_cell])

V2 = np.sqrt(u[after_cell]**2 + v[after_cell]**2)
V1 = np.sqrt(u[before_cell]**2 + v[before_cell]**2)

M2 = V2 / a2
M1 = V1 / a1

print("M")
print(M2)
print(M1)
print(M2 / M1)
print("\n")