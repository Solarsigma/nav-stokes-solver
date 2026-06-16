from constants.fluid_properties import R, gamma
import numpy as np

def get_Qv(Q):
    p = (gamma - 1) * (Q[3] - (0.5 * (Q[1]**2 + Q[2]**2) / Q[0]))
    return np.asarray([
        p,
        Q[1] / Q[0],
        Q[2] / Q[0],
        p / (R * Q[0])
    ])


def get_Q(p, u, v, T):
    rho = p / (R * T)
    return np.asarray([
        rho,
        rho * u,
        rho * v,
        (p / (gamma - 1)) + 0.5 * rho * (u**2 + v**2)
    ])
