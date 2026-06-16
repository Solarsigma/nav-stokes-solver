from constants.fluid_properties import gamma, R, Cp, Pr

def get_mu(Q):
    T = (gamma - 1) * (Q[3] - (0.5 * (Q[1]**2 + Q[2]**2)) / Q[0]) / (Q[0] * R)
    return 1.458 * 1e-6 * (T**1.5) / (T + 110.4)


def get_k(mu):
    return Cp * mu / Pr
