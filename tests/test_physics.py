from physics import get_k, get_mu
from constants.fluid_properties import R, gamma, Cp, Pr
import pytest

def test_get_mu():
    T = 273.15 # K
    Q = [1, 0, 0, R * T / (gamma - 1)]
    mu = get_mu(Q)
    assert mu == pytest.approx(1.716 * 1e-5, rel=1e-2)

def test_get_k():
    k = get_k(1)
    assert k == pytest.approx(Cp / Pr)