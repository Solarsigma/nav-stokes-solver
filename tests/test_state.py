from state import get_Q, get_Qv
from constants.fluid_properties import R, gamma
import pytest

def test_get_Q():
    Q = get_Q(287,1,1,1)
    Q_actual = [1, 1, 1, 718.5]
    assert Q == pytest.approx(Q_actual)

def test_get_Qv():
    Qv = get_Qv([1,1,1,718.5])
    Qv_actual = [287,1,1,1]
    assert Qv == pytest.approx(Qv_actual)

def test_round_trip_Q_Qv():
    Q = [1,1,1,718.5]
    Qv = get_Qv(Q)
    assert get_Q(Qv[0], Qv[1], Qv[2], Qv[3]) == pytest.approx(Q)

    assert get_Qv(get_Q(Qv[0], Qv[1], Qv[2], Qv[3])) == pytest.approx(Qv)