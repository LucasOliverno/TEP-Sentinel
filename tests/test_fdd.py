import numpy as np
from sklearn.preprocessing import StandardScaler

from common import class_weights
from fdd import AlarmFilter, DiagnosisSmoother, top_deviations
from sources import SyntheticOffsets


def test_alarm_needs_consecutive_hits_and_clears():
    f = AlarmFilter(confirm=3, clear=2)
    assert [f.update(x) for x in (1, 1, 0, 1, 1, 1)] == [False, False, False, False, False, True]
    assert f.update(0) is True  # one normal sample is not enough to clear
    assert f.update(0) is False


def test_smoother_averages():
    s = DiagnosisSmoother(n=2)
    s.update(np.array([1.0, 0.0]))
    s.update(np.array([0.0, 1.0]))
    out = s.update(np.array([0.0, 1.0]))  # oldest sample dropped
    assert out.tolist() == [0.0, 1.0]


def test_top_deviations_ranks_by_sigma():
    rng = np.random.default_rng(0)
    scaler = StandardScaler().fit(rng.normal(size=(200, 52)))
    window = np.zeros((100, 52))
    window[-10:, 7] = 6.0
    window[-10:, 3] = -2.0
    devs = top_deviations(window, scaler, k=2)
    assert [d["index"] for d in devs] == [7, 3]
    assert devs[0]["z"] == 6.0


def test_synthetic_offsets_ramp():
    o = SyntheticOffsets({0: -8.0}, ramp_steps=4)
    vals = [o.apply(np.zeros(52))[0] for _ in range(6)]
    assert vals[0] == -2.0 and vals[3] == -8.0 and vals[5] == -8.0
    assert SyntheticOffsets().apply(np.ones(52))[0] == 1.0


def test_class_weights_balance():
    w = class_weights(np.array([0] * 90 + [1] * 10), 3)
    assert w[1] > w[0] and w[2] == 1.0
