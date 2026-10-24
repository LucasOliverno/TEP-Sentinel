import keras
import numpy as np
from keras.layers import LSTM, Dense, Input
from keras.models import Sequential

from config import N_FEATURES
from envs.tep_env import TEPEnv


def tiny_model():
    m = Sequential([Input(shape=(None, N_FEATURES)), LSTM(8, return_sequences=True), Dense(N_FEATURES)])
    return m


def test_env_contract_and_no_aliasing():
    keras.utils.set_random_seed(0)
    env = TEPEnv(model=tiny_model(), max_steps=5)
    obs, _ = env.reset(seed=1)
    assert obs.shape == (N_FEATURES,)
    # reset perturbs the plant: the task is non-trivial
    assert np.mean(obs[:41] ** 2) > 0.1
    obs2, r, term, trunc, info = env.step(np.zeros(11, np.float32))
    obs2[:] = 123.0  # mutating the returned array must not corrupt the env
    assert not np.any(env.state == 123.0)
    assert np.isfinite(r) and "xmeas_mse" in info
    for _ in range(4):
        _, _, term, trunc, _ = env.step(np.zeros(11, np.float32))
    assert trunc or term


def test_env_is_reproducible_with_seed():
    env = TEPEnv(model=tiny_model())
    a, _ = env.reset(seed=7)
    b, _ = env.reset(seed=7)
    assert np.allclose(a, b)
