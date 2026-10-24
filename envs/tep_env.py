import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import gymnasium as gym
import numpy as np
import tensorflow as tf
from gymnasium import spaces

from config import N_FEATURES, N_XMEAS, N_XMV, SURROGATE_PATH

EXPLOSION_LIMIT = 10.0  # |z| above this means the plant "blew up"


class TEPEnv(gym.Env):
    """Disturbance-rejection task on the neural surrogate of the TEP.

    Each episode starts from a perturbed state (N(0, init_std) on the XMEAS, in scaled units)
    and the agent must bring the measurements back to the setpoint (0 = nominal).
    The surrogate is fed a rolling context of the last `context_len` steps, because the LSTM
    needs history to model the dynamics.
    """

    metadata = {"render_modes": []}

    def __init__(self, model_path=None, init_std=1.0, context_len=20, max_steps=200, action_penalty=0.01,
                 model=None):
        super().__init__()
        self.model = model if model is not None else tf.keras.models.load_model(model_path or SURROGATE_PATH)
        self.init_std, self.context_len = init_std, context_len
        self.max_steps, self.action_penalty = max_steps, action_penalty

        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(N_XMV,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(N_FEATURES,), dtype=np.float32)
        self.xmeas_idx = slice(0, N_XMEAS)
        self.xmv_idx = slice(N_XMEAS, N_FEATURES)

        self.state = np.zeros(N_FEATURES, dtype=np.float32)
        self._context = np.zeros((context_len, N_FEATURES), dtype=np.float32)
        self.steps = 0
        self._predict = tf.function(lambda x: self.model(x, training=False))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        state = np.zeros(N_FEATURES, dtype=np.float32)
        state[self.xmeas_idx] = self.np_random.normal(0, self.init_std, N_XMEAS)
        state[self.xmv_idx] = self.np_random.normal(0, 0.1, N_XMV)
        self.state = state
        self._context = np.tile(state, (self.context_len, 1))
        self.steps = 0
        return self.state.copy(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), self.action_space.low, self.action_space.high)
        # The agent sets the valve positions; the surrogate predicts the next plant state.
        self._context[-1, self.xmv_idx] = action
        pred = self._predict(self._context[None]).numpy()[0, -1, :]
        next_state = pred.astype(np.float32)
        next_state[self.xmv_idx] = action  # valves stay where the agent put them

        self._context = np.roll(self._context, -1, axis=0)
        self._context[-1] = next_state
        self.state = next_state

        error = float(np.mean(np.square(next_state[self.xmeas_idx])))
        reward = -error - self.action_penalty * float(np.mean(np.square(action)))

        self.steps += 1
        terminated = bool(np.max(np.abs(next_state)) > EXPLOSION_LIMIT)
        if terminated:
            reward -= 100.0
        truncated = self.steps >= self.max_steps
        return self.state.copy(), reward, terminated, truncated, {"xmeas_mse": error}
