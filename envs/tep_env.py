import gymnasium as gym
from gymnasium import spaces
import numpy as np
import tensorflow as tf

import os

class TEPEnv(gym.Env):
    def __init__(self, model_path=None):
        if model_path is None:
            # Resolve absolute path relative to this file
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "models", "tep_surrogate.keras")
            
        self.model = tf.keras.models.load_model(model_path)
        
        # Action space: 11 XMVs (Manipulated Variables)
        # Scaled values (StandardScaler output, roughly -3 to 3)
        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(11,), dtype=np.float32)
        
        # Observation space: 52 Process Variables (XMEAS + XMV)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(52,), dtype=np.float32)
        
        # Initial State
        self.state = np.zeros(52, dtype=np.float32)
        self.steps = 0
        self.max_steps = 200 # Episode limit
        
        # Indices for XMEAS (0-40) and XMV (41-51)
        self.xmeas_idx = slice(0, 41)
        self.xmv_idx = slice(41, 52)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Start near steady-state (0 after scaling) with small noise
        self.state = np.random.normal(0, 0.1, size=(52,)).astype(np.float32)
        self.steps = 0
        return self.state, {}

    def step(self, action):
        # 1. Apply Action: Update XMV part of state
        # The agent provides the DESIRED valve position
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.state[self.xmv_idx] = action
        
        # 2. Simulate Physics (Predict Next State)
        # LSTM expects (Batch, Time, Features) -> (1, 1, 52)
        input_tensor = self.state.reshape(1, 1, 52)
        
        # Predict next step
        # Optimize: use tf.function compiled call
        if not hasattr(self, 'predict_fn'):
             self.predict_fn = tf.function(lambda x: self.model(x, training=False))
        
        next_state_tensor = self.predict_fn(input_tensor)
        self.state = next_state_tensor.numpy()[0, 0, :]
        
        # 3. Calculate Reward
        # Goal: Keep XMEAS close to 0 (Target Setpoint)
        # Penalty is proportional to squared error
        xmeas = self.state[self.xmeas_idx]
        loss = np.mean(np.square(xmeas))
        reward = -loss
        
        # 4. Check Termination
        self.steps += 1
        terminated = False
        truncated = self.steps >= self.max_steps
        
        # Safety Explosion Check (State > 10 std deviations)
        if np.max(np.abs(self.state)) > 10.0:
            terminated = True
            reward -= 100.0 # Huge Penalty for blowing up the plant
            
        return self.state, float(reward), terminated, truncated, {}

    def render(self):
        pass
