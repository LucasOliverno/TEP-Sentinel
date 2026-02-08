import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from envs.tep_env import TEPEnv
import os
import time

# Create log dir
MODELS_DIR = "models/PPO"
LOG_DIR = "logs"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def train():
    # Create Environment
    # Wrap in Monitor to log stats (reward per episode) for Tensorboard
    env = Monitor(TEPEnv(), LOG_DIR)
    
    # Create Model
    # MlpPolicy because our state is a vector (52,), not an image
    # verbose=1 to see progress
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR)

    # Train
    TIMESTEPS = 20000 
    checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=MODELS_DIR, name_prefix="tep_ppo")

    print(f"Starting PPO Training for {TIMESTEPS} timesteps...")
    start_time = time.time()
    
    model.learn(total_timesteps=TIMESTEPS, callback=checkpoint_callback)
    
    print(f"Training Complete in {time.time() - start_time:.2f}s.")
    
    # Save Final Model
    final_path = f"{MODELS_DIR}/tep_final_ppo"
    model.save(final_path)
    print(f"Model saved to {final_path}")

if __name__ == "__main__":
    train()
