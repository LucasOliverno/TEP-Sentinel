"""Train a PPO agent on the surrogate environment and compare it with a do-nothing baseline.

The best checkpoint (by evaluation reward) is saved to models/PPO/best_model.zip, which is the
file TEPSentinelSystem loads.
"""
import argparse

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from common import log, save_json, set_seed, setup_logging
from config import MODELS_DIR, RL_BEST_PATH, ROOT
from envs.tep_env import TEPEnv

LOG_DIR = ROOT / "logs"


def evaluate(policy, env, episodes=20, seed=1000):
    """Mean cumulative reward and mean final XMEAS MSE over fixed-seed episodes."""
    rewards, finals, blowups = [], [], 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        total, done, info = 0.0, False, {}
        while not done:
            obs, r, term, trunc, info = env.step(policy(obs))
            total += r
            done = term or trunc
            blowups += int(term)
        rewards.append(total)
        finals.append(info["xmeas_mse"])
    return {"mean_reward": float(np.mean(rewards)), "final_xmeas_mse": float(np.mean(finals)), "blowups": blowups}


def train(timesteps: int):
    setup_logging()
    set_seed()
    RL_BEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    env = Monitor(TEPEnv(), str(LOG_DIR))
    eval_env = Monitor(TEPEnv())
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=str(LOG_DIR), seed=42)
    cb = EvalCallback(eval_env, best_model_save_path=str(RL_BEST_PATH.parent), log_path=str(LOG_DIR),
                      eval_freq=5000, n_eval_episodes=5, deterministic=True)
    model.learn(total_timesteps=timesteps, callback=cb)

    best = PPO.load(RL_BEST_PATH)
    zeros = np.zeros(11, dtype=np.float32)
    report = {
        "baseline_zero_action": evaluate(lambda o: zeros, TEPEnv()),
        "ppo_best": evaluate(lambda o: best.predict(o, deterministic=True)[0], TEPEnv()),
        "timesteps": timesteps,
    }
    save_json(MODELS_DIR / "rl_report.json", report)
    log.info("RL report: %s", report)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=100_000)
    train(ap.parse_args().timesteps)
