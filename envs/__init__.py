from gymnasium.envs.registration import register

from .tep_env import TEPEnv

__all__ = ["TEPEnv"]

register(
    id='TEP-v0',
    entry_point='envs.tep_env:TEPEnv',
)
