"""Vectorized-env factory. `DummyVecEnv` for smoke tests (fast startup, no
subprocess overhead); `SubprocVecEnv` for real training runs to use the
machine's 15 CPU cores (docs/RESEARCH_LOG.md machine audit)."""

from __future__ import annotations

from stable_baselines3.common.env_util import make_vec_env as sb3_make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from config.loader import Config
from envs.g1_walk_env import G1WalkEnv


def make_vec_env(config: Config, n_envs: int, seed: int, subproc: bool = True):
    def env_fn():
        return G1WalkEnv(config)

    vec_cls = SubprocVecEnv if (subproc and n_envs > 1) else DummyVecEnv
    return sb3_make_vec_env(env_fn, n_envs=n_envs, seed=seed, vec_env_cls=vec_cls)
