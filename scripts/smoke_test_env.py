"""Vertical slice 3: environment contract. Validates the Gymnasium API
(reset/step shapes, dtypes, bounds), deterministic reset with a seed,
random-action rollout, and zero-action rollout, before touching PPO.

Run: python scripts/smoke_test_env.py
"""

from __future__ import annotations

import numpy as np
from gymnasium.utils.env_checker import check_env

from config.loader import load_config
from envs.g1_walk_env import G1WalkEnv


def main() -> None:
    cfg = load_config()
    env = G1WalkEnv(cfg)

    print(f"observation_space={env.observation_space}, action_space={env.action_space}")
    print(f"episode_control_steps={cfg.sim.episode_control_steps} (control_dt={cfg.sim.control_dt}s)")

    print("\n-- gymnasium check_env --")
    check_env(env.unwrapped, skip_render_check=True)
    print("check_env passed")

    print("\n-- deterministic reset with seed --")
    obs1, info1 = env.reset(seed=42)
    obs2, info2 = env.reset(seed=42)
    assert np.allclose(obs1, obs2), "reset(seed=42) is not repeatable"
    assert np.allclose(info1["command"], info2["command"]), "command sampling not repeatable with seed"
    print(f"repeatable: obs match={np.allclose(obs1, obs2)}, command match={np.allclose(info1['command'], info2['command'])}")
    assert obs1.shape == env.observation_space.shape
    assert obs1.dtype == np.float32

    print("\n-- zero-action rollout (expect fall around ~1.3s per PD-hold smoke test) --")
    env.reset(seed=0)
    zero_action = np.zeros(12, dtype=np.float32)
    steps = 0
    terminated = truncated = False
    total_reward = 0.0
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(zero_action)
        assert np.all(np.isfinite(obs)), f"non-finite obs at step {steps}"
        assert env.observation_space.contains(obs), f"obs out of declared space at step {steps}"
        total_reward += reward
        steps += 1
        if steps > cfg.sim.episode_control_steps:
            raise RuntimeError("zero-action rollout did not terminate/truncate in time")
    print(f"zero-action episode: steps={steps} ({steps * cfg.sim.control_dt:.2f}s), "
          f"terminated={terminated}, truncated={truncated}, reason={info['termination_reason']}, "
          f"total_reward={total_reward:.2f}")

    print("\n-- random-action rollout --")
    env.reset(seed=1)
    rng = np.random.default_rng(1)
    steps = 0
    terminated = truncated = False
    total_reward = 0.0
    while not (terminated or truncated):
        action = rng.uniform(-1.0, 1.0, size=12).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.all(np.isfinite(obs)), f"non-finite obs at step {steps}"
        total_reward += reward
        steps += 1
        if steps > cfg.sim.episode_control_steps:
            raise RuntimeError("random-action rollout did not terminate/truncate in time")
    print(f"random-action episode: steps={steps} ({steps * cfg.sim.control_dt:.2f}s), "
          f"terminated={terminated}, truncated={truncated}, reason={info['termination_reason']}, "
          f"total_reward={total_reward:.2f}")

    print("\n-- full-length rollout under a hand-tuned 'stand still' command (vx=vy=0) --")
    env.reset(seed=2)
    env._command = np.zeros(2, dtype=np.float32)  # override sampled command for this check
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(zero_action)
        steps += 1
        if steps > cfg.sim.episode_control_steps:
            break
    print(f"steps={steps}, terminated={terminated}, truncated={truncated}")

    env.close()
    print("\nENV CONTRACT SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
