import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from config.loader import load_config
from envs.g1_walk_env import G1WalkEnv
from envs.observations import quat_to_yaw


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture()
def env(cfg):
    e = G1WalkEnv(cfg)
    yield e
    e.close()


def test_gymnasium_check_env(env):
    check_env(env.unwrapped, skip_render_check=True)


def test_reset_is_deterministic_with_seed(env):
    obs1, info1 = env.reset(seed=42)
    obs2, info2 = env.reset(seed=42)
    assert np.allclose(obs1, obs2)
    assert np.allclose(info1["command"], info2["command"])


def test_reset_observation_matches_declared_space(env, cfg):
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)


def test_zero_action_rollout_terminates_and_stays_finite(env):
    env.reset(seed=0)
    zero_action = np.zeros(12, dtype=np.float32)
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated) and steps < 200:
        obs, reward, terminated, truncated, info = env.step(zero_action)
        assert np.all(np.isfinite(obs))
        assert np.isfinite(reward)
        steps += 1
    assert terminated  # expect a fall well before the 20s truncation limit
    assert info["termination_reason"] in ("fell_height", "fell_tilt")


def test_yaw_at_reset_matches_pelvis_heading_immediately_after_reset(env):
    env.reset(seed=0)
    actual_yaw = quat_to_yaw(env.data.xquat[env.pelvis_id])
    assert np.isclose(env._yaw_at_reset, actual_yaw)


def test_heading_deviation_near_zero_immediately_after_reset(env):
    """The robot starts upright with no rotation, so heading deviation
    should be ~0 on the very first step -- a basic sanity check that the
    heading-deviation mechanism (used by heading_deviation_penalty,
    F-0004/ADR-009) is wired to the correct reference, not some stale or
    uninitialized value."""
    env.reset(seed=0)
    zero_action = np.zeros(12, dtype=np.float32)
    env.step(zero_action)
    current_yaw = quat_to_yaw(env.data.xquat[env.pelvis_id])
    deviation = current_yaw - env._yaw_at_reset
    assert abs(deviation) < 0.05  # radians, ~3 degrees -- essentially no rotation in one step


def test_random_action_rollout_stays_finite(env):
    env.reset(seed=1)
    rng = np.random.default_rng(1)
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated) and steps < 200:
        action = rng.uniform(-1.0, 1.0, size=12).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.all(np.isfinite(obs))
        steps += 1
    assert terminated or truncated
