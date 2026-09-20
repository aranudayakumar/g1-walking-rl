import dataclasses

import pytest

from config.loader import RewardScheduleTerm, load_config
from envs.g1_walk_env import G1WalkEnv


@pytest.fixture(scope="module")
def base_cfg():
    return load_config("config/exp0012_air_time.yaml")


def test_config_with_no_reward_schedule_section_defaults_to_empty(base_cfg):
    assert base_cfg.reward_schedule == {}


def test_config_loads_reward_schedule_terms(tmp_path, base_cfg):
    import yaml

    raw = yaml.safe_load(open("config/exp0012_air_time.yaml"))
    raw["reward_schedule"] = {
        "feet_air_time_bonus": {"start": 0.0, "end": 6.0, "end_fraction": 0.4},
    }
    p = tmp_path / "scheduled.yaml"
    p.write_text(yaml.dump(raw))

    cfg = load_config(p)
    assert cfg.reward_schedule == {"feet_air_time_bonus": RewardScheduleTerm(start=0.0, end=6.0, end_fraction=0.4)}


def test_no_schedule_returns_static_weights_unchanged(base_cfg):
    env = G1WalkEnv(base_cfg)
    try:
        env.set_curriculum_progress(0.5)
        assert env._effective_reward_weights() is base_cfg.reward_weights
    finally:
        env.close()


def test_schedule_interpolates_linearly_and_clamps_at_end_fraction(base_cfg):
    scheduled_cfg = dataclasses.replace(
        base_cfg,
        reward_schedule={"feet_air_time_bonus": RewardScheduleTerm(start=0.0, end=6.0, end_fraction=0.5)},
    )
    env = G1WalkEnv(scheduled_cfg)
    try:
        env.set_curriculum_progress(0.0)
        assert env._effective_reward_weights().feet_air_time_bonus == pytest.approx(0.0)

        env.set_curriculum_progress(0.25)
        assert env._effective_reward_weights().feet_air_time_bonus == pytest.approx(3.0)

        env.set_curriculum_progress(0.5)
        assert env._effective_reward_weights().feet_air_time_bonus == pytest.approx(6.0)

        env.set_curriculum_progress(1.0)  # past end_fraction -- clamped, not extrapolated
        assert env._effective_reward_weights().feet_air_time_bonus == pytest.approx(6.0)
    finally:
        env.close()


def test_schedule_leaves_unscheduled_weights_untouched(base_cfg):
    scheduled_cfg = dataclasses.replace(
        base_cfg,
        reward_schedule={"feet_air_time_bonus": RewardScheduleTerm(start=0.0, end=6.0, end_fraction=0.5)},
    )
    env = G1WalkEnv(scheduled_cfg)
    try:
        env.set_curriculum_progress(0.25)
        weights = env._effective_reward_weights()
        assert weights.tracking_lin_vel == base_cfg.reward_weights.tracking_lin_vel
        assert weights.ang_vel_penalty == base_cfg.reward_weights.ang_vel_penalty
    finally:
        env.close()


def test_set_curriculum_progress_clips_to_unit_interval(base_cfg):
    env = G1WalkEnv(base_cfg)
    try:
        env.set_curriculum_progress(-0.5)
        assert env._curriculum_progress == 0.0
        env.set_curriculum_progress(1.5)
        assert env._curriculum_progress == 1.0
    finally:
        env.close()
