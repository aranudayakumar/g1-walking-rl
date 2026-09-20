import numpy as np

from config.loader import RewardWeightsConfig
from rewards.walking_reward import compute_reward

WEIGHTS = RewardWeightsConfig(
    tracking_lin_vel=1.0, tracking_sigma=0.25, alive_bonus=0.5, torque_penalty=0.0002
)
WEIGHTS_WITH_GAIT_QUALITY = RewardWeightsConfig(
    tracking_lin_vel=1.0,
    tracking_sigma=0.25,
    alive_bonus=0.5,
    torque_penalty=0.0002,
    ang_vel_penalty=1.0,
    action_rate_penalty=0.01,
    feet_air_time_bonus=3.0,
    feet_air_time_threshold_s=0.2,
    stance_overrun_penalty=2.0,
    max_stance_s=1.0,
    heading_deviation_penalty=0.5,
)

ZERO_ACTION = np.zeros(12)


def reward(weights=WEIGHTS, **overrides):
    kwargs = dict(
        base_lin_vel_xy=np.array([0.1, 0.0]),
        command_vel_xy=np.array([0.3, 0.0]),
        torque=np.zeros(12),
        base_ang_vel_z=0.0,
        action=ZERO_ACTION,
        prev_action=ZERO_ACTION,
        air_time_reward_raw=0.0,
        stance_overrun_raw=0.0,
        heading_deviation_rad=0.0,
    )
    kwargs.update(overrides)
    return compute_reward(weights=weights, **kwargs)


def test_reward_finite_for_typical_state():
    total, info = reward(
        weights=WEIGHTS_WITH_GAIT_QUALITY,
        base_lin_vel_xy=np.array([0.1, 0.0]),
        torque=np.full(12, 20.0),
        base_ang_vel_z=0.1,
    )
    assert np.isfinite(total)
    assert all(np.isfinite(v) for v in info.values())


def test_perfect_tracking_maximizes_tracking_component():
    reward_perfect, info_perfect = reward(base_lin_vel_xy=np.array([0.3, 0.0]))
    reward_off, info_off = reward(base_lin_vel_xy=np.array([0.0, 0.0]))
    assert info_perfect["tracking_lin_vel"] == WEIGHTS.tracking_lin_vel  # exp(0) == 1
    assert info_perfect["tracking_lin_vel"] > info_off["tracking_lin_vel"]
    assert reward_perfect > reward_off


def test_torque_penalty_is_never_positive():
    _, info = reward(base_lin_vel_xy=np.array([0.0, 0.0]), command_vel_xy=np.array([0.0, 0.0]), torque=np.full(12, 50.0))
    assert info["torque_penalty"] <= 0.0


def test_ang_vel_penalty_zero_when_not_rotating():
    _, info = reward(weights=WEIGHTS_WITH_GAIT_QUALITY, base_lin_vel_xy=np.array([0.3, 0.0]), base_ang_vel_z=0.0)
    assert info["ang_vel_penalty"] == 0.0


def test_ang_vel_penalty_grows_with_yaw_rate_and_is_never_positive():
    _, info_slow = reward(weights=WEIGHTS_WITH_GAIT_QUALITY, base_lin_vel_xy=np.array([0.3, 0.0]), base_ang_vel_z=0.1)
    _, info_fast = reward(weights=WEIGHTS_WITH_GAIT_QUALITY, base_lin_vel_xy=np.array([0.3, 0.0]), base_ang_vel_z=1.0)
    assert info_slow["ang_vel_penalty"] <= 0.0
    assert info_fast["ang_vel_penalty"] < info_slow["ang_vel_penalty"]  # more negative


def test_ang_vel_penalty_symmetric_in_rotation_direction():
    _, info_pos = reward(weights=WEIGHTS_WITH_GAIT_QUALITY, base_lin_vel_xy=np.array([0.3, 0.0]), base_ang_vel_z=0.5)
    _, info_neg = reward(weights=WEIGHTS_WITH_GAIT_QUALITY, base_lin_vel_xy=np.array([0.3, 0.0]), base_ang_vel_z=-0.5)
    assert info_pos["ang_vel_penalty"] == info_neg["ang_vel_penalty"]


def test_action_rate_penalty_zero_when_action_unchanged():
    action = np.linspace(-0.5, 0.5, 12)
    _, info = reward(weights=WEIGHTS_WITH_GAIT_QUALITY, base_lin_vel_xy=np.array([0.3, 0.0]), action=action, prev_action=action)
    assert info["action_rate_penalty"] == 0.0


def test_action_rate_penalty_penalizes_large_jumps():
    _, info_small = reward(weights=WEIGHTS_WITH_GAIT_QUALITY, base_lin_vel_xy=np.array([0.3, 0.0]), action=np.full(12, 0.1))
    _, info_large = reward(weights=WEIGHTS_WITH_GAIT_QUALITY, base_lin_vel_xy=np.array([0.3, 0.0]), action=np.full(12, 1.0))
    assert info_large["action_rate_penalty"] < info_small["action_rate_penalty"] < 0.0


def test_air_time_bonus_zero_when_raw_is_zero():
    _, info = reward(weights=WEIGHTS_WITH_GAIT_QUALITY, base_lin_vel_xy=np.array([0.3, 0.0]), air_time_reward_raw=0.0)
    assert info["feet_air_time_bonus"] == 0.0


def test_air_time_bonus_scales_linearly_with_raw_value():
    _, info_neg = reward(weights=WEIGHTS_WITH_GAIT_QUALITY, base_lin_vel_xy=np.array([0.3, 0.0]), air_time_reward_raw=-0.1)
    _, info_pos = reward(weights=WEIGHTS_WITH_GAIT_QUALITY, base_lin_vel_xy=np.array([0.3, 0.0]), air_time_reward_raw=0.1)
    assert info_neg["feet_air_time_bonus"] < 0.0 < info_pos["feet_air_time_bonus"]
    assert np.isclose(info_pos["feet_air_time_bonus"], -info_neg["feet_air_time_bonus"])
    assert np.isclose(info_pos["feet_air_time_bonus"], WEIGHTS_WITH_GAIT_QUALITY.feet_air_time_bonus * 0.1)


def test_stance_overrun_penalty_zero_when_raw_is_zero():
    _, info = reward(weights=WEIGHTS_WITH_GAIT_QUALITY, base_lin_vel_xy=np.array([0.3, 0.0]), stance_overrun_raw=0.0)
    assert info["stance_overrun_penalty"] == 0.0


def test_stance_overrun_penalty_never_positive_and_scales_with_raw_value():
    _, info_small = reward(weights=WEIGHTS_WITH_GAIT_QUALITY, base_lin_vel_xy=np.array([0.3, 0.0]), stance_overrun_raw=0.1)
    _, info_large = reward(weights=WEIGHTS_WITH_GAIT_QUALITY, base_lin_vel_xy=np.array([0.3, 0.0]), stance_overrun_raw=1.0)
    assert info_small["stance_overrun_penalty"] <= 0.0
    assert info_large["stance_overrun_penalty"] < info_small["stance_overrun_penalty"]


def test_heading_deviation_penalty_zero_when_on_original_heading():
    _, info = reward(weights=WEIGHTS_WITH_GAIT_QUALITY, base_lin_vel_xy=np.array([0.3, 0.0]), heading_deviation_rad=0.0)
    assert info["heading_deviation_penalty"] == 0.0


def test_heading_deviation_penalty_grows_with_deviation_and_is_never_positive():
    _, info_small = reward(weights=WEIGHTS_WITH_GAIT_QUALITY, base_lin_vel_xy=np.array([0.3, 0.0]), heading_deviation_rad=0.1)
    _, info_large = reward(weights=WEIGHTS_WITH_GAIT_QUALITY, base_lin_vel_xy=np.array([0.3, 0.0]), heading_deviation_rad=0.5)
    assert info_small["heading_deviation_penalty"] <= 0.0
    assert info_large["heading_deviation_penalty"] < info_small["heading_deviation_penalty"]


def test_heading_deviation_penalty_symmetric_in_direction():
    _, info_pos = reward(weights=WEIGHTS_WITH_GAIT_QUALITY, base_lin_vel_xy=np.array([0.3, 0.0]), heading_deviation_rad=0.3)
    _, info_neg = reward(weights=WEIGHTS_WITH_GAIT_QUALITY, base_lin_vel_xy=np.array([0.3, 0.0]), heading_deviation_rad=-0.3)
    assert info_pos["heading_deviation_penalty"] == info_neg["heading_deviation_penalty"]


def test_default_gait_quality_weights_are_zero_for_backward_compatibility():
    """A config predating F-0003/F-0004/heading-deviation (no gait-quality
    keys) must still load and behave exactly as before -- see config/loader.py."""
    assert WEIGHTS.ang_vel_penalty == 0.0
    assert WEIGHTS.action_rate_penalty == 0.0
    assert WEIGHTS.feet_air_time_bonus == 0.0
    assert WEIGHTS.stance_overrun_penalty == 0.0
    assert WEIGHTS.heading_deviation_penalty == 0.0
    _, info = reward(
        weights=WEIGHTS, base_lin_vel_xy=np.array([0.3, 0.0]), base_ang_vel_z=2.0,
        action=np.full(12, 1.0), prev_action=np.full(12, -1.0),
        air_time_reward_raw=5.0, stance_overrun_raw=5.0, heading_deviation_rad=1.0,
    )
    assert info["ang_vel_penalty"] == 0.0
    assert info["action_rate_penalty"] == 0.0
    assert info["feet_air_time_bonus"] == 0.0
    assert info["stance_overrun_penalty"] == 0.0
    assert info["heading_deviation_penalty"] == 0.0


def test_no_single_component_dominates_at_typical_scales():
    """Sanity check from PROJECT_PLAN.md Phase 4: no penalty should
    numerically dominate by accident at a plausible mid-episode state
    (moderate tracking error, moderate torque usage, modest yaw rate,
    action change, and air-time shortfall)."""
    _, info = reward(
        weights=WEIGHTS_WITH_GAIT_QUALITY,
        base_lin_vel_xy=np.array([0.15, 0.02]),
        torque=np.full(12, 30.0),  # ~30 N*m, well under most joint limits
        base_ang_vel_z=0.2,
        action=np.full(12, 0.1),
        prev_action=np.full(12, 0.05),
        air_time_reward_raw=-0.05,  # a typical single-foot touchdown shortfall
        stance_overrun_raw=0.03,  # a typical brief stance-overrun sample
        heading_deviation_rad=0.2,  # a modest mid-episode heading deviation
    )
    components = {k: abs(v) for k, v in info.items() if k not in ("total", "raw_vel_error")}
    max_component = max(components.values())
    total_magnitude = sum(components.values())
    assert max_component / total_magnitude < 0.95, f"one component dominates: {info}"
