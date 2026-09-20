"""Minimal reward per the challenge doc: velocity tracking + survival +
torque penalty. Every component is returned separately (reward_info) so a
single dominating term is visible immediately (docs/FAILURE_ANALYSIS.md
rule: consider setup bugs / reward domination before adding new terms).

Two additional terms (ang_vel_penalty, action_rate_penalty) were added
after F-0003 diagnosed two specific, measured gait-quality failures in
the canonical policy: it doesn't walk straight (yaw drifted 95 degrees
over 10s under a pure-forward command, because tracking_lin_vel is
computed in the robot's own local frame and nothing penalizes rotation)
and it takes tiny, chattery steps rather than clean strides (no term
rewards smooth, low-frequency motion). See docs/FAILURE_ANALYSIS.md F-0003
and docs/DECISIONS.md for the reasoning.

A fifth term (feet_air_time_bonus) was added after F-0004 measured that
action_rate_penalty alone only marginally improved step chatter (median
swing duration still ~20ms, an order of magnitude below a deliberate
stride) -- it rewards a longer swing phase at the moment of touchdown
(envs/contacts.AirTimeTracker), directly targeting stride duration
instead of indirectly discouraging it via action smoothness.

feet_air_time_bonus alone was then found (F-0004) to be gameable: since
it only fires at touchdown, a policy that never lifts a foot at all never
triggers it, and "never step" turned out to be an easier local optimum
than "take longer steps" (the policy learned to skate, sliding its feet
along the ground at ~0.25 m/s while remaining in geometric contact). A
sixth term (stance_overrun_penalty) closes this loophole: it penalizes a
foot for staying in continuous contact beyond max_stance_s, accruing
*every step* it continues to overstay -- unlike a touchdown-only reward,
this cannot be dodged by avoiding touchdowns.

A seventh term (heading_deviation_penalty) was added after ADR-009
(EXP-0018) found that pushing feet_air_time_bonus harder for longer
strides measurably degraded heading control, and that ang_vel_penalty
(instantaneous rotation-rate squared) doesn't clearly track *net* drift:
a slow, steady turn and a larger but zero-mean oscillation can have
similar mean ang_vel_z^2, yet only the former accumulates into real
heading loss over an episode. This term penalizes deviation from the
heading at episode start directly, targeting drift specifically.

Do not add further terms without the same kind of named, measured
failure.
"""

from __future__ import annotations

import numpy as np

from config.loader import RewardWeightsConfig


def compute_reward(
    base_lin_vel_xy: np.ndarray,
    command_vel_xy: np.ndarray,
    torque: np.ndarray,
    base_ang_vel_z: float,
    action: np.ndarray,
    prev_action: np.ndarray,
    air_time_reward_raw: float,
    stance_overrun_raw: float,
    heading_deviation_rad: float,
    weights: RewardWeightsConfig,
) -> tuple[float, dict[str, float]]:
    vel_err = np.sum((command_vel_xy - base_lin_vel_xy) ** 2)
    tracking = float(np.exp(-vel_err / weights.tracking_sigma**2))
    alive = 1.0
    torque_cost = float(np.sum(torque**2))
    ang_vel_cost = float(base_ang_vel_z**2)
    action_rate_cost = float(np.sum((action - prev_action) ** 2))
    heading_deviation_cost = float(heading_deviation_rad**2)

    reward_info = {
        "tracking_lin_vel": weights.tracking_lin_vel * tracking,
        "alive_bonus": weights.alive_bonus * alive,
        "torque_penalty": -weights.torque_penalty * torque_cost,
        "ang_vel_penalty": -weights.ang_vel_penalty * ang_vel_cost,
        "action_rate_penalty": -weights.action_rate_penalty * action_rate_cost,
        "feet_air_time_bonus": weights.feet_air_time_bonus * air_time_reward_raw,
        "stance_overrun_penalty": -weights.stance_overrun_penalty * stance_overrun_raw,
        "heading_deviation_penalty": -weights.heading_deviation_penalty * heading_deviation_cost,
    }
    total = sum(reward_info.values())
    reward_info["total"] = total
    reward_info["raw_vel_error"] = float(np.sqrt(vel_err))
    return float(total), reward_info
