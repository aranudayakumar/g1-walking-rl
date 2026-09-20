"""action -> target joint position -> PD torque, in canonical joint order.

Pure numpy, independent of MuJoCo, so it is unit-testable without a
simulator. This is the control mapping named in the challenge doc
("target joint positions as outputs"): the policy outputs a normalized
delta from a default pose, and a PD law converts that position target
into the torque actually applied to the (now direct-torque) leg motors.
"""

from __future__ import annotations

import numpy as np

from control.joint_map import JOINT_GROUPS

# Starting PD gains, one (kp, kd) pair per leg-joint group, applied to
# both legs. Not yet tuned for walking -- these are cited as an informed
# starting point from a sibling project's validated standing-task gains
# for the identical robot/actuator setup (docs/RESEARCH_LOG.md R-0003),
# to avoid guessing blind. Revisit if PD-hold or walking experiments show
# they're wrong for this task (docs/FAILURE_ANALYSIS.md).
DEFAULT_KP_BY_GROUP: dict[str, float] = {
    "hip_pitch": 100.0,
    "hip_roll": 100.0,
    "hip_yaw": 100.0,
    "knee": 150.0,
    "ankle_pitch": 40.0,
    "ankle_roll": 40.0,
}
DEFAULT_KD_BY_GROUP: dict[str, float] = {
    "hip_pitch": 2.0,
    "hip_roll": 2.0,
    "hip_yaw": 2.0,
    "knee": 4.0,
    "ankle_pitch": 2.0,
    "ankle_roll": 2.0,
}


def build_gains(
    kp_by_group: dict[str, float] | None = None,
    kd_by_group: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (kp, kd) arrays of shape (12,) in CANONICAL_ORDER (left block
    then right block, each in JOINT_GROUPS order)."""
    kp_by_group = kp_by_group or DEFAULT_KP_BY_GROUP
    kd_by_group = kd_by_group or DEFAULT_KD_BY_GROUP
    kp = np.array([kp_by_group[g] for g in JOINT_GROUPS] * 2, dtype=np.float64)
    kd = np.array([kd_by_group[g] for g in JOINT_GROUPS] * 2, dtype=np.float64)
    return kp, kd


def gains_from_config(pd_gains_cfg) -> tuple[np.ndarray, np.ndarray]:
    """Adapt a config.loader.PDGainsConfig (per-group kp/kd dataclasses)
    into the (kp, kd) arrays build_gains expects."""
    kp_by_group = {g: getattr(pd_gains_cfg, g).kp for g in JOINT_GROUPS}
    kd_by_group = {g: getattr(pd_gains_cfg, g).kd for g in JOINT_GROUPS}
    return build_gains(kp_by_group, kd_by_group)


def action_to_target(
    action: np.ndarray,
    q_default: np.ndarray,
    action_scale: float,
    joint_range: np.ndarray,
) -> np.ndarray:
    """target_q = q_default + action_scale * clip(action, -1, 1), clipped to joint limits.

    action is assumed already in [-1, 1] (the policy's action space); we
    clip defensively in case of numerical overshoot.
    """
    clipped_action = np.clip(action, -1.0, 1.0)
    target = q_default + action_scale * clipped_action
    return np.clip(target, joint_range[:, 0], joint_range[:, 1])


def pd_torque(
    q_target: np.ndarray,
    q: np.ndarray,
    qd: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
    forcerange: np.ndarray | None = None,
) -> np.ndarray:
    """torque = kp * (q_target - q) - kd * qd, optionally clipped to forcerange."""
    torque = kp * (q_target - q) - kd * qd
    if forcerange is not None:
        torque = np.clip(torque, forcerange[:, 0], forcerange[:, 1])
    return torque
