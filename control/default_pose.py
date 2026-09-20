"""Default leg pose q_default, in CANONICAL_ORDER (control/joint_map.py).

This is the pose the policy's action is a delta from (control/pd_controller
.action_to_target), and the pose used at reset. A slight knee bend is a
standard walking-ready stance (also used in the sibling standing-rl
project for the identical robot, docs/RESEARCH_LOG.md R-0003) rather than
the fully-straight upstream keyframe pose (all leg joints at 0), which is
a valid but less dynamically useful starting point for locomotion.
"""

from __future__ import annotations

import numpy as np

from control.joint_map import JOINT_GROUPS

_LEG_DEFAULT_BY_GROUP: dict[str, float] = {
    "hip_pitch": -0.10,
    "hip_roll": 0.00,
    "hip_yaw": 0.00,
    "knee": 0.30,
    "ankle_pitch": -0.20,
    "ankle_roll": 0.00,
}


def build_q_default() -> np.ndarray:
    """Return the (12,) default leg joint positions in CANONICAL_ORDER
    (left block then right block, mirrored -- both legs use the same
    per-group targets)."""
    values = [_LEG_DEFAULT_BY_GROUP[g] for g in JOINT_GROUPS]
    return np.array(values + values, dtype=np.float64)


# Upper body (waist + arms) is held fixed at the upstream "stand" keyframe
# pose and is not RL-controlled in this project (docs/DECISIONS.md).
UPPER_BODY_ORDER: tuple[str, ...] = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

# Values straight from the upstream g1.xml <keyframe name="stand"> ctrl
# array (verified in scripts/smoke_test_model.py), for the same 17 joints
# in UPPER_BODY_ORDER.
UPPER_BODY_TARGETS: np.ndarray = np.array(
    [
        0.0, 0.0, 0.0,  # waist yaw, roll, pitch
        0.2, 0.2, 0.0, 1.28, 0.0, 0.0, 0.0,  # left arm: shoulder x3, elbow, wrist x3
        0.2, -0.2, 0.0, 1.28, 0.0, 0.0, 0.0,  # right arm
    ],
    dtype=np.float64,
)
