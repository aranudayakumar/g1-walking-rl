"""Foot ground-contact detection, shared by the environment (for the
air-time reward, F-0004) and diagnostics (diagnostics/gait_diagnostic.py).

Geom ids found by inspecting the model directly (control/joint_map-style
verification, not assumed): each foot has 4 corner collision spheres.
Verified in diagnostics/gait_diagnostic.py against assets/g1/g1_walk.xml.
"""

from __future__ import annotations

import mujoco
import numpy as np

LEFT_FOOT_GEOMS = (15, 16, 17, 18)
RIGHT_FOOT_GEOMS = (30, 31, 32, 33)


def verify_foot_geoms(model: mujoco.MjModel) -> None:
    """Assert the hardcoded geom ids above actually belong to the
    left/right ankle_roll (foot) bodies in this model -- never trust a
    hardcoded index without checking it against live model data."""
    left_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    right_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
    for g in LEFT_FOOT_GEOMS:
        if model.geom_bodyid[g] != left_body:
            raise ValueError(f"geom {g} does not belong to left_ankle_roll_link in this model")
    for g in RIGHT_FOOT_GEOMS:
        if model.geom_bodyid[g] != right_body:
            raise ValueError(f"geom {g} does not belong to right_ankle_roll_link in this model")


def foot_in_contact(data: mujoco.MjData, foot_geoms: tuple[int, ...], floor_geom: int) -> bool:
    for i in range(data.ncon):
        c = data.contact[i]
        if (c.geom1 == floor_geom and c.geom2 in foot_geoms) or (c.geom2 == floor_geom and c.geom1 in foot_geoms):
            return True
    return False


def both_feet_contact(model: mujoco.MjModel, data: mujoco.MjData, floor_geom: int) -> np.ndarray:
    """Return [left_in_contact, right_in_contact] as a (2,) bool array."""
    return np.array(
        [
            foot_in_contact(data, LEFT_FOOT_GEOMS, floor_geom),
            foot_in_contact(data, RIGHT_FOOT_GEOMS, floor_geom),
        ]
    )


class AirTimeTracker:
    """Per-foot swing-phase (air time) bookkeeping for the feet_air_time
    reward (docs/FAILURE_ANALYSIS.md F-0004): rewards a longer swing phase
    at the moment of touchdown, directly targeting the chattery,
    ~20-30ms-swing stepping pattern measured in the canonical policy
    (median 20ms, 90th percentile 40ms -- an order of magnitude below a
    deliberate stride) without needing to guess at foot-height thresholds
    that would depend on the robot's exact geometry.

    Returns raw (unweighted) reward at each step -- the env multiplies by
    the configured weight, keeping this class a pure state machine.
    """

    def __init__(self) -> None:
        self.air_time = np.zeros(2)
        self.prev_contact = np.array([True, True])

    def reset(self, initial_contact: np.ndarray) -> None:
        self.air_time[:] = 0.0
        self.prev_contact = initial_contact.copy()

    def step(self, contact: np.ndarray, control_dt: float, threshold_s: float) -> float:
        """Advance by one control step given the new contact state; return
        the raw (pre-weight) reward contribution for this step."""
        raw_reward = 0.0
        for i in range(2):
            if not contact[i]:
                self.air_time[i] += control_dt
            else:
                if not self.prev_contact[i] and self.air_time[i] > 0:
                    raw_reward += self.air_time[i] - threshold_s
                self.air_time[i] = 0.0
        self.prev_contact = contact.copy()
        return raw_reward


class GroundTimeTracker:
    """Per-foot continuous-stance bookkeeping for the stance_overrun_penalty
    (docs/FAILURE_ANALYSIS.md F-0004): a foot that stays in contact beyond
    `max_stance_s` accrues a growing penalty *every step* it continues to
    overstay -- unlike AirTimeTracker's touchdown-only reward, this cannot
    be dodged by never lifting a foot, which is exactly the exploit F-0004
    found (a foot that never leaves the ground never triggers a
    touchdown-only reward, so "never step" was a zero-cost escape from it).
    """

    def __init__(self) -> None:
        self.stance_time = np.zeros(2)

    def reset(self, initial_contact: np.ndarray) -> None:
        self.stance_time[:] = 0.0
        self.stance_time[initial_contact] = 1e-6  # avoid a spurious overrun signal exactly at t=0

    def step(self, contact: np.ndarray, control_dt: float, max_stance_s: float) -> float:
        raw_penalty = 0.0
        for i in range(2):
            if contact[i]:
                self.stance_time[i] += control_dt
                if self.stance_time[i] > max_stance_s:
                    raw_penalty += self.stance_time[i] - max_stance_s
            else:
                self.stance_time[i] = 0.0
        return raw_penalty
