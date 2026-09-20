"""Observation construction.

Per the challenge doc, the observation is exactly: joint positions,
joint velocities, base angular velocity, projected gravity, commanded
velocity, and previous action. Base linear velocity is deliberately NOT
observed (it is hard to estimate on real hardware without a state
estimator -- standard practice in the sim2real literature we were
pointed at, e.g. Berkeley Humanoid), even though it is used internally to
compute the reward (privileged information at training time is fine; the
policy itself never sees it).

All quantities are in the pelvis (base) body frame. See docs/RL_PIPELINE
.md for the full field-by-field spec (meaning/units/frame/source).
"""

from __future__ import annotations

import mujoco
import numpy as np

from control.joint_map import JointMap

N_LEG_JOINTS = 12
N_COMMAND = 2  # vx, vy
OBS_DIM = N_LEG_JOINTS + N_LEG_JOINTS + 3 + 3 + N_COMMAND + N_LEG_JOINTS  # 44

_WORLD_GRAVITY_DIR = np.array([0.0, 0.0, -1.0])


def projected_gravity(model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> np.ndarray:
    """Unit gravity direction expressed in `body_id`'s local frame."""
    r_wb = data.xmat[body_id].reshape(3, 3)  # body -> world rotation
    return r_wb.T @ _WORLD_GRAVITY_DIR


def quat_to_yaw(quat_wxyz: np.ndarray) -> float:
    """World-frame yaw (heading) angle from a body's (w,x,y,z) quaternion."""
    w, x, y, z = quat_wxyz
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def base_frame_velocity(
    model: mujoco.MjModel, data: mujoco.MjData, body_id: int
) -> tuple[np.ndarray, np.ndarray]:
    """(linear_vel, angular_vel), both (3,), expressed in `body_id`'s own
    local frame (mj_objectVelocity with flg_local=True)."""
    res = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, res, 1)
    ang_vel, lin_vel = res[:3], res[3:]
    return lin_vel, ang_vel


def build_observation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_map: JointMap,
    q_default: np.ndarray,
    pelvis_body_id: int,
    command_vel_xy: np.ndarray,
    prev_action: np.ndarray,
) -> np.ndarray:
    q = data.qpos[joint_map.qpos_adr] - q_default
    qd = data.qvel[joint_map.qvel_adr]
    _, ang_vel = base_frame_velocity(model, data, pelvis_body_id)
    grav = projected_gravity(model, data, pelvis_body_id)

    obs = np.concatenate(
        [q, qd, ang_vel, grav, command_vel_xy, prev_action]
    ).astype(np.float32)
    assert obs.shape == (OBS_DIM,), f"observation shape mismatch: {obs.shape} != ({OBS_DIM},)"
    return obs
