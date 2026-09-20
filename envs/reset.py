"""Reset behavior. No domain randomization yet (challenge Phase 8 -- add
only after basic walking works, per CLAUDE.md). Reset is deterministic
given a seed: same keyframe base pose + default leg pose, zero velocity.
"""

from __future__ import annotations

import mujoco
import numpy as np

from control.joint_map import JointMap


def reset_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_map: JointMap,
    q_default: np.ndarray,
    stand_key_id: int,
) -> None:
    mujoco.mj_resetDataKeyframe(model, data, stand_key_id)
    data.qpos[joint_map.qpos_adr] = q_default
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def sample_command(rng: np.random.Generator, vx_range: tuple[float, float], vy_range: tuple[float, float]) -> np.ndarray:
    vx = rng.uniform(*vx_range)
    vy = rng.uniform(*vy_range)
    return np.array([vx, vy], dtype=np.float32)
