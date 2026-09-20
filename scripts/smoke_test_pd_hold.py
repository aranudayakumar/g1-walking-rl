"""Vertical slice 2: control mapping. Verify the PD controller computes
sane torques and the simulation stays numerically well-behaved when
holding the default leg pose (zero policy action), before building the
full Gymnasium environment on top of it.

This does NOT test whether the robot can stand indefinitely with zero
action -- a free-standing biped with pure joint-space PD and no active
balance feedback is not expected to be self-stabilizing (that is what
the RL policy has to learn; see docs/SIMULATION.md "PD-hold baseline").
It tests: no NaNs, no immediate instability, no torque saturation, and a
gradual (not violent) fall, which is what we need before trusting the
control mapping inside the full environment.

Run: python scripts/smoke_test_pd_hold.py
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from control.default_pose import UPPER_BODY_ORDER, UPPER_BODY_TARGETS, build_q_default
from control.joint_map import build_joint_map
from control.pd_controller import action_to_target, build_gains, pd_torque

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "assets" / "g1" / "scene_walk.xml"

PHYSICS_DT = 0.002
CONTROL_DECIMATION = 10  # -> 50 Hz control, matches config.yaml
EPISODE_SECONDS = 5.0


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    assert abs(model.opt.timestep - PHYSICS_DT) < 1e-9, "physics dt mismatch vs config"

    joint_map = build_joint_map(model)
    q_default = build_q_default()
    kp, kd = build_gains()
    upper_body_actuator_ids = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in UPPER_BODY_ORDER]
    )
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    # Start from the default leg pose (slight crouch), not the fully
    # straight-leg keyframe, since that's the pose the controller targets.
    data.qpos[joint_map.qpos_adr] = q_default

    n_control_steps = int(EPISODE_SECONDS / (PHYSICS_DT * CONTROL_DECIMATION))
    prev_action = np.zeros(12)

    heights = []
    max_torque_frac = 0.0
    fell_below_term_height_at: float | None = None
    TERM_HEIGHT = 0.5  # candidate termination threshold, see docs/SIMULATION.md
    for step in range(n_control_steps):
        action = np.zeros(12)  # zero action -> target == q_default
        target = action_to_target(action, q_default, action_scale=0.25, joint_range=joint_map.joint_range)

        for _ in range(CONTROL_DECIMATION):
            q = data.qpos[joint_map.qpos_adr]
            qd = data.qvel[joint_map.qvel_adr]
            torque = pd_torque(target, q, qd, kp, kd, forcerange=joint_map.actuator_forcerange)
            data.ctrl[joint_map.actuator_ids] = torque
            data.ctrl[upper_body_actuator_ids] = UPPER_BODY_TARGETS
            mujoco.mj_step(model, data)

            frac = np.max(np.abs(torque) / joint_map.actuator_forcerange[:, 1])
            max_torque_frac = max(max_torque_frac, float(frac))

        if not (np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))):
            print(f"NON-FINITE STATE at control step {step}")
            break

        pelvis_height = float(data.xpos[pelvis_id][2])
        heights.append(pelvis_height)
        if fell_below_term_height_at is None and pelvis_height < TERM_HEIGHT:
            fell_below_term_height_at = step * PHYSICS_DT * CONTROL_DECIMATION
        prev_action = action

    heights = np.array(heights)
    print(f"steps completed: {len(heights)}/{n_control_steps} (all finite)")
    print(f"pelvis height: start={heights[0]:.4f} end={heights[-1]:.4f} min={heights.min():.4f} max={heights.max():.4f}")
    print(f"max |torque|/limit fraction observed: {max_torque_frac:.3f} (1.0 = saturated)")
    if fell_below_term_height_at is not None:
        print(f"pelvis height crossed {TERM_HEIGHT}m at t={fell_below_term_height_at:.2f}s "
              f"(expected: pure PD with no balance feedback is not self-stabilizing; "
              f"see docs/SIMULATION.md)")
    else:
        print(f"pelvis height never dropped below {TERM_HEIGHT}m in {EPISODE_SECONDS}s")

    # Pass criteria: numerically stable, no torque saturation, and the fall
    # (if any) is gradual rather than an instantaneous blow-up in the first
    # control step -- NOT "stands forever", which pure PD is not expected to do.
    no_saturation = max_torque_frac < 1.0
    gradual = len(heights) < 5 or heights[4] > 0.6  # still near-standing after 0.4s
    all_finite = len(heights) == n_control_steps
    ok = no_saturation and gradual and all_finite
    print("\nPD CONTROL-MAPPING SMOKE TEST " + ("PASSED" if ok else "FAILED"))


if __name__ == "__main__":
    main()
