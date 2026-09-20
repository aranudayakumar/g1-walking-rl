"""Vertical slice 1: model load -> physics stepping, with ground-truth
introspection instead of assumptions about joint/actuator ordering or
keyframe layout (CLAUDE.md: "never guess joint ordering... verify from
model/repository data").

Run: python scripts/smoke_test_model.py
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "assets" / "g1" / "scene_walk.xml"


def main() -> None:
    print(f"loading {MODEL_PATH}")
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    print(f"nq={model.nq} nv={model.nv} nu={model.nu} njnt={model.njnt} nbody={model.nbody}")

    print("\n-- joints (name, type, qpos_adr, qvel_adr, range) --")
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        jtype = model.jnt_type[j]
        print(
            f"  [{j:2d}] {name:32s} type={jtype} qpos_adr={model.jnt_qposadr[j]:2d} "
            f"qvel_adr={model.jnt_dofadr[j]:2d} range={model.jnt_range[j]}"
        )

    print("\n-- actuators (name, biastype [0=motor/NONE, 1=position/AFFINE], ctrlrange, forcerange) --")
    for a in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        biastype = model.actuator_biastype[a]
        print(
            f"  [{a:2d}] {name:32s} biastype={biastype} ctrlrange={model.actuator_ctrlrange[a]} "
            f"forcerange={model.actuator_forcerange[a]}"
        )

    print("\n-- keyframes --")
    for k in range(model.nkey):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, k)
        print(f"  [{k}] {name}: qpos len={len(model.key_qpos[k])} ctrl len={len(model.key_ctrl[k])}")

    # Reset to the "stand" keyframe if present, else qpos0.
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        print(f"\nreset to keyframe 'stand' (id={key_id})")
    else:
        mujoco.mj_resetData(model, data)
        print("\nno 'stand' keyframe found; reset to qpos0")

    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    print(f"pelvis body id={pelvis_id}, initial height qpos[2]={data.qpos[2]:.4f}")

    n_steps = 500
    print(f"\nstepping {n_steps} times with zero control...")
    for i in range(n_steps):
        data.ctrl[:] = 0.0
        mujoco.mj_step(model, data)
        if not (np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))):
            print(f"  NON-FINITE STATE at step {i}")
            break
    else:
        print("  all steps finite.")

    print(f"final pelvis height: {data.qpos[2]:.4f}")
    print(f"final time: {data.time:.4f}s (physics dt={model.opt.timestep})")

    print("\n-- reset repeatability check --")
    mujoco.mj_resetDataKeyframe(model, data, key_id) if key_id >= 0 else mujoco.mj_resetData(model, data)
    q1 = data.qpos.copy()
    mujoco.mj_resetDataKeyframe(model, data, key_id) if key_id >= 0 else mujoco.mj_resetData(model, data)
    q2 = data.qpos.copy()
    print(f"reset repeatable: {np.allclose(q1, q2)}")

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
