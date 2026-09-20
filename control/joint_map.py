"""Canonical 12-leg-joint ordering, resolved against a live MuJoCo model.

action[i] always refers to the joint at CANONICAL_ORDER[i]. Verified
against the model in scripts/smoke_test_model.py: joint ids 1-12 in
assets/g1/g1_walk.xml are exactly these 12 names in this order, with
qpos_adr 7-18 and qvel_adr 6-17 (floating base occupies qpos[0:7],
qvel[0:6]).
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

CANONICAL_ORDER: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)

# Joint-group name (order matches the 6 per-leg entries above); used to
# look up PD gains / default-pose targets from config, and shared by both
# legs (left block then right block).
JOINT_GROUPS: tuple[str, ...] = (
    "hip_pitch",
    "hip_roll",
    "hip_yaw",
    "knee",
    "ankle_pitch",
    "ankle_roll",
)


class JointMapError(RuntimeError):
    pass


@dataclass(frozen=True)
class JointMap:
    names: tuple[str, ...]
    joint_ids: np.ndarray  # (12,) mjModel joint id
    qpos_adr: np.ndarray  # (12,) index into qpos for each joint
    qvel_adr: np.ndarray  # (12,) index into qvel for each joint
    actuator_ids: np.ndarray  # (12,) mjModel actuator id, same order
    joint_range: np.ndarray  # (12, 2) [low, high] rad
    actuator_forcerange: np.ndarray  # (12, 2) [low, high] N*m
    group: tuple[str, ...]  # (12,) joint-group name, e.g. "hip_pitch"


def build_joint_map(model: mujoco.MjModel) -> JointMap:
    """Resolve CANONICAL_ORDER against `model`, verifying every assumption
    instead of trusting a fixed index scheme (CLAUDE.md: never guess joint
    ordering when it can be verified from model data)."""
    n = len(CANONICAL_ORDER)
    joint_ids = np.zeros(n, dtype=np.int32)
    qpos_adr = np.zeros(n, dtype=np.int32)
    qvel_adr = np.zeros(n, dtype=np.int32)
    actuator_ids = np.zeros(n, dtype=np.int32)
    joint_range = np.zeros((n, 2), dtype=np.float64)
    actuator_forcerange = np.zeros((n, 2), dtype=np.float64)

    seen_qpos_adr: set[int] = set()
    for i, name in enumerate(CANONICAL_ORDER):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise JointMapError(f"joint not found in model: {name}")
        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
            raise JointMapError(f"joint {name} is not a hinge joint")

        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            raise JointMapError(f"actuator not found in model: {name}")
        if model.actuator_trnid[aid, 0] != jid:
            raise JointMapError(f"actuator {name} does not drive joint {name} (trnid mismatch)")
        # A <position> actuator also reports gaintype==FIXED (MuJoCo encodes
        # its kp/kv as bias terms, not gain terms) -- biastype is what
        # actually distinguishes it from a <motor> (direct torque) actuator.
        # Verified empirically: both g1.xml (<position>) and g1_walk.xml
        # (<motor>) report gaintype=FIXED; only biastype differs (AFFINE vs
        # NONE). Confirmed by tests/test_joint_map.py against the real model.
        if model.actuator_biastype[aid] != mujoco.mjtBias.mjBIAS_NONE:
            raise JointMapError(
                f"actuator {name} is not a direct-torque (motor) actuator "
                f"(biastype={model.actuator_biastype[aid]}); expected "
                "assets/g1/g1_walk.xml, not g1.xml"
            )

        qadr = int(model.jnt_qposadr[jid])
        if qadr in seen_qpos_adr:
            raise JointMapError(f"duplicate qpos address for joint {name}")
        seen_qpos_adr.add(qadr)

        joint_ids[i] = jid
        qpos_adr[i] = qadr
        qvel_adr[i] = model.jnt_dofadr[jid]
        actuator_ids[i] = aid
        joint_range[i] = model.jnt_range[jid]
        actuator_forcerange[i] = model.actuator_forcerange[aid]

    if len(set(joint_ids.tolist())) != n:
        raise JointMapError("canonical order does not resolve to unique joints")

    group = JOINT_GROUPS * 2  # left block then right block
    return JointMap(
        names=CANONICAL_ORDER,
        joint_ids=joint_ids,
        qpos_adr=qpos_adr,
        qvel_adr=qvel_adr,
        actuator_ids=actuator_ids,
        joint_range=joint_range,
        actuator_forcerange=actuator_forcerange,
        group=group,
    )
