import mujoco
import numpy as np
import pytest

from control.joint_map import CANONICAL_ORDER, JointMapError, build_joint_map

MODEL_PATH = "assets/g1/scene_walk.xml"
UPSTREAM_MODEL_PATH = "assets/g1/scene.xml"


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(MODEL_PATH)


def test_resolves_all_12_leg_joints(model):
    jm = build_joint_map(model)
    assert jm.names == CANONICAL_ORDER
    assert len(set(jm.joint_ids.tolist())) == 12


def test_qpos_qvel_adr_matches_known_layout(model):
    jm = build_joint_map(model)
    # Verified once via scripts/smoke_test_model.py: floating base occupies
    # qpos[0:7]/qvel[0:6], so legs start at qpos_adr=7, qvel_adr=6.
    assert jm.qpos_adr[0] == 7
    assert jm.qvel_adr[0] == 6
    assert list(jm.qpos_adr) == list(range(7, 19))
    assert list(jm.qvel_adr) == list(range(6, 18))


def test_actuators_are_motors_not_position(model):
    jm = build_joint_map(model)
    for aid in jm.actuator_ids:
        # Both <position> and <motor> report gaintype==FIXED; biastype is
        # what actually distinguishes them (NONE for motor, AFFINE for
        # position) -- verified empirically, see control/joint_map.py.
        assert model.actuator_biastype[aid] == mujoco.mjtBias.mjBIAS_NONE


def test_rejects_upstream_model_with_position_actuators():
    upstream = mujoco.MjModel.from_xml_path(UPSTREAM_MODEL_PATH)
    with pytest.raises(JointMapError):
        build_joint_map(upstream)


def test_forcerange_matches_joint_actuatorfrcrange(model):
    jm = build_joint_map(model)
    # Spot-check one joint against the value read directly from the MJCF.
    idx = CANONICAL_ORDER.index("left_knee_joint")
    assert np.allclose(jm.actuator_forcerange[idx], [-139.0, 139.0])
