import numpy as np

from control.joint_map import JOINT_GROUPS
from control.pd_controller import action_to_target, build_gains, pd_torque


def test_build_gains_shape_and_order():
    kp, kd = build_gains()
    assert kp.shape == (12,)
    assert kd.shape == (12,)
    # Left block then right block, both using the same per-group values.
    assert np.allclose(kp[:6], kp[6:])
    assert np.allclose(kd[:6], kd[6:])


def test_action_to_target_zero_action_returns_default():
    q_default = np.linspace(-0.5, 0.5, 12)
    joint_range = np.stack([q_default - 1.0, q_default + 1.0], axis=1)
    target = action_to_target(np.zeros(12), q_default, action_scale=0.25, joint_range=joint_range)
    assert np.allclose(target, q_default)


def test_action_to_target_scales_and_clips_to_joint_range():
    q_default = np.zeros(12)
    joint_range = np.tile([-0.1, 0.1], (12, 1))
    # action=1 with scale=0.25 would want q_default+0.25, but joint_range caps at 0.1.
    target = action_to_target(np.ones(12), q_default, action_scale=0.25, joint_range=joint_range)
    assert np.allclose(target, 0.1)


def test_action_to_target_clips_out_of_bounds_action():
    q_default = np.zeros(12)
    joint_range = np.tile([-1.0, 1.0], (12, 1))
    target = action_to_target(np.full(12, 5.0), q_default, action_scale=0.25, joint_range=joint_range)
    # action clipped to 1.0 first, so target = 0 + 0.25*1.0 = 0.25
    assert np.allclose(target, 0.25)


def test_pd_torque_zero_error_zero_velocity_gives_zero_torque():
    q = np.zeros(12)
    torque = pd_torque(q, q, np.zeros(12), kp=np.ones(12) * 100, kd=np.ones(12) * 2)
    assert np.allclose(torque, 0.0)


def test_pd_torque_sign_pulls_toward_target():
    kp = np.ones(12) * 100
    kd = np.ones(12) * 2
    q = np.zeros(12)
    target = np.ones(12) * 0.1
    torque = pd_torque(target, q, np.zeros(12), kp, kd)
    assert np.all(torque > 0)  # target ahead of q -> positive restoring torque


def test_pd_torque_respects_forcerange():
    kp = np.ones(12) * 1000  # deliberately huge error * gain to force saturation
    kd = np.zeros(12)
    q = np.zeros(12)
    target = np.ones(12) * 10.0
    forcerange = np.tile([-50.0, 50.0], (12, 1))
    torque = pd_torque(target, q, np.zeros(12), kp, kd, forcerange=forcerange)
    assert np.all(torque <= 50.0) and np.all(torque >= -50.0)


def test_joint_groups_len_matches_half_canonical_order():
    assert len(JOINT_GROUPS) == 6
