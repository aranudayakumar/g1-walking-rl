import numpy as np

from config.loader import TerminationConfig
from envs.termination import check_termination

CFG = TerminationConfig(min_pelvis_height_m=0.5, min_projected_gravity_z=-0.3)
OK_QPOS = np.zeros(36)
OK_QVEL = np.zeros(35)


def test_upright_standing_does_not_terminate():
    result = check_termination(
        pelvis_height=0.79, projected_gravity_z=-1.0, qpos=OK_QPOS, qvel=OK_QVEL, cfg=CFG
    )
    assert not result.terminated


def test_low_pelvis_terminates_as_fell_height():
    result = check_termination(
        pelvis_height=0.3, projected_gravity_z=-1.0, qpos=OK_QPOS, qvel=OK_QVEL, cfg=CFG
    )
    assert result.terminated and result.reason == "fell_height"


def test_excessive_tilt_terminates_as_fell_tilt():
    result = check_termination(
        pelvis_height=0.79, projected_gravity_z=0.0, qpos=OK_QPOS, qvel=OK_QVEL, cfg=CFG
    )
    assert result.terminated and result.reason == "fell_tilt"


def test_nonfinite_state_terminates_as_numerical_invalid():
    bad_qpos = OK_QPOS.copy()
    bad_qpos[10] = np.nan
    result = check_termination(
        pelvis_height=0.79, projected_gravity_z=-1.0, qpos=bad_qpos, qvel=OK_QVEL, cfg=CFG
    )
    assert result.terminated and result.reason == "numerical_invalid"


def test_numerical_check_takes_priority_over_height():
    """A NaN state might also report a bogus (e.g. zero) pelvis height;
    the numerical check must win so the reason is diagnosable."""
    bad_qpos = OK_QPOS.copy()
    bad_qpos[0] = np.inf
    result = check_termination(
        pelvis_height=0.0, projected_gravity_z=-1.0, qpos=bad_qpos, qvel=OK_QVEL, cfg=CFG
    )
    assert result.reason == "numerical_invalid"
