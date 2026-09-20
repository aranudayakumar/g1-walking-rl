import numpy as np

from envs.contacts import AirTimeTracker

DT = 0.02
THRESH = 0.2


def test_no_reward_while_grounded():
    tracker = AirTimeTracker()
    tracker.reset(np.array([True, True]))
    for _ in range(10):
        r = tracker.step(np.array([True, True]), DT, THRESH)
        assert r == 0.0


def test_no_reward_while_airborne_only_at_touchdown():
    tracker = AirTimeTracker()
    tracker.reset(np.array([True, True]))
    # left foot lifts off for 5 steps, no reward until it lands
    for _ in range(5):
        r = tracker.step(np.array([False, True]), DT, THRESH)
        assert r == 0.0
    # touchdown: air_time = 5*0.02 = 0.1s, threshold 0.2s -> shortfall -0.1
    r = tracker.step(np.array([True, True]), DT, THRESH)
    assert np.isclose(r, 5 * DT - THRESH)


def test_swing_longer_than_threshold_gives_positive_reward():
    tracker = AirTimeTracker()
    tracker.reset(np.array([True, True]))
    n_air_steps = int(THRESH / DT) + 5  # exceed threshold
    for _ in range(n_air_steps):
        tracker.step(np.array([False, True]), DT, THRESH)
    r = tracker.step(np.array([True, True]), DT, THRESH)
    assert r > 0.0


def test_both_feet_touchdown_same_step_sums_contributions():
    tracker = AirTimeTracker()
    tracker.reset(np.array([True, True]))
    for _ in range(3):
        tracker.step(np.array([False, False]), DT, THRESH)
    r = tracker.step(np.array([True, True]), DT, THRESH)
    expected = 2 * (3 * DT - THRESH)  # both feet touch down simultaneously
    assert np.isclose(r, expected)


def test_reset_clears_state():
    tracker = AirTimeTracker()
    tracker.reset(np.array([True, True]))
    for _ in range(10):
        tracker.step(np.array([False, True]), DT, THRESH)
    tracker.reset(np.array([True, True]))
    # immediately grounded after reset -> no leftover air_time counted
    r = tracker.step(np.array([True, True]), DT, THRESH)
    assert r == 0.0
