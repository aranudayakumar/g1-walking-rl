import numpy as np

from envs.contacts import GroundTimeTracker

DT = 0.02
MAX_STANCE = 0.5


def test_no_penalty_while_within_max_stance():
    tracker = GroundTimeTracker()
    tracker.reset(np.array([True, True]))
    n_steps = int(MAX_STANCE / DT) - 2  # stay just under the threshold
    for _ in range(n_steps):
        r = tracker.step(np.array([True, True]), DT, MAX_STANCE)
        assert r == 0.0


def test_penalty_grows_once_stance_overruns_threshold():
    tracker = GroundTimeTracker()
    tracker.reset(np.array([False, False]))
    n_over = int(MAX_STANCE / DT) + 5
    penalties = []
    for _ in range(n_over):
        penalties.append(tracker.step(np.array([True, True]), DT, MAX_STANCE))
    # once past the threshold, penalty should be positive and non-decreasing
    assert any(p > 0 for p in penalties)
    nonzero = [p for p in penalties if p > 0]
    assert all(b >= a - 1e-9 for a, b in zip(nonzero, nonzero[1:]))


def test_liftoff_resets_stance_time_and_penalty():
    tracker = GroundTimeTracker()
    tracker.reset(np.array([False, False]))
    n_over = int(MAX_STANCE / DT) + 10
    for _ in range(n_over):
        tracker.step(np.array([True, True]), DT, MAX_STANCE)
    # foot lifts off -> stance resets
    r = tracker.step(np.array([False, False]), DT, MAX_STANCE)
    assert r == 0.0
    # immediately back on the ground -> still well under threshold
    r = tracker.step(np.array([True, True]), DT, MAX_STANCE)
    assert r == 0.0


def test_never_lifting_accrues_unbounded_growing_penalty():
    """This is the exact exploit F-0004 found in a touchdown-only reward:
    a foot that never lifts never triggers it. GroundTimeTracker must
    keep penalizing every step a foot overstays, with no cap."""
    tracker = GroundTimeTracker()
    tracker.reset(np.array([False, False]))
    total = 0.0
    for _ in range(500):  # 10s of continuous, unbroken contact
        total += tracker.step(np.array([True, True]), DT, MAX_STANCE)
    assert total > 10.0  # substantial accumulated penalty, not zero


def test_reset_does_not_immediately_flag_initial_contact_as_overrun():
    tracker = GroundTimeTracker()
    tracker.reset(np.array([True, True]))
    r = tracker.step(np.array([True, True]), DT, MAX_STANCE)
    assert r == 0.0
