import mujoco
import numpy as np
import pytest

from config.loader import DomainRandomizationConfig
from envs.domain_randomization import DomainRandomizer, PushScheduler

MODEL_PATH = "assets/g1/scene_walk.xml"

DISABLED = DomainRandomizationConfig(enabled=False)
ENABLED = DomainRandomizationConfig(
    enabled=True,
    floor_friction_range=(0.5, 1.5),
    mass_scale_range=(0.8, 1.2),
    motor_strength_range=(0.7, 1.0),
    push_interval_s_range=(2.0, 4.0),
    push_velocity_range=(0.5, 1.5),
)


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(MODEL_PATH)


@pytest.fixture()
def floor_geom_id(model):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")


def test_disabled_leaves_model_unchanged(model, floor_geom_id):
    randomizer = DomainRandomizer(model, floor_geom_id)
    nominal_friction = model.geom_friction[floor_geom_id].copy()
    nominal_mass = model.body_mass.copy()
    rng = np.random.default_rng(0)
    result = randomizer.randomize_episode(rng, DISABLED)
    assert result.motor_strength == 1.0
    assert np.allclose(model.geom_friction[floor_geom_id], nominal_friction)
    assert np.allclose(model.body_mass, nominal_mass)


def test_disabled_consumes_no_rng_draws(model, floor_geom_id):
    """Backward compatibility: every config predating domain randomization
    (enabled=False by default) must sample identical RNG sequences to
    before this feature existed -- reproducibility of the command sampler
    depends on this."""
    randomizer = DomainRandomizer(model, floor_geom_id)
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    randomizer.randomize_episode(rng_a, DISABLED)
    draw_a = rng_a.uniform()
    draw_b = rng_b.uniform()  # no randomize_episode call at all
    assert draw_a == draw_b


def test_enabled_randomizes_friction_within_range(model, floor_geom_id):
    randomizer = DomainRandomizer(model, floor_geom_id)
    nominal = randomizer._nominal_floor_friction[0]
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(20):
        randomizer.randomize_episode(rng, ENABLED)
        f = model.geom_friction[floor_geom_id][0]
        lo, hi = ENABLED.floor_friction_range
        assert nominal * lo - 1e-9 <= f <= nominal * hi + 1e-9
        seen.add(round(f, 6))
    assert len(seen) > 1  # actually varies across episodes


def test_enabled_randomizes_mass_and_inertia_together(model, floor_geom_id):
    randomizer = DomainRandomizer(model, floor_geom_id)
    nominal_mass = randomizer._nominal_body_mass.copy()
    nominal_inertia = randomizer._nominal_body_inertia.copy()  # shape (nbody, 3)
    rng = np.random.default_rng(1)
    randomizer.randomize_episode(rng, ENABLED)

    nonzero = nominal_mass > 0
    ratio_mass = model.body_mass[nonzero] / nominal_mass[nonzero]
    # mass must scale by one consistent factor across all (nonzero-mass) bodies
    assert np.allclose(ratio_mass, ratio_mass[0])

    ratio_inertia = model.body_inertia[nonzero] / nominal_inertia[nonzero]
    # inertia must scale by that exact same factor, per body per axis
    assert np.allclose(ratio_inertia, ratio_mass[0], atol=1e-6)


def test_motor_strength_within_configured_range():
    rng = np.random.default_rng(2)
    class FakeModel:
        geom_friction = np.array([[1.0, 0.0, 0.0]])
        body_mass = np.array([1.0])
        body_inertia = np.array([[1.0, 1.0, 1.0]])
    randomizer = DomainRandomizer.__new__(DomainRandomizer)
    randomizer.model = FakeModel()
    randomizer.floor_geom_id = 0
    randomizer._nominal_floor_friction = FakeModel.geom_friction[0].copy()
    randomizer._nominal_body_mass = FakeModel.body_mass.copy()
    randomizer._nominal_body_inertia = FakeModel.body_inertia.copy()
    for _ in range(50):
        result = randomizer.randomize_episode(rng, ENABLED)
        lo, hi = ENABLED.motor_strength_range
        assert lo - 1e-9 <= result.motor_strength <= hi + 1e-9


def test_push_scheduler_disabled_never_pushes():
    scheduler = PushScheduler()
    rng = np.random.default_rng(0)
    scheduler.reset(rng, DISABLED)
    data = type("D", (), {"qvel": np.zeros(35)})()
    for _ in range(1000):
        pushed = scheduler.maybe_push(data, 0.02, rng, DISABLED)
        assert not pushed
    assert np.allclose(data.qvel, 0.0)


def test_push_scheduler_enabled_eventually_pushes_and_changes_velocity():
    scheduler = PushScheduler()
    rng = np.random.default_rng(3)
    scheduler.reset(rng, ENABLED)
    data = type("D", (), {"qvel": np.zeros(35)})()
    pushed_at_least_once = False
    for _ in range(1000):  # 20s of simulated time, well beyond max push interval (4s)
        if scheduler.maybe_push(data, 0.02, rng, ENABLED):
            pushed_at_least_once = True
    assert pushed_at_least_once
    assert not np.allclose(data.qvel[:2], 0.0)
