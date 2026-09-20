# Unitree G1 Walking Policy: Implementation Writeup

A reinforcement-learning walking policy for the Unitree G1 humanoid in MuJoCo, trained on a CPU-only Apple M5 Pro. It is the Wisconsin Humanoids locomotion new-member challenge, implemented end to end.

This file describes the **current canonical implementation**: what the environment, control loop and reward are, exactly how the canonical checkpoint was trained, how it was evaluated, and what is and is not established. The full history (32 experiments, the last one incomplete; 11 decision records; 4 failure analyses) lives in `experiments/EXPERIMENT_LOG.md`, `docs/DECISIONS.md` and `docs/FAILURE_ANALYSIS.md`.

## 1. Result

Canonical checkpoint: `experiments/runs/exp0026_air_time_on_exp0023_long/best_checkpoint_2M.zip`

Deterministic evaluation over 20 fixed seeds (1000-1019), one 20 s episode each:

| Metric | Value |
|---|---|
| Falls | **0 / 20** |
| Mean episode length | 20.0 s (all episodes ran the full length) |
| Mean total reward | 1347 |
| Mean velocity-tracking error | 0.036 m/s |
| Mean distance traveled | 4.28 m per episode |
| Distance ratio (traveled / commanded) | 0.918 |
| Yaw drift, mixed commands | 17.6 deg / 10 s |
| Yaw drift, controlled straight-line test (vx = 0.3) | -14.1 deg / 10 s |
| Swing duration (763 touchdowns) | mean 55 ms, median 60 ms, p90 80 ms, max 100 ms |
| Foot-lift range | 3.2 to 5.7 cm |
| Mean torque cost (sum of torque^2 per step) | 0.045 |

Artifacts in the run directory: `curves.png` (training curves), `rollout_seed1002.mp4` (20 s deterministic rollout), `eval_summary.json`, `checkpoint_trend.json`. The controlled gait plot is `diagnostics/reports/gait_diagnostic_exp0026_2M.png`.

## 2. Architecture choice

Direct **MuJoCo + Gymnasium + stable-baselines3 PPO**. I do not reimplement PPO; the library owns rollouts, GAE, clipping and the value loss. The project owns the environment, control mapping, reward, evaluation and experiment tracking.

The official Unitree stacks were evaluated and rejected for this machine:

- `unitree_rl_gym` trains in Isaac Gym, which needs an NVIDIA GPU and Linux.
- `unitree_rl_mjlab` trains with MuJoCo Warp, which needs an NVIDIA GPU (macOS is evaluation-only).

The machine is an Apple M5 Pro (15 cores, 24 GB, no CUDA). PyTorch runs on CPU (`device: cpu`) because SB3 does not reliably use Apple MPS. The environment is Python 3.12; dependencies are in `requirements.txt`. Reasoning and sources: `docs/DECISIONS.md` ADR-001 and `docs/RESEARCH_LOG.md`.

## 3. Robot, simulation and control

**Model.** `g1_29dof_rev_1_0` from `mujoco_menagerie` (vendored in `assets/g1/`, upstream license in `assets/g1/LICENSE`, provenance in `assets/g1/PROVENANCE.md`). `scripts/build_walk_model.py` derives `assets/g1/g1_walk.xml` from it by converting the 12 leg actuators from position actuators to direct-torque `<motor>` actuators, so a PD law owned by this project produces the torque.

**RL-controlled joints (12):** per leg, hip pitch/roll/yaw, knee, ankle pitch/roll. Order is left block then right block (`control/joint_map.py`, `CANONICAL_ORDER`). The waist and both arms (17 joints) are held at the upstream "stand" keyframe pose by position actuators and are not learned.

**Timing.**

| Quantity | Value |
|---|---|
| Physics step | 0.002 s (500 Hz) |
| Control decimation | 10 (policy runs at 50 Hz, dt = 0.02 s) |
| Episode | 20 s = 1000 control steps |

**Action** (12-dim, clipped to [-1, 1]) is a position-target offset from a default crouch:

```
q_target = clip( q_default + 0.25 * clip(action, -1, 1),  joint_min, joint_max )
```

Default leg pose per leg (radians): hip pitch -0.10, hip roll 0, hip yaw 0, knee 0.30, ankle pitch -0.20, ankle roll 0.

**PD law**, evaluated every physics step, then clipped to each actuator's `actuatorfrcrange` from the MJCF:

```
torque = kp * (q_target - q) - kd * qd
```

| Joint group | kp | kd |
|---|---|---|
| hip pitch / roll / yaw | 100 | 2 |
| knee | 150 | 4 |
| ankle pitch / roll | 40 | 2 |

The gains were taken from a sibling project's validated standing setup for the identical robot and were not tuned. Measured torque use: no joint exceeded about 63% of its limit on any evaluated checkpoint, so torque saturation is not what limits the gait (`docs/DECISIONS.md` ADR-010 addendum).

## 4. Observation (44 dimensions, pelvis frame, no normalization)

| Field | Dim | Meaning |
|---|---|---|
| Joint positions | 12 | `q - q_default` (rad) |
| Joint velocities | 12 | `qd` (rad/s) |
| Base angular velocity | 3 | pelvis local frame (rad/s) |
| Projected gravity | 3 | unit gravity direction in the pelvis frame |
| Commanded velocity | 2 | (vx, vy) in the pelvis frame |
| Previous action | 12 | last action vector |

Base **linear** velocity is deliberately **not** observed: it is hard to estimate on real hardware without a state estimator. It is used only inside the reward, as privileged training-time information. No `VecNormalize` or other observation/reward normalization is applied.

## 5. Commands, reset and termination

- **Command:** sampled once per episode (no mid-episode resampling): `vx ~ U[0.0, 0.6] m/s`, `vy ~ U[-0.2, 0.2] m/s`.
- **Reset:** upstream "stand" keyframe base pose, legs at the default crouch, zero velocity. Deterministic given a seed.
- **Termination** (episode ends as a failure):
  - pelvis height < 0.5 m (`fell_height`)
  - projected gravity z > -0.3, i.e. tilted more than about 72.5 degrees (`fell_tilt`)
  - non-finite state (`numerical_invalid`)
- **Truncation:** at 1000 control steps.

## 6. Reward function

The canonical reward (`config/exp0012_air_time.yaml`, unchanged since EXP-0012):

```
r_t =  1.0    * exp( -||v_cmd_xy - v_base_xy||^2 / 0.25^2 )   tracking_lin_vel
     + 0.5    * 1                                              alive_bonus
     - 0.0002 * sum_i( tau_i^2 )                               torque_penalty
     - 0.1    * (base_ang_vel_z)^2                             ang_vel_penalty
     - 0.01   * sum_i( (a_i - a_prev_i)^2 )                    action_rate_penalty
     + 3.0    * sum_{feet touching down at t}( T_swing - 0.2 ) feet_air_time_bonus
```

- `v_base_xy` is the pelvis linear velocity in the pelvis frame (reward-only, not observed).
- `T_swing` is how long that foot was airborne before touching down, in seconds (`envs/contacts.py`, `AirTimeTracker`). The threshold is 0.2 s. The achieved swings (about 60 ms) are below the threshold, so this term is a small net **penalty** that pushes the swing longer. Measured cost was about -0.11 per step on the gait it was fine-tuned from, roughly 7.6% of that gait's reward per step.
- The first three terms are the challenge's stated minimum. Each further term was added only after a named, measured failure (`docs/FAILURE_ANALYSIS.md`):

| Term | Added after | Failure it targets |
|---|---|---|
| `ang_vel_penalty`, `action_rate_penalty` | F-0003 | policy drifted 95 degrees in 10 s and took tiny chattery steps |
| `feet_air_time_bonus` | F-0004 | median swing about 20 ms |

- Two further terms exist in code but are **off** (weight 0) in the canonical config: `stance_overrun_penalty` (closes a "never lift the foot" exploit; it caused 100% falls in EXP-0013) and `heading_deviation_penalty` (`-w * (yaw - yaw_at_reset)^2`, used in the `exp0019` alternative). There is also an opt-in reward-annealing schedule, unused by the canonical, described in section 10.
- Weights were calibrated by replaying the actual target gait through a candidate term and checking what fraction of the per-step reward it costs (ADR-005). A guessed `ang_vel_penalty` of 1.0 in EXP-0010 exceeded the tracking reward and collapsed the policy to standing still.

## 7. PPO configuration (stable-baselines3)

| Setting | Value |
|---|---|
| Policy | `MlpPolicy`, separate policy and value networks, each 2 x 256 (SB3 default activation) |
| Parallel envs | 12 (`SubprocVecEnv`) |
| Rollout | `n_steps` 1024 per env = 12,288 samples per update |
| Minibatch / epochs | 256 / 10 |
| gamma / GAE lambda | 0.99 / 0.95 |
| Clip range | 0.2 |
| Entropy / value coef. / max grad norm | 0.0 / 0.5 / 0.5 |
| Learning rate | 3e-4 by default; **5e-5 for every run in the canonical lineage** |
| Seed | 0 |
| Throughput | about 7,000 to 7,300 steps/s on 12 CPU envs |

## 8. How the canonical checkpoint was trained

Three runs, each warm-starting from the previous one. Wall-clock times are from each run's `run_meta.json`.

| Stage | Run | Config | Init | Budget | Wall-clock |
|---|---|---|---|---|---|
| 1. Base gait | `exp0023_low_lr_from_scratch` | `config/exp0023_low_lr_from_scratch.yaml` (minimal 3-term reward, LR 5e-5) | random | 10M steps | 1368 s (about 23 min) |
| 2. Stride | `exp0025_air_time_on_exp0023` | `config/exp0012_air_time.yaml` | stage 1, 8M checkpoint, LR 5e-5 | 5M steps | 687 s (about 11 min) |
| 3. Selection | `exp0026_air_time_on_exp0023_long` | `config/exp0012_air_time.yaml` | stage 2, 4M checkpoint, LR 5e-5 | 10M steps | 1376 s (about 23 min) |

All three stages ran in full: about 57 minutes and 25M steps in total. The selected checkpoint is 2M steps into stage 3, so the steps that actually fed the canonical are about 8M + 4M + 2M = 14M, roughly 32 minutes. Checkpoints were chosen by evaluating the whole checkpoint trend on the 20 fixed seeds (`evaluation/evaluate_checkpoints.py`), not by taking the final model. In stage 3, `dist_ratio` peaks at 0.918 at 2M and then declines with more budget.

**Why this recipe.** Each step answers a measured failure:

1. Training the minimal reward from scratch at the standard LR (3e-4) shows a repeating climb-crash-recover reward curve. Lowering the LR to 5e-5 removes it (`approx_kl` stayed under 0.061) and reached better tracking (`dist_ratio` 0.848) than any standard-LR base. But that reward has nothing asking for a longer stride, so the gait was a 1.2 cm shuffle, which only the controlled gait diagnostic caught.
2. Adding the stride terms to a from-scratch run collapses to standing still, at both learning rates (EXP-0022, EXP-0024). Introducing them by fine-tuning an already-walking policy at low LR does not (ADR-007). Applying that fine-tune to the better base, instead of to the older standard-LR base, kept the tracking and added a real 60 ms stride (ADR-011).

## 9. Evaluation protocol

`evaluation/evaluate.py`: 20 fixed seeds (1000-1019, disjoint from training seeds), `model.predict(obs, deterministic=True)`, one episode per seed. Reported per episode and in aggregate: length, reward, velocity error, torque cost, termination reason, final pelvis height, distance traveled, distance ratio, yaw drift.

- **Distance ratio** = net displacement / (commanded speed x episode time). It is the metric that separates real walking from surviving while standing still, which reward and survival alone repeatedly failed to do. It is averaged only over episodes that ran at least 50% of the maximum length, because very short episodes can give spuriously large ratios (ADR-003).
- **Yaw drift** = heading change per 10 s, under whatever command each episode sampled.
- **Controlled straight-line test** (`diagnostics/gait_diagnostic.py`, vx = 0.3, vy = 0, 10 s): trajectory plot, foot height, foot contact, liftoff counts. Swing durations are measured directly from contact events. This is the check that catches gait-quality problems the aggregate metrics miss.
- `results/compare.py` tabulates every evaluated run side by side.

Reference points (clean physics, same 20 seeds):

| Checkpoint | dist_ratio | vel_err | falls | yaw (mixed) | Swing median |
|---|---|---|---|---|---|
| **`exp0026` canonical** | **0.918** | **0.036** | 0/20 | 17.6 | 60 ms |
| `exp0017` previous canonical | 0.709 | 0.061 | 0/20 | 36.3 | 40 ms |
| `exp0009` first standard-LR base | 0.705 | 0.119 | 3/20 | 63.9 | about 20 ms |
| `baseline0001` first run (3M steps) | 0.479 | 0.155 | 5/20 | n/a | n/a |

Documented alternatives, none of which replaced the canonical:

| Checkpoint | What it is | Trade-off |
|---|---|---|
| `exp0029_dr_on_exp0026` | Domain randomization on the canonical | Under randomized physics, falls 14/20 to 4/20 and survival 10.2 s to 18.8 s; clean `dist_ratio` drops to 0.643 |
| `exp0031_heading_on_exp0026_lowlr` | Heading penalty fine-tuned at LR 1e-5 | Best raw `dist_ratio` (1.099) with 0/20 falls; `vel_err` worse (0.061); heading gain inconclusive |
| `exp0019_heading_hold` | Older heading-penalty checkpoint | Best single controlled-heading number (-7.5 deg/10 s) but weaker tracking |

## 10. Domain randomization and reward annealing

**Domain randomization** (`envs/domain_randomization.py`, off by default, enabled in `config/exp0021_dr_on_canonical.yaml`):

| Randomized | Range |
|---|---|
| Floor friction scale (per episode) | x[0.5, 1.5] |
| Global body mass and inertia scale (per episode) | x[0.85, 1.15] |
| Motor strength scale (torque multiplier, per episode) | x[0.7, 1.0] |
| Horizontal pelvis velocity push, every 3 to 6 s | 0.1 to 0.5 m/s |

**Reward annealing** (new, `config.loader.RewardScheduleTerm`, `G1WalkEnv.set_curriculum_progress`, `training/curriculum_callback.py`): a reward weight ramps linearly from `start` to `end` over the first `end_fraction` of training, driven by an SB3 callback that pushes progress into every env worker once per rollout. It is opt-in; an empty `reward_schedule` leaves behavior unchanged (`tests/test_reward_schedule.py`). Its first experiment, EXP-0032 (stride and heading terms annealed in over the first 40% of a from-scratch run), was **interrupted at about 13.5M of 20M steps and produced no evaluation**. Only the mechanism is verified, not the outcome.

## 11. Reproducing the canonical

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. python -m pytest tests/ -q          # 65 tests

# Stage 1: base gait (about 23 min)
PYTHONPATH=. python -m training.train --run-name exp0023_low_lr_from_scratch \
  --timesteps 10000000 --n-envs 12 --seed 0 --checkpoint-freq 2000000 \
  --config config/exp0023_low_lr_from_scratch.yaml

# Stage 2: add the stride reward by fine-tuning (about 11 min)
PYTHONPATH=. python -m training.train --run-name exp0025_air_time_on_exp0023 \
  --timesteps 5000000 --n-envs 12 --seed 0 --checkpoint-freq 1000000 \
  --config config/exp0012_air_time.yaml \
  --init-from experiments/runs/exp0023_low_lr_from_scratch/best_checkpoint_8M.zip \
  --init-learning-rate 5e-5

# Stage 3: continue and select (about 23 min); canonical = 2M-step checkpoint
PYTHONPATH=. python -m training.train --run-name exp0026_air_time_on_exp0023_long \
  --timesteps 10000000 --n-envs 12 --seed 0 --checkpoint-freq 2000000 \
  --config config/exp0012_air_time.yaml \
  --init-from experiments/runs/exp0025_air_time_on_exp0023/best_checkpoint_4M.zip \
  --init-learning-rate 5e-5

# Evaluate, plot, record, diagnose
PYTHONPATH=. python -m evaluation.evaluate_checkpoints --run exp0026_air_time_on_exp0023_long --config config/exp0012_air_time.yaml
PYTHONPATH=. python -m evaluation.evaluate --model experiments/runs/exp0026_air_time_on_exp0023_long/best_checkpoint_2M.zip --config config/exp0012_air_time.yaml --episodes 20
PYTHONPATH=. python -m results.plot_curves --run exp0026_air_time_on_exp0023_long
PYTHONPATH=. python -m evaluation.record_video --model experiments/runs/exp0026_air_time_on_exp0023_long/best_checkpoint_2M.zip --config config/exp0012_air_time.yaml --seed 1002 --out rollout.mp4
PYTHONPATH=. python -m diagnostics.gait_diagnostic --model experiments/runs/exp0026_air_time_on_exp0023_long/best_checkpoint_2M.zip --config config/exp0012_air_time.yaml --vx 0.3 --vy 0.0
```

Stage 2 and 3 copy the selected checkpoint (for example `ppo_g1_walk_7999968_steps.zip` in stage 1's `checkpoints/` becomes `best_checkpoint_8M.zip`) to `best_checkpoint_<N>M.zip` after evaluating the trend. Training is seeded, but wall-clock time varied a lot in this project (a 10M-step run took from about 23 minutes to about 3.5 hours under machine load), so budget time generously.

## 12. Repository layout

```
config/        one YAML per experiment; config/loader.py is the typed schema
control/       joint_map.py (joint order/indexing), pd_controller.py, default_pose.py
envs/          g1_walk_env.py (Gymnasium env), observations.py, contacts.py,
               termination.py, reset.py, domain_randomization.py
rewards/       walking_reward.py (all reward terms, returned separately)
training/      train.py (PPO entry), vec_env.py, curriculum_callback.py
evaluation/    evaluate.py, evaluate_checkpoints.py, record_video.py
diagnostics/   gait_diagnostic.py and reports/ (controlled-gait plots)
results/       compare.py (cross-run table), plot_curves.py
scripts/       model build and smoke tests
tests/         65 unit/contract tests
assets/g1/     vendored G1 MJCF and meshes (upstream license included)
experiments/   EXPERIMENT_LOG.md and runs/<name>/ (configs, eval JSON, curves, videos,
               best checkpoints)
docs/          DECISIONS (ADR-001..011), FAILURE_ANALYSIS, RL_PIPELINE, SIMULATION,
               DELIVERABLES, RESEARCH_LOG, AUDIT, LEARNING_NOTES
```

`PROJECT_STATE.md` is the short handoff file; `CLAUDE.md`, `MASTER_PROMPT.md` and `PROJECT_PLAN.md` are the original instructions this work followed.

## 13. Limits and caveats

- **Single training seed.** Every run used seed 0, as did all earlier runs, so no seed-to-seed variance is measured. The canonical checkpoint sits at a narrow peak (more fine-tuning on the same recipe made it worse), so treat the exact numbers as one draw.
- **Sim only, CPU only.** No hardware transfer, no GPU convergence run. The challenge names GPU convergence as optional.
- **Stride is short of human-scale.** A 60 ms median swing is the longest verified in this project but is well under a deliberate 200 to 300 ms stride.
- **Heading is not fully resolved.** `exp0019` still holds the best single controlled-heading measurement (-7.5 vs -14.1 deg/10 s), and three attempts to close that gap on the canonical (EXP-0027, 0028, 0031) did not. The controlled test is one 10 s trial and is noisy between checkpoints.
- **High commanded speeds** (above about 0.4 m/s) no longer cause falls on the canonical, but heading still degrades disproportionately there.
- **Upper body is not controlled**; no arm swing.
- **Robustness is a separate checkpoint.** The canonical was trained on nominal physics only; under randomized physics it fell in 14 of 20 episodes.
- **EXP-0032** (reward annealing) did not finish; its result is unknown.
