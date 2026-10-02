# Unitree G1 Walking with Reinforcement Learning

A CPU-friendly humanoid locomotion project built with **MuJoCo, Gymnasium, and Stable-Baselines3 PPO**. A policy controls the G1's 12 leg joints to follow commanded forward and lateral velocities. The repository includes the simulation environment, explicit PD controller, training and evaluation tools, selected checkpoints, videos, learning curves, and an evidence-driven experiment history.

The goal is to make the complete learning pipeline understandable: model → observation → policy → joint targets → torque control → simulation → reward → PPO update → deterministic evaluation.

## Contents

- [Results and limitations](#results-and-limitations)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Evaluate the saved policy](#evaluate-the-saved-policy)
- [Training and fine-tuning](#training-and-fine-tuning)
- [Environment and control](#environment-and-control)
- [Configuration](#configuration)
- [Evaluation and experiment workflow](#evaluation-and-experiment-workflow)
- [Repository layout](#repository-layout)
- [Troubleshooting](#troubleshooting)
- [Documentation and model provenance](#documentation-and-model-provenance)

## Results and limitations

The canonical checkpoint is [`best_checkpoint_2M.zip`](experiments/runs/exp0026_air_time_on_exp0023_long/best_checkpoint_2M.zip), from EXP-0026. Use it with **`config/exp0012_air_time.yaml`**, rather than the minimal default reward configuration.

Recorded deterministic evaluation over seeds 1000–1019, with a 20-second horizon and clean simulation physics:

| Metric | Recorded result |
| --- | ---: |
| Falls | 0 / 20 episodes |
| Mean episode duration | 20.0 s |
| Mean velocity error | 0.0361 m/s |
| Mean net displacement | 4.28 m |
| Mean reliable distance ratio | 0.918 |
| Mean absolute endpoint yaw drift, normalized to 10 s | 17.62° |
| Mean episode return | 1346.86 |

These are historical results from the saved [evaluation report](experiments/runs/exp0026_air_time_on_exp0023_long/eval_summary.json), not a guarantee for other commands, physics settings, or dependency versions. Distance is net horizontal displacement, not accumulated path length. Return depends on the reward configuration and should not be compared across different rewards without inspecting physical metrics.

The policy demonstrates stable velocity tracking in simulation, but natural forward walking remains an open problem. Later local investigations documented short swing phases, wide leg abduction, intermittent loaded-foot slip, and heading drift; they did not replace EXP-0026. Survival and speed tracking alone do not establish good gait quality. The repository also records domain-randomized alternatives and their clean-physics tradeoffs. There is no validated real-robot deployment pipeline.

## Installation

Use **Python 3.12**, the version used for this project. Training runs on CPU; CUDA is not required. Development and experiments were performed on Apple Silicon macOS. Other platforms have not been established by the project records.

```bash
git clone https://github.com/aranudayakumar/g1-walking-rl.git
cd g1-walking-rl
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate` instead. Run the following commands from the **repository root** so configuration paths and Python module imports resolve correctly.

Dependencies include MuJoCo, Gymnasium, Stable-Baselines3, PyTorch, PyYAML, pandas, pytest, TensorBoard, Matplotlib, and ImageIO/FFmpeg. `requirements.txt` specifies minimum versions rather than a locked environment; preserve installed versions when reproducing an experiment exactly. The robot XML and meshes are vendored under `assets/g1/`.

## Quick start

Validate the model, controller, and environment before committing to a long run:

```bash
python -m scripts.smoke_test_model
python -m scripts.smoke_test_pd_hold
python -m scripts.smoke_test_env
python -m pytest -q
```

The PD hold check verifies numerical behavior and control mapping. A zero-action free-standing biped can fall because fixed joint targets do not provide active balance control.

Run a small end-to-end PPO smoke experiment:

```bash
python -m training.train --run-name smoke0001 --timesteps 4096 --n-envs 2 --no-subproc
python -m evaluation.evaluate --model experiments/runs/smoke0001/final_model.zip --episodes 5
```

This validates rollout collection, updates, saving, and reloading; 4096 steps are not enough to expect a learned walking gait. Use a new run name for each experiment to avoid overwriting outputs.

## Evaluate the saved policy

Evaluate the canonical checkpoint without training:

```bash
python -m evaluation.evaluate \
  --model experiments/runs/exp0026_air_time_on_exp0023_long/best_checkpoint_2M.zip \
  --config config/exp0012_air_time.yaml --episodes 20
```

The command prints JSON containing per-episode measurements and an aggregate summary. To save it, redirect stdout to a new JSON file.

Record an offscreen rollout:

```bash
python -m evaluation.record_video \
  --model experiments/runs/exp0026_air_time_on_exp0023_long/best_checkpoint_2M.zip \
  --config config/exp0012_air_time.yaml \
  --seed 1002 --out rollout_seed1002.mp4
```

Recording supports `--width`, `--height`, and `--max-seconds`. It requires a working graphics backend even though it does not open an on-screen viewer. Existing videos and plots are available in the [experiment artifacts](experiments/runs/).

## Training and fine-tuning

Start a longer minimal-reward baseline:

```bash
python -m training.train \
  --run-name baseline_new --config config/default.yaml \
  --timesteps 2000000 --n-envs 8 --seed 0 --checkpoint-freq 250000
```

Training uses subprocess vector environments by default. Adjust environment count to your CPU and memory; use `--no-subproc` for a small in-process debugging run. A baseline run is not a reproduction of the canonical policy, which came from a sequence of base training and fine-tuning experiments.

Warm-start from an existing checkpoint:

```bash
python -m training.train \
  --run-name finetune_new --config config/exp0012_air_time.yaml \
  --init-from experiments/runs/exp0026_air_time_on_exp0023_long/best_checkpoint_2M.zip \
  --init-learning-rate 0.00001 \
  --timesteps 262144 --n-envs 8 --seed 0 --checkpoint-freq 131072
```

`--init-from` retains the checkpoint's policy architecture and PPO hyperparameters. The configuration still supplies the environment and device; `--init-learning-rate` explicitly overrides the loaded learning rate. Observation and action spaces must match. Without `--init-from`, the learning-rate override is ignored. This example demonstrates continuation, not a claim of improved performance.

Outputs are written to `experiments/runs/<run-name>/`:

| Artifact | Purpose |
| --- | --- |
| `run_meta.json` | Seed, budget, environment count, config snapshot, effective PPO settings, warm-start source, and timing |
| `progress.csv` | Training statistics and reward component logs |
| TensorBoard event files | Interactive training inspection |
| `checkpoints/ppo_g1_walk_*_steps.zip` | Periodic policies when checkpoint saving is enabled |
| `final_model.zip` | Final policy |

PPO collects whole rollouts, so actual steps can exceed the requested budget. Checkpoint frequency is adjusted for the number of vector environments. Routine checkpoint directories, final models, and TensorBoard events are ignored by Git; selected best checkpoints and analysis artifacts are retained.

```bash
tensorboard --logdir experiments/runs
python -m results.plot_curves --run baseline_new
```

## Environment and control

The model has a floating base and 29 actuated joints. RL controls the 12 leg joints; the waist and arms remain at fixed position targets. The walking model converts leg position actuators to direct torque motors so the project's own PD gains determine control.

| Property | Default |
| --- | --- |
| Physics timestep | 0.002 s, 500 Hz |
| Control decimation | 10 physics steps per action |
| Policy frequency | 50 Hz |
| Episode limit | 20 s / 1000 control steps |
| Action space | 12 float32 values in `[-1, 1]` |
| Observation space | 44 float32 values |
| Forward command | Uniform in `[0.0, 0.6]` m/s |
| Lateral command | Uniform in `[-0.2, 0.2]` m/s |

Commands are sampled once at episode reset. Velocity commands and tracking use the pelvis frame; this is not a world-frame navigation or yaw-command task.

Observation fields, in order:

| Field | Size | Meaning |
| --- | ---: | --- |
| Joint position offsets | 12 | Leg positions relative to the default pose, radians |
| Joint velocities | 12 | Leg angular velocities, rad/s |
| Base angular velocity | 3 | Pelvis-frame angular velocity, rad/s |
| Projected gravity | 3 | Unit gravity direction in the pelvis frame |
| Velocity command | 2 | Desired forward/lateral velocity, m/s |
| Previous action | 12 | Previous normalized joint action |

Base linear velocity is used to calculate reward but is not supplied to the policy. Joint ordering is left leg then right leg, with hip pitch, hip roll, hip yaw, knee, ankle pitch, and ankle roll in each block. Indices are resolved by name against the loaded model.

For the default action scale, the controller computes:

```text
q_target = clip_to_joint_limits(q_default + 0.25 * action)
torque   = clip_to_actuator_limits(kp * (q_target - q) - kd * q_velocity)
```

The joint target is held over ten physics steps, while PD torque is recomputed at each physics step. Default gains for both legs:

| Joint group | kp | kd |
| --- | ---: | ---: |
| Hip pitch / roll / yaw | 100 | 2 |
| Knee | 150 | 4 |
| Ankle pitch / roll | 40 | 2 |

The minimal reward combines exponential velocity tracking, an alive bonus, and a squared-torque penalty. Experimental configurations add terms such as angular-velocity and action-rate penalties, foot-air-time bonuses, or heading penalties to address measured failures. See [the pipeline specification](docs/RL_PIPELINE.md) and [failure analysis](docs/FAILURE_ANALYSIS.md) for the rationale.

Episodes terminate for non-finite state, pelvis height below 0.5 m, or projected-gravity z above −0.3 (roughly 72.5° tilt). Reaching the time limit is a separate truncation.

## Configuration

`config/loader.py` defines the configuration schema. YAML files specify model path, simulation timing, action scale, PD gains, command ranges, reward weights, termination thresholds, and PPO settings. Experiment configurations also support optional domain randomization and reward-weight schedules.

- `config/default.yaml`: minimal starting reward and CPU PPO settings.
- `config/exp0012_air_time.yaml`: environment/reward configuration for the canonical EXP-0026 checkpoint.
- `config/exp0021_dr_on_canonical.yaml`: documented randomization recipe from an earlier canonical lineage; match it to its experiment record before use.
- Other `config/exp*.yaml` files: hypotheses and alternatives, not necessarily successful improvements.

Default PPO uses separate two-layer `[256, 256]` policy and value networks, learning rate `3e-4`, 1024 steps per environment per rollout, batch size 256, ten update epochs, discount 0.99, and GAE lambda 0.95. Stable-Baselines3 owns PPO, GAE, clipping, and optimization.

When reproducing a saved run, consult its `run_meta.json` and experiment record: the YAML filename alone does not capture loaded checkpoint hyperparameters or an overridden learning rate. Keep configured physics timestep consistent with the model XML; the environment asserts this at construction.

## Evaluation and experiment workflow

Evaluation uses deterministic action selection and a fixed set of up to 20 seeds, disjoint from typical training seeds. `--episodes` selects a prefix of this set; requesting more than 20 does not create additional evaluation seeds.

Reports include episode duration, return, velocity error, torque cost, termination reason, final height, net displacement, distance ratio, and endpoint yaw drift. Distance ratios are included in the reliable aggregate only for episodes lasting at least half the configured horizon. Endpoint yaw is wrapped to ±180° and is not a measure of accumulated turning.

Evaluate a run's available periodic checkpoints and compare recorded experiments:

```bash
python -m evaluation.evaluate_checkpoints \
  --run baseline_new --config config/default.yaml --episodes 20
python -m results.compare
```

Inspect a fixed-command gait directly:

```bash
python -m diagnostics.gait_diagnostic \
  --model experiments/runs/exp0026_air_time_on_exp0023_long/best_checkpoint_2M.zip \
  --config config/exp0012_air_time.yaml \
  --vx 0.3 --vy 0.0 --seconds 10 --out gait_diagnostic.png
```

Before a substantial experiment, record a hypothesis, exact configuration and seeds, training budget, metric, and stop condition. Compare checkpoints under the same commands and physics settings. Examine displacement, heading, and stepping alongside reward and survival; select checkpoints based on evaluated behavior rather than assuming the final training step is best. Record both improvements and regressions in [the experiment log](experiments/EXPERIMENT_LOG.md).

## Repository layout

```text
assets/g1/       Vendored robot meshes, source XML, derived walking model
config/          Configuration schema and experiment YAML files
control/         Joint mapping, default pose, PD target/torque control
envs/            Gymnasium environment, observations, reset, contacts,
                 termination, and domain randomization
rewards/         Walking reward calculation
training/        PPO entry point, vector environments, curriculum callback
evaluation/      Deterministic evaluation, checkpoint sweeps, video recording
diagnostics/     Direct gait measurements and research audits
results/         Training plots and cross-experiment comparison
scripts/         Model generation and simulation smoke checks
tests/           Environment, control, reward, contact, and randomization tests
experiments/     Experiment log, selected policies, metrics, plots, videos
docs/            Design decisions, pipeline explanation, research, failure analysis
```

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Imports or YAML paths fail | Activate `.venv` and run `python -m ...` from the repository root. |
| A dependency cannot install | Use the project's Python 3.12 environment and update pip; newer Python versions may lack compatible wheels. |
| Training is slow | Reduce `--n-envs`, close competing workloads, and use a small budget to measure local throughput. |
| Subprocess debugging is difficult | Start with `--n-envs 1 --no-subproc`. |
| XML timestep assertion fails | Match `sim.physics_dt` to the loaded model's timestep. |
| A checkpoint fails to load or behaves differently | Confirm file existence, matching observation/action spaces, original config, and run metadata. Ignored final/periodic models are not included in a fresh clone. |
| Offscreen recording fails | Check MuJoCo's graphics backend on your platform. On a compatible headless Linux system, EGL may be selected with `MUJOCO_GL=egl`; it still requires suitable graphics libraries. |
| High reward but poor-looking walking | Inspect fixed-command gait, contact/swing timing, displacement, and yaw; reward is not a gait-quality certificate. |

## Documentation and model provenance

- [Start here](START_HERE.md): project orientation.
- [Simulation](docs/SIMULATION.md): model and control details.
- [RL pipeline](docs/RL_PIPELINE.md): observations, rewards, and learning contract.
- [Decisions](docs/DECISIONS.md): framework choice and experiment conclusions.
- [Failure analysis](docs/FAILURE_ANALYSIS.md): measured failures and attempted fixes.
- [Experiment log](experiments/EXPERIMENT_LOG.md): historical budgets, protocols, and outcomes.
- [Deliverables](docs/DELIVERABLES.md): writeup and artifact guide.
- [Project state](PROJECT_STATE.md) and [plan](PROJECT_PLAN.md): research handoff and next steps.

The G1 assets were vendored from Google DeepMind's MuJoCo Menagerie, `unitree_g1/`, at commit `da76818e269b82289eba39808e2fb91d679d6994`. See [asset provenance](assets/g1/README.md) and the [asset license](assets/g1/LICENSE). The walking-model transformation is implemented in `scripts/build_walk_model.py`; the historical asset README also contains standing-model naming inherited from an earlier scaffold. This repository does not currently provide a separate top-level license for its project code.
