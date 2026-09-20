# Locomotion Challenge — Claude Project Instructions

## Mission
Implement the locomotion new-member challenge in this repository from first principles with a fast, evidence-driven loop.

Expected source: `Locomotion New Member Document.md`. If renamed, locate the obvious locomotion/new-member challenge file first.

Treat that document as authoritative for required goals and deliverables. External research clarifies implementation choices; it does not silently replace the assignment.

## End-to-end goal
Build and understand:
model load/step -> observation -> policy -> action/control -> reward/termination ->
rollout -> PPO update -> checkpoint -> deterministic evaluation -> measured iteration.

A correct, explainable pipeline with clear learning progress matters more than a polished gait.

## Own the task-specific parts
Understand and implement explicitly:
- robot/model loading and joint/actuator indexing;
- physics dt and control dt/decimation;
- action semantics and control mapping;
- observations, units, normalization, and reference frames;
- reward terms and scales;
- reset, termination, and truncation;
- evaluation and experiment tracking.

A maintained library may own PPO/GAE/value/clipping internals. Do not reimplement PPO unless required.

For important variables, know: physical meaning, units, source, why it exists, and what failure a wrong value would cause.

## Fast iteration
Prefer:
model loads -> one step -> N steps -> repeatable reset -> zero/random actions ->
scripted smoke test -> PPO import -> tiny PPO smoke run -> short learning run ->
deterministic evaluation -> diagnosis -> one justified change -> repeat.

Do not begin with a long training run.

Before expensive work record:
- hypothesis;
- metric;
- budget;
- stop condition.

After meaningful changes:
- run the smallest relevant test;
- record the result;
- update `PROJECT_STATE.md`;
- revise `PROJECT_PLAN.md` when evidence changes it.

## Research
Research only when it can change a decision, resolve uncertainty, or explain failure.

Prefer:
1. challenge document;
2. official Unitree repositories/docs;
3. MuJoCo docs;
4. PPO/GAE original papers or maintained implementation docs;
5. Berkeley Humanoid reference code/paper;
6. secondary sources only when necessary.

Append material findings to `docs/RESEARCH_LOG.md` with question, sources, finding, project implication, and confidence.

Never copy a complicated reward/config without tracing why each part exists.

## Framework choice must be researched
At startup compare at least:
- official `unitreerobotics/unitree_rl_gym`;
- official `unitreerobotics/unitree_rl_mjlab`;
- direct MuJoCo + Gymnasium + maintained PPO library.

Evaluate actual MuJoCo-training support, hardware requirements, OS support, CPU feasibility, framework opacity, setup risk, and challenge alignment.

Record the choice and rejected alternatives in `docs/DECISIONS.md`.
Do not choose a route merely because it is newer.

## Hardware awareness
Inspect OS, architecture, Python, GPU/CUDA availability, memory if useful, and installed tooling.
Never assume NVIDIA/CUDA exists.

With limited compute:
- validate correctness with tiny runs;
- reduce environment count/budget;
- separate pipeline validation from convergence;
- make remote/GPU convergence optional rather than blocking local progress.

## Experiments
`experiments/EXPERIMENT_LOG.md` is append-only history.

For substantial runs record:
- run ID/date and code state;
- hypothesis;
- exact config and seeds;
- env count and training budget;
- physics/control timestep;
- reward scales;
- relevant PPO settings;
- metrics and behavior;
- artifacts/checkpoints;
- conclusion and next action.

Do not say "improved" without naming a comparable metric/protocol.

## Evaluation
Keep training and evaluation separate.

Evaluation should normally be deterministic with a fixed seed/command set/horizon.

Track at minimum:
- episode length/survival;
- return and reward components;
- commanded vs actual velocity;
- base height/orientation;
- termination reason;
- action saturation;
- torque usage when available.

Add metrics only to answer concrete questions.

## Rewards
Start minimal:
- velocity tracking;
- survival/alive signal if justified;
- torque/effort penalty.

Only add a term after identifying the measured behavior it is meant to fix and logging the hypothesis.

## Subagents
Use `.claude/agents/` selectively when work is independent, parallel, or benefits from isolated context:
- `locomotion-researcher`;
- `sim-environment-engineer`;
- `rl-training-engineer`;
- `experiment-runner`;
- `failure-analyst`.

For trivial edits or tightly sequential work, work directly. The main agent owns integration and decisions.

## Living files
Claude may edit, split, rename, archive, or add planning/support files when justified:
- `PROJECT_PLAN.md`;
- `PROJECT_STATE.md`;
- `docs/*.md`;
- `experiments/*.md`;
- `prompts/*.md`;
- `.claude/agents/*.md`;
- `.claude/commands/*.md`.

Do not modify the original challenge document unless the user explicitly asks.
Record material direction changes in `docs/DECISIONS.md`.

## Definition of done
Minimum:
- model renders/steps;
- environment contract is tested;
- PPO collects and updates without silent shape/reset bugs;
- deterministic evaluation works;
- at least one measured learning-progress comparison exists;
- experiments/failures are documented;
- requested writeup/curves/video-attempt can be produced.

Before completion run `/locomotion-audit` or an equivalent independent audit.
