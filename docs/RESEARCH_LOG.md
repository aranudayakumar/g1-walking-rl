# Research Log

Append concise, decision-relevant research here. Prefer primary sources.

## Seed research to re-verify at project start

### R-0001 — Unitree framework landscape
Question: Does the challenge-linked Unitree repository train G1 directly in MuJoCo?

Sources:
- https://github.com/unitreerobotics/unitree_rl_gym
- https://github.com/unitreerobotics/unitree_rl_mjlab
- https://github.com/mujocolab/mjlab

Finding at scaffold creation:
- `unitree_rl_gym` documents training in its Gym/Isaac-Gym environment and MuJoCo as the Sim2Sim deployment stage.
- `unitree_rl_mjlab` documents MuJoCo as its training physics backend.
- `mjlab` documents NVIDIA GPU as a training requirement and macOS as evaluation-only.

Implication:
Do not assume the challenge-linked repository is the best direct MuJoCo-training path. Detect the actual machine and make a documented framework decision.

Confidence:
High from official project READMEs, but re-verify because repositories change.

### R-0002 — Claude Code project structure
Question: How should this repository expose persistent instructions and subagents to Claude Code?

Sources:
- https://code.claude.com/docs/
- https://code.claude.com/docs/en/best-practices
- https://code.claude.com/docs/fr/claude-directory

Finding:
- project `CLAUDE.md` is loaded at session start;
- custom project subagents live under `.claude/agents/*.md`;
- commands/skills can package reusable prompts;
- subagents are most useful for isolated/parallel work rather than trivial edits.

Implication:
This scaffold uses `CLAUDE.md`, custom subagents, and reusable locomotion commands.

Confidence:
High from official Claude Code documentation.

### R-0003 — Re-verification of Unitree/mjlab training backends + local sibling evidence (2026-09-18)
Question: Is R-0001's finding still current, and does this specific machine change the decision?

Sources:
- https://github.com/unitreerobotics/unitree_rl_gym (README, doc/setup_en.md, deploy/deploy_mujoco/) — last push 2025-07-25
- https://github.com/unitreerobotics/unitree_rl_mjlab and https://github.com/mujocolab/mjlab (README, installation docs) — both pushed within 24h of this check
- https://github.com/DLR-RM/stable-baselines3 issue #914 (MPS support)
- Local: `../standing-rl` and `../walking-rl` (sibling projects, same machine, same G1 robot, already working)
- Local machine audit: `uname -a`, `sysctl hw.memsize`/`hw.ncpu`, `system_profiler SPDisplaysDataType` -> Apple M5 Pro, arm64, no NVIDIA GPU, 24GB RAM, 15 cores, macOS 26.5.2, system Python 3.14.6.

Finding:
- `unitree_rl_gym` trains via Isaac Gym (Nvidia+Linux); MuJoCo is Sim2Sim validation only. No macOS/CPU training path.
- `mjlab`/`unitree_rl_mjlab` requires an NVIDIA GPU for training per its own README ("macOS is supported for evaluation only"); uses MuJoCo Warp, not MJX-on-CPU. This is current, actively-maintained behavior, not stale docs.
- Direct MuJoCo + Gymnasium + stable-baselines3 has a real CPU/macOS-arm64 story. SB3 does not reliably use Apple MPS (open upstream issue) — must set `device="cpu"` explicitly.
- Sibling local projects `standing-rl` and `walking-rl` (same team, same G1 robot) already independently built and run exactly this stack: Python 3.12 venv (not 3.14 — wheels lag), `mujoco>=3.1`, `gymnasium>=0.29`, `stable-baselines3>=2.3`, `torch>=2.2`, `device: cpu`, vendored G1 MJCF from `mujoco_menagerie` (commit `da76818e269b82289eba39808e2fb91d679d6994`) with leg actuators converted from `<position>` to `<motor>` so a custom per-joint-group PD controller can drive them. Validated numeric starting points found there: `physics_dt=0.002`, `control_decimation=10` (50 Hz control), leg PD gains (hip 100/2, knee 150/4, ankle 40/2), `action_scale=0.25`, PPO `n_steps=512, batch_size=256, net=[256,256], device=cpu`.

Implication:
Confirms ADR-001 (direct MuJoCo + Gymnasium + stable-baselines3, Python 3.12). These sibling numeric values are used only as informed, cited starting points for our own from-scratch walking environment/config — not copied wholesale, since the challenge's own environment/reward must be understood and owned here, and the task (walking, velocity tracking) differs from standing-rl's task (holding a fixed pose).

Confidence:
High — verified against live upstream READMEs/commit activity today, cross-checked against two independently-working local implementations on the identical machine.

---

## Template for new entries

### R-XXXX — Title
Question:

Sources:
- 

Finding:

Implication:

Confidence / remaining uncertainty:
