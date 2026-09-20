# Master Prompt

Read `CLAUDE.md` and the locomotion new-member challenge document in this directory first.

Take ownership of implementing the challenge end-to-end. Work from first principles and optimize for fast, measurable iteration rather than long speculative coding or long training runs.

Start by auditing the repository and machine, then research the material implementation choices that could change the plan. In particular, verify the current official Unitree G1 + MuJoCo options instead of assuming the linked repository trains in MuJoCo. Compare the linked/official Unitree infrastructure, current MuJoCo-native Unitree options, and a direct MuJoCo + Gymnasium + maintained PPO-library path. Choose the path that best satisfies the challenge under the actual hardware constraints, and record the reasoning in the project decision log.

Use the custom subagents in `.claude/agents/` for independent research, simulation/environment work, RL integration, experiment execution, and failure analysis when delegation will genuinely help. Do not spawn agents for trivial tasks. You are the integration owner.

Build in vertical slices:
model load/render -> physics stepping -> control mapping -> environment reset/step -> observation/action validation -> random/scripted smoke tests -> PPO smoke run -> short training -> deterministic evaluation -> failure analysis -> one justified change -> repeat.

Do not cargo-cult rewards or configs. Begin with the minimum reward requested by the challenge, and only add a term when you can name a measured failure it should address. Do not reimplement PPO unless necessary; understand and configure the framework implementation.

Maintain and freely edit the living project files as the evidence changes:
`PROJECT_PLAN.md`, `PROJECT_STATE.md`, `docs/*.md`, `experiments/*.md`, and the Claude agent/command files if useful. Do not edit the original challenge document.

For every substantial experiment, log the hypothesis, exact configuration, seed(s), training budget, metrics, behavior, result, conclusion, and next action. Keep deterministic evaluation separate from training. Prefer small runs that answer one question over expensive runs that answer none.

Begin now. Do not stop at a plan: after research and planning, implement the smallest safe next slice, test it, log the result, update project state, and continue until blocked by a real external limitation or until the challenge is substantially complete.
