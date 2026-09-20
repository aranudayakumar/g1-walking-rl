# Start Here

This scaffold is meant to be extracted into the same directory that already contains the locomotion challenge document.

Claude Code automatically reads `CLAUDE.md` at the start of a session. Project subagents live in `.claude/agents/`, and the project commands in `.claude/commands/` provide repeatable workflows.

## Recommended start

Open a terminal in this directory and start Claude Code normally.

Then either paste the text from `MASTER_PROMPT.md`, or run:

`/locomotion-start`

The first session should not immediately train a policy. It should:
1. read the challenge;
2. audit the machine and repository;
3. research the current implementation choices;
4. choose the smallest viable architecture;
5. update the plan;
6. build the first vertical slice;
7. run a smoke test;
8. log what happened.

## Useful commands

- `/locomotion-start` — bootstrap the project from the challenge.
- `/locomotion-next` — inspect current state and execute the highest-value next step.
- `/locomotion-experiment <idea>` — turn an idea into a bounded, logged experiment.
- `/locomotion-audit` — independently audit correctness, evidence, and challenge coverage.

All files in this scaffold are living documents except the original challenge source.
