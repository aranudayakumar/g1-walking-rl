# Experiment discipline

When touching training, reward, control, environment, or evaluation code:

- Preserve a canonical baseline.
- Prefer one conceptual change per experiment.
- Never start an expensive run without hypothesis, budget, metrics, and stop condition.
- Keep deterministic evaluation separate from stochastic training.
- Log failed runs too.
- Record seeds and relevant config.
- Treat reward curves as training diagnostics, not sufficient evidence of locomotion quality.
- If a change affects dt, action mapping, PD gains, reward scaling, normalization, reset, termination, or joint indexing, explicitly call it out in the experiment record.
