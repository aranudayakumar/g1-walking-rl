"""SB3 callback that pushes training progress into every env worker so
reward-term annealing (config.loader.RewardScheduleTerm,
envs.g1_walk_env.G1WalkEnv.set_curriculum_progress) can ramp a weight in
gradually over a single training run, instead of either presenting it from
step 0 (EXP-0022/0024, which collapsed to a degenerate optimum regardless
of learning rate) or requiring a separate fine-tuning stage (EXP-0016/17,
25/26). Only attached in training/train.py when cfg.reward_schedule is
non-empty -- a no-op run pays no extra cost.
"""

from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback


class CurriculumProgressCallback(BaseCallback):
    def __init__(self, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps

    def _on_rollout_start(self) -> None:
        progress = min(1.0, self.num_timesteps / self.total_timesteps)
        self.training_env.env_method("set_curriculum_progress", progress)

    def _on_step(self) -> bool:
        return True
