"""PPO training entry point. We do not reimplement PPO/GAE/value/clipping
-- stable-baselines3 owns all of that; our job is environment + config
(CLAUDE.md).

Usage:
    python -m training.train --run-name smoke0001 --timesteps 4096 --n-envs 2 --no-subproc
    python -m training.train --run-name baseline0001 --timesteps 2_000_000 --n-envs 12
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

from config.loader import load_config
from training.curriculum_callback import CurriculumProgressCallback
from training.vec_env import make_vec_env

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "experiments" / "runs"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--timesteps", type=int, required=True)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-subproc", action="store_true", help="use DummyVecEnv (smoke tests)")
    parser.add_argument("--checkpoint-freq", type=int, default=0, help="0 = only save at the end")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument(
        "--init-from",
        default=None,
        help="warm-start from an existing checkpoint (.zip) instead of random init; "
        "PPO hyperparameters/policy architecture still come from --config, but the "
        "loaded weights determine starting behavior. Env obs/action space must match.",
    )
    parser.add_argument(
        "--init-learning-rate",
        type=float,
        default=None,
        help="override the learning rate after --init-from (e.g. a lower LR for fine-tuning); ignored without --init-from",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = RUNS_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    vec_env = make_vec_env(cfg, n_envs=args.n_envs, seed=args.seed, subproc=not args.no_subproc)

    if args.init_from:
        model = PPO.load(args.init_from, env=vec_env, device=cfg.ppo.device)
        if args.init_learning_rate is not None:
            # SB3 pre-compiles a schedule from learning_rate at construction time;
            # setting the attribute alone is not enough, the schedule must be rebuilt.
            from stable_baselines3.common.utils import get_schedule_fn

            model.learning_rate = args.init_learning_rate
            model.lr_schedule = get_schedule_fn(args.init_learning_rate)
        print(f"warm-started from {args.init_from}" + (
            f" (learning_rate overridden to {args.init_learning_rate})" if args.init_learning_rate else ""
        ))
    else:
        model = PPO(
            cfg.ppo.policy,
            vec_env,
            learning_rate=cfg.ppo.learning_rate,
            n_steps=cfg.ppo.n_steps,
            batch_size=cfg.ppo.batch_size,
            n_epochs=cfg.ppo.n_epochs,
            gamma=cfg.ppo.gamma,
            gae_lambda=cfg.ppo.gae_lambda,
            clip_range=cfg.ppo.clip_range,
            ent_coef=cfg.ppo.ent_coef,
            vf_coef=cfg.ppo.vf_coef,
            max_grad_norm=cfg.ppo.max_grad_norm,
            policy_kwargs=dict(net_arch=dict(pi=cfg.ppo.policy_net, vf=cfg.ppo.value_net)),
            target_kl=cfg.ppo.target_kl,
            device=cfg.ppo.device,
            seed=args.seed,
            verbose=1,
        )
    model.set_logger(configure(str(run_dir), ["stdout", "csv", "tensorboard"]))

    callbacks = []
    if args.checkpoint_freq > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=max(args.checkpoint_freq // args.n_envs, 1),
                save_path=str(run_dir / "checkpoints"),
                name_prefix="ppo_g1_walk",
            )
        )
    if cfg.reward_schedule:
        callbacks.append(CurriculumProgressCallback(total_timesteps=args.timesteps))
        print(f"reward annealing active: {cfg.reward_schedule}")

    meta = {
        "run_name": args.run_name,
        "timesteps": args.timesteps,
        "n_envs": args.n_envs,
        "seed": args.seed,
        "subproc": not args.no_subproc,
        "config_path": args.config,
        "init_from": args.init_from,
        "init_learning_rate": args.init_learning_rate,
        "reward_schedule": {k: vars(v) for k, v in cfg.reward_schedule.items()} or None,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=callbacks or None, progress_bar=False)
    elapsed = time.time() - t0

    final_path = run_dir / "final_model.zip"
    model.save(str(final_path))
    meta["elapsed_seconds"] = elapsed
    meta["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nsaved {final_path} (elapsed {elapsed:.1f}s, {args.timesteps / elapsed:.0f} steps/s)")
    vec_env.close()


if __name__ == "__main__":
    main()
