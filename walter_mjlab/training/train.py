"""Minimal PPO entry point; uses the same task factory as joystick evaluation."""

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    """Train with a shared JSON task configuration and local TensorBoard logging."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log-dir", type=Path)
    args = parser.parse_args(argv)
    if args.num_envs < 1 or args.iterations < 1:
        parser.error("num-envs and iterations must be positive")

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import (
        MjlabOnPolicyRunner,
        RslRlModelCfg,
        RslRlOnPolicyRunnerCfg,
        RslRlPpoAlgorithmCfg,
        RslRlVecEnvWrapper,
    )

    from walter_mjlab.provenance import collect_provenance
    from walter_mjlab.tasks.velocity import VelocityTaskConfig, make_env_cfg

    options = {} if args.config is None else json.loads(args.config.read_text())
    task = VelocityTaskConfig(**{**options, "num_envs": args.num_envs})
    agent_cfg = RslRlOnPolicyRunnerCfg(
        seed=task.seed,
        num_steps_per_env=24,
        max_iterations=args.iterations,
        clip_actions=1.0,
        logger="tensorboard",
        upload_model=False,
        experiment_name="walter_velocity",
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 256, 128),
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(hidden_dims=(512, 256, 256, 128), obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(learning_rate=3e-4, gamma=0.995, entropy_coef=0.01),
    )
    output = args.log_dir or Path("artifacts/training") / datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "task.json").write_text(json.dumps(asdict(task), indent=2) + "\n")
    (output / "agent.json").write_text(json.dumps(asdict(agent_cfg), indent=2) + "\n")
    env = ManagerBasedRlEnv(make_env_cfg(task), device=args.device)
    try:
        (output / "provenance.json").write_text(
            json.dumps(collect_provenance(env.sim.mj_model), indent=2) + "\n"
        )
        wrapper = RslRlVecEnvWrapper(env, clip_actions=1.0)
        runner = MjlabOnPolicyRunner(wrapper, asdict(agent_cfg), str(output), args.device)
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
    finally:
        env.close()


if __name__ == "__main__":
    main()
