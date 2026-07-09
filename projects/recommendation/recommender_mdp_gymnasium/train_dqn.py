import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from dqn_agent import DQNAgent
from micro_recsim_env import MicroRecSimEnv


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "artifacts" / "dqn"


@dataclass
class EpisodeStats:
    total_reward: float
    steps: int
    clicks: int
    final_patience: float
    action_counts: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a DQN agent on the Micro-RecSim environment."
    )
    parser.add_argument(
        "--train-episodes",
        type=int,
        default=5000,
        help="Number of training episodes.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=500,
        help="Number of greedy evaluation episodes.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum number of steps per episode.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden width for the Q-network MLP.",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "silu"],
        help="Activation function used in the Q-network.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for Adam.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.95,
        help="Discount factor.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Mini-batch size sampled from replay buffer.",
    )
    parser.add_argument(
        "--replay-buffer-capacity",
        type=int,
        default=20000,
        help="Maximum replay buffer size.",
    )
    parser.add_argument(
        "--min-buffer-size",
        type=int,
        default=500,
        help="Warmup buffer size before gradient updates start.",
    )
    parser.add_argument(
        "--target-sync-interval",
        type=int,
        default=200,
        help="Sync target network every N gradient updates.",
    )
    parser.add_argument(
        "--epsilon-start",
        type=float,
        default=1.0,
        help="Starting epsilon for exploration.",
    )
    parser.add_argument(
        "--epsilon-end",
        type=float,
        default=0.05,
        help="Final epsilon for exploration.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=500,
        help="Run greedy evaluation every N training episodes.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop after this many eval checkpoints without improving best eval reward. Set 0 to disable.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Torch device selection.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for saving checkpoints and metrics.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def epsilon_by_episode(
    episode_idx: int,
    total_episodes: int,
    epsilon_start: float,
    epsilon_end: float,
) -> float:
    if total_episodes <= 1:
        return epsilon_end

    progress = min(episode_idx / (total_episodes - 1), 1.0)
    return float(epsilon_start + progress * (epsilon_end - epsilon_start))


def run_episode(
    env: MicroRecSimEnv,
    agent: DQNAgent,
    episode_seed: int,
    max_steps: int,
    epsilon: float,
    training: bool,
    min_buffer_size: int,
) -> tuple[EpisodeStats, list[dict[str, float]]]:
    observation, _ = env.reset(seed=episode_seed)
    total_reward = 0.0
    clicks = 0
    action_counts = np.zeros(env.action_space.n, dtype=np.int32)
    update_metrics = []

    for step_idx in range(max_steps):
        if training:
            action = agent.select_action(observation, epsilon=epsilon)
        else:
            action = agent.greedy_action(observation)

        action_counts[action] += 1

        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if training:
            agent.store_transition(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=done,
            )
            if agent.can_update(min_buffer_size=min_buffer_size):
                update_metrics.append(agent.update())

        total_reward += reward
        clicks += int(info["clicked"])
        observation = next_observation

        if done:
            return (
                EpisodeStats(
                    total_reward=total_reward,
                    steps=step_idx + 1,
                    clicks=clicks,
                    final_patience=float(observation[-1]),
                    action_counts=action_counts,
                ),
                update_metrics,
            )

    return (
        EpisodeStats(
            total_reward=total_reward,
            steps=max_steps,
            clicks=clicks,
            final_patience=float(observation[-1]),
            action_counts=action_counts,
        ),
        update_metrics,
    )


def aggregate_episode_stats(episode_stats: list[EpisodeStats]) -> dict:
    rewards = np.array([stats.total_reward for stats in episode_stats], dtype=np.float32)
    steps = np.array([stats.steps for stats in episode_stats], dtype=np.float32)
    clicks = np.array([stats.clicks for stats in episode_stats], dtype=np.float32)
    final_patience = np.array(
        [stats.final_patience for stats in episode_stats],
        dtype=np.float32,
    )
    action_counts = np.sum([stats.action_counts for stats in episode_stats], axis=0)
    action_frequency = action_counts / np.sum(action_counts)

    return {
        "average_total_reward": float(np.mean(rewards)),
        "average_episode_length": float(np.mean(steps)),
        "average_clicks": float(np.mean(clicks)),
        "average_final_patience": float(np.mean(final_patience)),
        "action_frequency": [float(x) for x in action_frequency],
    }


def aggregate_update_metrics(update_metrics: list[dict[str, float]]) -> dict:
    if not update_metrics:
        return {}

    losses = np.array([metric["loss"] for metric in update_metrics], dtype=np.float32)
    mean_q_values = np.array(
        [metric["mean_q_value"] for metric in update_metrics],
        dtype=np.float32,
    )
    mean_targets = np.array(
        [metric["mean_target"] for metric in update_metrics],
        dtype=np.float32,
    )
    return {
        "average_loss": float(np.mean(losses)),
        "average_q_value": float(np.mean(mean_q_values)),
        "average_target": float(np.mean(mean_targets)),
        "update_steps": int(len(update_metrics)),
    }


def evaluate_agent(
    agent: DQNAgent,
    episodes: int,
    max_steps: int,
    seed: int,
) -> dict:
    env = MicroRecSimEnv()
    episode_stats = []

    for episode_idx in range(episodes):
        episode_seed = seed + episode_idx
        stats, _ = run_episode(
            env=env,
            agent=agent,
            episode_seed=episode_seed,
            max_steps=max_steps,
            epsilon=0.0,
            training=False,
            min_buffer_size=0,
        )
        episode_stats.append(stats)

    return aggregate_episode_stats(episode_stats)


def train_agent(args: argparse.Namespace) -> tuple[DQNAgent, list[dict], dict | None, dict | None]:
    env = MicroRecSimEnv()
    observation_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.n)
    device = resolve_device(args.device)

    agent = DQNAgent(
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        activation=args.activation,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        batch_size=args.batch_size,
        target_sync_interval=args.target_sync_interval,
        replay_buffer_capacity=args.replay_buffer_capacity,
        device=device,
        seed=args.seed,
    )

    history = []
    best_snapshot = None
    best_state_dict = None
    evals_since_improvement = 0

    for episode_idx in range(args.train_episodes):
        epsilon = epsilon_by_episode(
            episode_idx=episode_idx,
            total_episodes=args.train_episodes,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
        )
        episode_seed = args.seed + episode_idx
        train_stats, update_metrics = run_episode(
            env=env,
            agent=agent,
            episode_seed=episode_seed,
            max_steps=args.max_steps,
            epsilon=epsilon,
            training=True,
            min_buffer_size=args.min_buffer_size,
        )

        if args.eval_every > 0 and (
            (episode_idx + 1) % args.eval_every == 0
            or episode_idx == args.train_episodes - 1
        ):
            eval_summary = evaluate_agent(
                agent=agent,
                episodes=args.eval_episodes,
                max_steps=args.max_steps,
                seed=args.seed + 100000 + episode_idx,
            )
            update_summary = aggregate_update_metrics(update_metrics)
            snapshot = {
                "episode": episode_idx + 1,
                "epsilon": epsilon,
                "train_total_reward": train_stats.total_reward,
                "train_steps": train_stats.steps,
                "train_clicks": train_stats.clicks,
                "updates": update_summary,
                "eval": eval_summary,
            }
            history.append(snapshot)

            is_best = (
                best_snapshot is None
                or eval_summary["average_total_reward"]
                > best_snapshot["eval"]["average_total_reward"]
            )
            if is_best:
                best_snapshot = snapshot
                best_state_dict = copy.deepcopy(agent.state_dict())
                evals_since_improvement = 0
            else:
                evals_since_improvement += 1

            best_eval_reward = (
                best_snapshot["eval"]["average_total_reward"]
                if best_snapshot is not None
                else float("-inf")
            )
            avg_loss = update_summary.get("average_loss", float("nan"))
            print(
                f"Episode {episode_idx + 1:>5} | "
                f"epsilon={epsilon:.3f} | "
                f"train_reward={train_stats.total_reward:>6.2f} | "
                f"eval_reward={eval_summary['average_total_reward']:>6.3f} | "
                f"best_eval_reward={best_eval_reward:>6.3f} | "
                f"eval_clicks={eval_summary['average_clicks']:>6.3f} | "
                f"avg_loss={avg_loss:>7.4f}"
            )

            if (
                args.early_stop_patience > 0
                and evals_since_improvement >= args.early_stop_patience
            ):
                print(
                    f"Early stopping triggered after {evals_since_improvement} "
                    f"eval checkpoints without improvement."
                )
                break

    return agent, history, best_snapshot, best_state_dict


def save_outputs(
    output_dir: Path,
    args: argparse.Namespace,
    agent: DQNAgent,
    history: list[dict],
    best_snapshot: dict | None,
    best_state_dict: dict | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    final_checkpoint_path = output_dir / "dqn_final.pt"
    best_checkpoint_path = output_dir / "dqn_best.pt"
    metrics_path = output_dir / "training_metrics.json"

    final_state_dict = agent.state_dict()
    torch.save(final_state_dict, final_checkpoint_path)
    if best_state_dict is None:
        best_state_dict = final_state_dict
    torch.save(best_state_dict, best_checkpoint_path)

    final_eval = history[-1]["eval"] if history else {}
    metrics = {
        "config": {
            "train_episodes": args.train_episodes,
            "eval_episodes": args.eval_episodes,
            "max_steps": args.max_steps,
            "hidden_dim": args.hidden_dim,
            "activation": args.activation,
            "learning_rate": args.learning_rate,
            "gamma": args.gamma,
            "batch_size": args.batch_size,
            "replay_buffer_capacity": args.replay_buffer_capacity,
            "min_buffer_size": args.min_buffer_size,
            "target_sync_interval": args.target_sync_interval,
            "epsilon_start": args.epsilon_start,
            "epsilon_end": args.epsilon_end,
            "eval_every": args.eval_every,
            "early_stop_patience": args.early_stop_patience,
            "seed": args.seed,
            "device": args.device,
        },
        "observation_dim": agent.observation_dim,
        "action_dim": agent.action_dim,
        "final_eval": final_eval,
        "best_snapshot": best_snapshot,
        "history": history,
    }

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    print(f"\nSaved final checkpoint to: {final_checkpoint_path}")
    print(f"Saved best checkpoint to: {best_checkpoint_path}")
    print(f"Saved training metrics to: {metrics_path}")


def main() -> None:
    args = parse_args()
    agent, history, best_snapshot, best_state_dict = train_agent(args)

    summary = {
        "observation_dim": agent.observation_dim,
        "action_dim": agent.action_dim,
        "hidden_dim": args.hidden_dim,
        "activation": args.activation,
        "device": str(agent.device),
    }
    print("\nNetwork summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if best_snapshot is not None:
        print("\nBest evaluation snapshot:")
        print(json.dumps(best_snapshot, ensure_ascii=False, indent=2))
    if history:
        print("\nFinal evaluation summary:")
        print(json.dumps(history[-1]["eval"], ensure_ascii=False, indent=2))

    save_outputs(
        output_dir=Path(args.output_dir),
        args=args,
        agent=agent,
        history=history,
        best_snapshot=best_snapshot,
        best_state_dict=best_state_dict,
    )


if __name__ == "__main__":
    main()
