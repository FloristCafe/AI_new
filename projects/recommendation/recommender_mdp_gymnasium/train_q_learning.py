import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from micro_recsim_env import MicroRecSimEnv


DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent / "artifacts" / "tabular_q_learning"
)


@dataclass
class EpisodeStats:
    total_reward: float
    steps: int
    clicks: int
    final_patience: float
    action_counts: np.ndarray


@dataclass
class TrainingResult:
    agent: "TabularQAgent"
    encoder: "CompactStateEncoder"
    history: list[dict]
    best_snapshot: dict | None
    best_q_table: np.ndarray


class CompactStateEncoder:
    def __init__(self):
        self.action_count = 5
        self.fatigue_bin_edges = np.array([0.33, 0.66], dtype=np.float32)
        self.preference_value_bin_edges = np.array([0.45, 0.70], dtype=np.float32)
        self.preference_gap_bin_edges = np.array([0.08, 0.20], dtype=np.float32)
        self.state_shape = (3, 3, 3, 3, 3, 5, 5, 3, 3, 11)

    def encode(
        self,
        observation: np.ndarray,
    ) -> tuple[int, int, int, int, int, int, int, int, int, int]:
        fatigue = observation[: self.action_count]
        preference = observation[self.action_count : self.action_count * 2]

        fatigue_buckets = tuple(
            int(
                np.digitize(
                    fatigue_value,
                    self.fatigue_bin_edges,
                    right=False,
                )
            )
            for fatigue_value in fatigue
        )
        preference_rank = np.argsort(preference)[::-1]
        best_preference_action = int(preference_rank[0])
        second_preference_action = int(preference_rank[1])
        best_preference_value_bucket = int(
            np.digitize(
                preference[best_preference_action],
                self.preference_value_bin_edges,
                right=False,
            )
        )
        preference_gap_bucket = int(
            np.digitize(
                preference[best_preference_action] - preference[second_preference_action],
                self.preference_gap_bin_edges,
                right=False,
            )
        )
        patience_bucket = int(np.clip(np.rint(observation[-1]), 0, 10))

        return fatigue_buckets + (
            best_preference_action,
            second_preference_action,
            best_preference_value_bucket,
            preference_gap_bucket,
            patience_bucket,
        )

    def to_dict(self) -> dict:
        total_states = int(np.prod(self.state_shape))
        return {
            "encoder_version": "v3",
            "state_shape": list(self.state_shape),
            "total_states": total_states,
            "fatigue_bin_edges": [float(x) for x in self.fatigue_bin_edges],
            "preference_value_bin_edges": [
                float(x) for x in self.preference_value_bin_edges
            ],
            "preference_gap_bin_edges": [
                float(x) for x in self.preference_gap_bin_edges
            ],
            "state_definition": [
                "fatigue_bucket_action_0",
                "fatigue_bucket_action_1",
                "fatigue_bucket_action_2",
                "fatigue_bucket_action_3",
                "fatigue_bucket_action_4",
                "best_preference_action",
                "second_preference_action",
                "best_preference_value_bucket",
                "preference_gap_bucket",
                "patience_bucket",
            ],
        }


class TabularQAgent:
    def __init__(
        self,
        state_shape: tuple[int, ...],
        action_count: int,
        alpha: float,
        gamma: float,
    ):
        self.action_count = action_count
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = np.zeros(state_shape + (action_count,), dtype=np.float32)

    def select_action(
        self,
        state: tuple[int, ...],
        epsilon: float,
        rng: np.random.Generator,
    ) -> int:
        if rng.random() < epsilon:
            return int(rng.integers(self.action_count))
        return self.greedy_action(state, rng)

    def greedy_action(
        self,
        state: tuple[int, ...],
        rng: np.random.Generator,
    ) -> int:
        q_values = self.q_table[state]
        best_value = np.max(q_values)
        candidates = np.flatnonzero(np.isclose(q_values, best_value))
        return int(rng.choice(candidates))

    def update(
        self,
        state: tuple[int, ...],
        action: int,
        reward: float,
        next_state: tuple[int, ...],
        done: bool,
    ) -> None:
        state_action_index = state + (action,)
        best_next_value = 0.0 if done else float(np.max(self.q_table[next_state]))
        td_target = reward + self.gamma * best_next_value
        td_error = td_target - float(self.q_table[state_action_index])
        self.q_table[state_action_index] += self.alpha * td_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a tabular Q-learning agent on the Micro-RecSim environment."
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
        "--alpha",
        type=float,
        default=0.1,
        help="Learning rate for Q-learning.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.95,
        help="Discount factor for Q-learning.",
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
        "--seed",
        type=int,
        default=42,
        help="Base random seed.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for saving Q-table and metrics.",
    )
    return parser.parse_args()


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
    encoder: CompactStateEncoder,
    agent: TabularQAgent,
    rng: np.random.Generator,
    episode_seed: int,
    max_steps: int,
    epsilon: float,
    training: bool,
) -> EpisodeStats:
    observation, _ = env.reset(seed=episode_seed)
    state = encoder.encode(observation)

    total_reward = 0.0
    clicks = 0
    action_counts = np.zeros(env.action_space.n, dtype=np.int32)

    for step_idx in range(max_steps):
        if training:
            action = agent.select_action(state, epsilon=epsilon, rng=rng)
        else:
            action = agent.greedy_action(state, rng=rng)

        action_counts[action] += 1

        next_observation, reward, terminated, truncated, info = env.step(action)
        next_state = encoder.encode(next_observation)
        done = terminated or truncated

        if training:
            agent.update(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
            )

        total_reward += reward
        clicks += int(info["clicked"])
        state = next_state

        if done:
            return EpisodeStats(
                total_reward=total_reward,
                steps=step_idx + 1,
                clicks=clicks,
                final_patience=float(next_observation[-1]),
                action_counts=action_counts,
            )

    return EpisodeStats(
        total_reward=total_reward,
        steps=max_steps,
        clicks=clicks,
        final_patience=float(next_observation[-1]),
        action_counts=action_counts,
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


def evaluate_agent(
    agent: TabularQAgent,
    encoder: CompactStateEncoder,
    episodes: int,
    max_steps: int,
    seed: int,
) -> dict:
    env = MicroRecSimEnv()
    rng = np.random.default_rng(seed)
    episode_stats = []

    for episode_idx in range(episodes):
        episode_seed = seed + episode_idx
        stats = run_episode(
            env=env,
            encoder=encoder,
            agent=agent,
            rng=rng,
            episode_seed=episode_seed,
            max_steps=max_steps,
            epsilon=0.0,
            training=False,
        )
        episode_stats.append(stats)

    return aggregate_episode_stats(episode_stats)


def train_agent(args: argparse.Namespace) -> TrainingResult:
    env = MicroRecSimEnv()
    encoder = CompactStateEncoder()
    agent = TabularQAgent(
        state_shape=encoder.state_shape,
        action_count=env.action_space.n,
        alpha=args.alpha,
        gamma=args.gamma,
    )
    rng = np.random.default_rng(args.seed)
    history = []
    best_snapshot = None
    best_q_table = agent.q_table.copy()

    for episode_idx in range(args.train_episodes):
        epsilon = epsilon_by_episode(
            episode_idx=episode_idx,
            total_episodes=args.train_episodes,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
        )
        episode_seed = args.seed + episode_idx
        train_stats = run_episode(
            env=env,
            encoder=encoder,
            agent=agent,
            rng=rng,
            episode_seed=episode_seed,
            max_steps=args.max_steps,
            epsilon=epsilon,
            training=True,
        )

        if args.eval_every > 0 and (
            (episode_idx + 1) % args.eval_every == 0
            or episode_idx == args.train_episodes - 1
        ):
            eval_summary = evaluate_agent(
                agent=agent,
                encoder=encoder,
                episodes=args.eval_episodes,
                max_steps=args.max_steps,
                seed=args.seed + 100000 + episode_idx,
            )
            snapshot = {
                "episode": episode_idx + 1,
                "epsilon": epsilon,
                "train_total_reward": train_stats.total_reward,
                "train_steps": train_stats.steps,
                "train_clicks": train_stats.clicks,
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
                best_q_table = agent.q_table.copy()

            best_eval_reward = (
                best_snapshot["eval"]["average_total_reward"]
                if best_snapshot is not None
                else float("-inf")
            )
            print(
                f"Episode {episode_idx + 1:>5} | "
                f"epsilon={epsilon:.3f} | "
                f"train_reward={train_stats.total_reward:>6.2f} | "
                f"eval_reward={eval_summary['average_total_reward']:>6.3f} | "
                f"best_eval_reward={best_eval_reward:>6.3f} | "
                f"eval_clicks={eval_summary['average_clicks']:>6.3f}"
            )

    return TrainingResult(
        agent=agent,
        encoder=encoder,
        history=history,
        best_snapshot=best_snapshot,
        best_q_table=best_q_table,
    )


def save_outputs(
    output_dir: Path,
    args: argparse.Namespace,
    result: TrainingResult,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    final_q_table_path = output_dir / "q_table_final.npy"
    best_q_table_path = output_dir / "q_table_best.npy"
    metrics_path = output_dir / "training_metrics.json"

    np.save(final_q_table_path, result.agent.q_table)
    np.save(best_q_table_path, result.best_q_table)

    final_eval = result.history[-1]["eval"] if result.history else {}
    metrics = {
        "config": {
            "train_episodes": args.train_episodes,
            "eval_episodes": args.eval_episodes,
            "max_steps": args.max_steps,
            "alpha": args.alpha,
            "gamma": args.gamma,
            "epsilon_start": args.epsilon_start,
            "epsilon_end": args.epsilon_end,
            "eval_every": args.eval_every,
            "seed": args.seed,
        },
        "encoder": result.encoder.to_dict(),
        "q_table_shape": list(result.agent.q_table.shape),
        "final_eval": final_eval,
        "best_snapshot": result.best_snapshot,
        "history": result.history,
    }

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    print(f"\nSaved final Q-table to: {final_q_table_path}")
    print(f"Saved best Q-table to: {best_q_table_path}")
    print(f"Saved training metrics to: {metrics_path}")


def main() -> None:
    args = parse_args()
    result = train_agent(args)

    print("\nState encoder summary:")
    print(json.dumps(result.encoder.to_dict(), ensure_ascii=False, indent=2))

    if result.history:
        print("\nBest evaluation snapshot:")
        print(json.dumps(result.best_snapshot, ensure_ascii=False, indent=2))
        print("\nFinal evaluation summary:")
        print(json.dumps(result.history[-1]["eval"], ensure_ascii=False, indent=2))

    save_outputs(
        output_dir=Path(args.output_dir),
        args=args,
        result=result,
    )


if __name__ == "__main__":
    main()
