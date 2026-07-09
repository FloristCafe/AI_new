import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from micro_recsim_env import MicroRecSimEnv


@dataclass
class EpisodeStats:
    total_reward: float
    steps: int
    clicks: int
    final_patience: float
    action_counts: np.ndarray


class RandomPolicy:
    def __init__(self, rng: np.random.Generator):
        self.name = "random"
        self._rng = rng

    def reset_episode(self) -> None:
        pass

    def select_action(self, observation: np.ndarray, env: MicroRecSimEnv) -> int:
        return int(self._rng.integers(env.action_space.n))


class AlwaysSamePolicy:
    def __init__(self, action: int):
        self.name = f"always_same_{action}"
        self._action = action

    def reset_episode(self) -> None:
        pass

    def select_action(self, observation: np.ndarray, env: MicroRecSimEnv) -> int:
        return self._action


class RoundRobinPolicy:
    def __init__(self, start_action: int = 0):
        self.name = "round_robin"
        self._start_action = start_action
        self._next_action = start_action

    def reset_episode(self) -> None:
        self._next_action = self._start_action

    def select_action(self, observation: np.ndarray, env: MicroRecSimEnv) -> int:
        action = self._next_action
        self._next_action = (self._next_action + 1) % env.action_space.n
        return action


class LeastFatiguePolicy:
    def __init__(self, rng: np.random.Generator):
        self.name = "least_fatigue"
        self._rng = rng

    def reset_episode(self) -> None:
        pass

    def select_action(self, observation: np.ndarray, env: MicroRecSimEnv) -> int:
        fatigue = observation[: env.action_space.n]
        min_fatigue = np.min(fatigue)
        candidates = np.flatnonzero(np.isclose(fatigue, min_fatigue))
        return int(self._rng.choice(candidates))


class OraclePreferenceGreedyPolicy:
    def __init__(self, rng: np.random.Generator):
        self.name = "myopic_oracle_preference_greedy"
        self._rng = rng

    def reset_episode(self) -> None:
        pass

    def select_action(self, observation: np.ndarray, env: MicroRecSimEnv) -> int:
        current_fatigue = observation[: env.action_space.n]
        click_scores = env._true_preference * (1.0 - current_fatigue)
        best_score = np.max(click_scores)
        candidates = np.flatnonzero(np.isclose(click_scores, best_score))
        return int(self._rng.choice(candidates))


class ObservableClickGreedyPolicy:
    def __init__(self, rng: np.random.Generator):
        self.name = "observable_click_greedy"
        self._rng = rng

    def reset_episode(self) -> None:
        pass

    def select_action(self, observation: np.ndarray, env: MicroRecSimEnv) -> int:
        fatigue = observation[: env.action_space.n]
        preference = observation[env.action_space.n : env.action_space.n * 2]
        click_scores = preference * (1.0 - fatigue)
        best_score = np.max(click_scores)
        candidates = np.flatnonzero(np.isclose(click_scores, best_score))
        return int(self._rng.choice(candidates))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline policies on the Micro-RecSim environment."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=200,
        help="Number of episodes to evaluate for each baseline policy.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum number of steps allowed in a single episode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed used to generate reproducible user episodes.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="",
        help="Optional JSON path for saving aggregated baseline metrics.",
    )
    return parser.parse_args()


def build_policies(seed: int) -> list:
    return [
        RandomPolicy(rng=np.random.default_rng(seed)),
        AlwaysSamePolicy(action=0),
        RoundRobinPolicy(),
        LeastFatiguePolicy(rng=np.random.default_rng(seed + 1)),
        OraclePreferenceGreedyPolicy(rng=np.random.default_rng(seed + 2)),
        ObservableClickGreedyPolicy(rng=np.random.default_rng(seed + 3)),
    ]


def run_episode(
    env: MicroRecSimEnv,
    policy,
    episode_seed: int,
    max_steps: int,
) -> EpisodeStats:
    observation, _ = env.reset(seed=episode_seed)
    policy.reset_episode()

    total_reward = 0.0
    clicks = 0
    action_counts = np.zeros(env.action_space.n, dtype=np.int32)

    for step_idx in range(max_steps):
        action = policy.select_action(observation, env)
        action_counts[action] += 1

        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        clicks += int(info["clicked"])

        if terminated or truncated:
            return EpisodeStats(
                total_reward=total_reward,
                steps=step_idx + 1,
                clicks=clicks,
                final_patience=float(observation[-1]),
                action_counts=action_counts,
            )

    return EpisodeStats(
        total_reward=total_reward,
        steps=max_steps,
        clicks=clicks,
        final_patience=float(observation[-1]),
        action_counts=action_counts,
    )


def aggregate_results(policy_name: str, episode_stats: list[EpisodeStats]) -> dict:
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
        "policy": policy_name,
        "average_total_reward": float(np.mean(rewards)),
        "average_episode_length": float(np.mean(steps)),
        "average_clicks": float(np.mean(clicks)),
        "average_final_patience": float(np.mean(final_patience)),
        "action_frequency": [float(x) for x in action_frequency],
    }


def evaluate_policy(policy, episodes: int, max_steps: int, seed: int) -> dict:
    env = MicroRecSimEnv()
    episode_stats = []

    for episode_idx in range(episodes):
        episode_seed = seed + episode_idx
        stats = run_episode(
            env=env,
            policy=policy,
            episode_seed=episode_seed,
            max_steps=max_steps,
        )
        episode_stats.append(stats)

    return aggregate_results(policy.name, episode_stats)


def print_results(results: list[dict]) -> None:
    print(
        "Policy                AvgReward   AvgSteps   AvgClicks   AvgFinalPatience   ActionFreq"
    )
    print("-" * 95)
    for result in results:
        action_freq = ", ".join(f"{freq:.2f}" for freq in result["action_frequency"])
        print(
            f"{result['policy']:<20} "
            f"{result['average_total_reward']:>9.3f}   "
            f"{result['average_episode_length']:>8.3f}   "
            f"{result['average_clicks']:>9.3f}   "
            f"{result['average_final_patience']:>17.3f}   "
            f"[{action_freq}]"
        )


def maybe_save_results(results: list[dict], output_path: str) -> None:
    if not output_path:
        return

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    print(f"\nSaved results to: {path}")


def main() -> None:
    args = parse_args()
    policies = build_policies(seed=args.seed)
    results = [
        evaluate_policy(
            policy=policy,
            episodes=args.episodes,
            max_steps=args.max_steps,
            seed=args.seed,
        )
        for policy in policies
    ]
    print_results(results)
    maybe_save_results(results, args.output_path)


if __name__ == "__main__":
    main()
