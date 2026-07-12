import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "artifacts" / "dqn_multiseed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run train_dqn.py across multiple seeds and summarize best/final results."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[7, 42, 123],
        help="Seeds to evaluate.",
    )
    parser.add_argument(
        "--train-episodes",
        type=int,
        default=2000,
        help="Training episodes passed to train_dqn.py.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=500,
        help="Evaluation episodes passed to train_dqn.py.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=100,
        help="Evaluation interval passed to train_dqn.py.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden width for DQN.",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "silu"],
        help="Activation for DQN.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate passed to train_dqn.py.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.95,
        help="Discount factor passed to train_dqn.py.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size passed to train_dqn.py.",
    )
    parser.add_argument(
        "--replay-buffer-capacity",
        type=int,
        default=20000,
        help="Replay buffer capacity passed to train_dqn.py.",
    )
    parser.add_argument(
        "--min-buffer-size",
        type=int,
        default=500,
        help="Minimum warmup buffer size passed to train_dqn.py.",
    )
    parser.add_argument(
        "--target-sync-interval",
        type=int,
        default=200,
        help="Target sync interval passed to train_dqn.py.",
    )
    parser.add_argument(
        "--double-dqn",
        action="store_true",
        help="Pass Double DQN mode to train_dqn.py.",
    )
    parser.add_argument(
        "--epsilon-start",
        type=float,
        default=1.0,
        help="Starting epsilon passed to train_dqn.py.",
    )
    parser.add_argument(
        "--epsilon-end",
        type=float,
        default=0.05,
        help="Final epsilon passed to train_dqn.py.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Early stopping patience passed to train_dqn.py.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device passed to train_dqn.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for per-seed outputs and summary JSON.",
    )
    return parser.parse_args()


def run_seed(args: argparse.Namespace, seed: int, train_script: Path, output_root: Path) -> dict:
    seed_output_dir = output_root / f"seed_{seed}"
    seed_output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(train_script),
        "--train-episodes",
        str(args.train_episodes),
        "--eval-episodes",
        str(args.eval_episodes),
        "--eval-every",
        str(args.eval_every),
        "--hidden-dim",
        str(args.hidden_dim),
        "--activation",
        args.activation,
        "--learning-rate",
        str(args.learning_rate),
        "--gamma",
        str(args.gamma),
        "--batch-size",
        str(args.batch_size),
        "--replay-buffer-capacity",
        str(args.replay_buffer_capacity),
        "--min-buffer-size",
        str(args.min_buffer_size),
        "--target-sync-interval",
        str(args.target_sync_interval),
        *(["--double-dqn"] if args.double_dqn else []),
        "--epsilon-start",
        str(args.epsilon_start),
        "--epsilon-end",
        str(args.epsilon_end),
        "--early-stop-patience",
        str(args.early_stop_patience),
        "--seed",
        str(seed),
        "--device",
        args.device,
        "--output-dir",
        str(seed_output_dir),
    ]

    print(f"\nRunning seed {seed}")
    print(" ".join(command))
    subprocess.run(command, check=True)

    metrics_path = seed_output_dir / "training_metrics.json"
    with metrics_path.open("r", encoding="utf-8") as file:
        metrics = json.load(file)

    best_snapshot = metrics.get("best_snapshot", {})
    best_eval = best_snapshot.get("eval", {})
    final_eval = metrics.get("final_eval", {})

    result = {
        "seed": seed,
        "best_episode": best_snapshot.get("episode"),
        "best_average_total_reward": best_eval.get("average_total_reward"),
        "best_average_clicks": best_eval.get("average_clicks"),
        "final_average_total_reward": final_eval.get("average_total_reward"),
        "final_average_clicks": final_eval.get("average_clicks"),
        "metrics_path": str(metrics_path),
    }
    return result


def summarize_results(results: list[dict]) -> dict:
    best_rewards = [result["best_average_total_reward"] for result in results]
    best_clicks = [result["best_average_clicks"] for result in results]
    final_rewards = [result["final_average_total_reward"] for result in results]
    final_clicks = [result["final_average_clicks"] for result in results]

    def mean(values: list[float]) -> float:
        return float(sum(values) / len(values))

    def std(values: list[float]) -> float:
        avg = mean(values)
        return float((sum((value - avg) ** 2 for value in values) / len(values)) ** 0.5)

    return {
        "seed_count": len(results),
        "best_reward_mean": mean(best_rewards),
        "best_reward_std": std(best_rewards),
        "best_clicks_mean": mean(best_clicks),
        "best_clicks_std": std(best_clicks),
        "final_reward_mean": mean(final_rewards),
        "final_reward_std": std(final_rewards),
        "final_clicks_mean": mean(final_clicks),
        "final_clicks_std": std(final_clicks),
    }


def print_summary(results: list[dict], summary: dict) -> None:
    print("\nPer-seed summary")
    print("Seed   BestEp   BestReward   BestClicks   FinalReward   FinalClicks")
    print("-" * 68)
    for result in results:
        print(
            f"{result['seed']:>4}   "
            f"{str(result['best_episode']):>6}   "
            f"{result['best_average_total_reward']:>10.3f}   "
            f"{result['best_average_clicks']:>10.3f}   "
            f"{result['final_average_total_reward']:>11.3f}   "
            f"{result['final_average_clicks']:>11.3f}"
        )

    print("\nAggregate summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    train_script = Path(__file__).resolve().parent / "train_dqn.py"

    results = []
    for seed in args.seeds:
        result = run_seed(args=args, seed=seed, train_script=train_script, output_root=output_root)
        results.append(result)

    summary = summarize_results(results)
    print_summary(results, summary)

    summary_path = output_root / "multiseed_summary.json"
    payload = {
        "config": {
            "seeds": args.seeds,
            "train_episodes": args.train_episodes,
            "eval_episodes": args.eval_episodes,
            "eval_every": args.eval_every,
            "hidden_dim": args.hidden_dim,
            "activation": args.activation,
            "learning_rate": args.learning_rate,
            "gamma": args.gamma,
            "batch_size": args.batch_size,
            "replay_buffer_capacity": args.replay_buffer_capacity,
            "min_buffer_size": args.min_buffer_size,
            "target_sync_interval": args.target_sync_interval,
            "double_dqn": args.double_dqn,
            "epsilon_start": args.epsilon_start,
            "epsilon_end": args.epsilon_end,
            "early_stop_patience": args.early_stop_patience,
            "device": args.device,
        },
        "results": results,
        "summary": summary,
    }
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    print(f"\nSaved multi-seed summary to: {summary_path}")


if __name__ == "__main__":
    main()
