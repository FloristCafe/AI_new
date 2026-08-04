from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


DEFAULT_EXPERIMENTS_ROOT = Path(
    r"D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_dqn_offline_rl\artifacts\experiments"
)
DEFAULT_OUTPUT_DIR = (
    DEFAULT_EXPERIMENTS_ROOT / "summaries"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize multi-seed SASRec-DQN experiment results."
    )
    parser.add_argument(
        "--experiment-dirs",
        type=str,
        nargs="+",
        required=True,
        help="Experiment directories to aggregate.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for JSON and CSV summaries.",
    )
    parser.add_argument(
        "--summary-name",
        type=str,
        default="multi_seed_summary",
        help="Base filename used for output artifacts.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open("r", encoding="utf-8") as fin:
        return json.load(fin)


def compute_mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": math.nan, "std": math.nan}

    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return {
        "mean": mean_value,
        "std": math.sqrt(variance),
    }


def collect_run_row(experiment_dir: Path) -> dict[str, object]:
    metrics_path = experiment_dir / "metrics" / "training_metrics.json"
    test_path = experiment_dir / "predictions" / "test_sasrec_dqn_metrics.json"
    all_path = experiment_dir / "predictions" / "all_sasrec_dqn_metrics.json"

    training_metrics = load_json(metrics_path)
    test_metrics = load_json(test_path)
    all_metrics = load_json(all_path)

    return {
        "run_name": experiment_dir.name,
        "run_dir": str(experiment_dir),
        "seed": int(training_metrics["seed"]),
        "best_epoch": int(training_metrics["best_epoch"]),
        "best_valid_ndcg_at_10": float(training_metrics["best_selection_metric_value"]),
        "test_hr_at_10": float(test_metrics["hr_at_k"]),
        "test_ndcg_at_10": float(test_metrics["ndcg_at_k"]),
        "test_top1_exact_hit_rate": float(test_metrics["top1_exact_hit_rate"]),
        "all_hr_at_10": float(all_metrics["hr_at_k"]),
        "all_ndcg_at_10": float(all_metrics["ndcg_at_k"]),
        "all_top1_exact_hit_rate": float(all_metrics["top1_exact_hit_rate"]),
        "all_mean_cumulative_reward_per_user": float(
            all_metrics["mean_cumulative_reward_per_user"]
        ),
    }


def build_aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    metric_names = [
        "best_valid_ndcg_at_10",
        "test_hr_at_10",
        "test_ndcg_at_10",
        "test_top1_exact_hit_rate",
        "all_hr_at_10",
        "all_ndcg_at_10",
        "all_top1_exact_hit_rate",
        "all_mean_cumulative_reward_per_user",
    ]

    aggregate_metrics: dict[str, dict[str, float]] = {}
    for metric_name in metric_names:
        values = [float(row[metric_name]) for row in rows]
        aggregate_metrics[metric_name] = compute_mean_std(values)

    return {
        "run_count": len(rows),
        "seeds": [int(row["seed"]) for row in rows],
        "metrics": aggregate_metrics,
    }


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No rows available for CSV export.")

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    experiment_dirs = [Path(path) for path in args.experiment_dirs]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [collect_run_row(experiment_dir) for experiment_dir in experiment_dirs]
    rows.sort(key=lambda row: int(row["seed"]))

    summary = {
        "summary_name": args.summary_name,
        "experiment_dirs": [str(path) for path in experiment_dirs],
        "rows": rows,
        "aggregate": build_aggregate(rows),
    }

    json_path = output_dir / f"{args.summary_name}.json"
    csv_path = output_dir / f"{args.summary_name}.csv"

    with json_path.open("w", encoding="utf-8") as fout:
        json.dump(summary, fout, ensure_ascii=False, indent=2)
    save_csv(csv_path, rows)

    print(f"Summary JSON saved to: {json_path}")
    print(f"Summary CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
