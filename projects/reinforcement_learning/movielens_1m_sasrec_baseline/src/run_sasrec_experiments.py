import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = Path(
    r"D:\Python\Datasets\movielens_1m\processed\interactions.csv"
)
DEFAULT_EXPERIMENTS_DIR = PROJECT_ROOT / "artifacts" / "experiments"


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    max_seq_len: int
    embedding_dim: int
    num_heads: int
    num_blocks: int
    dropout: float
    learning_rate: float
    weight_decay: float
    batch_size: int = 256
    eval_batch_size: int = 256
    epochs: int = 20
    early_stop_patience: int = 3
    selection_metric: str = "ndcg_at_10"
    loss_type: str = "cross_entropy"
    num_negative_samples: int = 1
    seed: int = 42


PRESET_CONFIGS: dict[str, list[ExperimentConfig]] = {
    "baseline_triplet": [
        ExperimentConfig(
            name="baseline_seq50_dim64_drop02",
            max_seq_len=50,
            embedding_dim=64,
            num_heads=2,
            num_blocks=2,
            dropout=0.2,
            learning_rate=1e-3,
            weight_decay=1e-5,
        ),
        ExperimentConfig(
            name="seq100_dim64_drop02",
            max_seq_len=100,
            embedding_dim=64,
            num_heads=2,
            num_blocks=2,
            dropout=0.2,
            learning_rate=1e-3,
            weight_decay=1e-5,
        ),
        ExperimentConfig(
            name="seq50_dim128_drop02",
            max_seq_len=50,
            embedding_dim=128,
            num_heads=2,
            num_blocks=2,
            dropout=0.2,
            learning_rate=1e-3,
            weight_decay=1e-5,
        ),
    ],
    "capacity_and_regularization": [
        ExperimentConfig(
            name="baseline_seq50_dim64_drop02",
            max_seq_len=50,
            embedding_dim=64,
            num_heads=2,
            num_blocks=2,
            dropout=0.2,
            learning_rate=1e-3,
            weight_decay=1e-5,
        ),
        ExperimentConfig(
            name="seq50_dim64_drop05",
            max_seq_len=50,
            embedding_dim=64,
            num_heads=2,
            num_blocks=2,
            dropout=0.5,
            learning_rate=1e-3,
            weight_decay=1e-5,
        ),
        ExperimentConfig(
            name="seq50_dim128_drop02",
            max_seq_len=50,
            embedding_dim=128,
            num_heads=2,
            num_blocks=2,
            dropout=0.2,
            learning_rate=1e-3,
            weight_decay=1e-5,
        ),
        ExperimentConfig(
            name="seq50_dim128_drop05",
            max_seq_len=50,
            embedding_dim=128,
            num_heads=2,
            num_blocks=2,
            dropout=0.5,
            learning_rate=1e-3,
            weight_decay=1e-5,
        ),
    ],
    "dim128_finegrained_structure": [
        ExperimentConfig(
            name="seq50_dim128_drop02",
            max_seq_len=50,
            embedding_dim=128,
            num_heads=2,
            num_blocks=2,
            dropout=0.2,
            learning_rate=1e-3,
            weight_decay=1e-5,
        ),
        ExperimentConfig(
            name="seq50_dim128_drop03",
            max_seq_len=50,
            embedding_dim=128,
            num_heads=2,
            num_blocks=2,
            dropout=0.3,
            learning_rate=1e-3,
            weight_decay=1e-5,
        ),
        ExperimentConfig(
            name="seq50_dim128_drop04",
            max_seq_len=50,
            embedding_dim=128,
            num_heads=2,
            num_blocks=2,
            dropout=0.4,
            learning_rate=1e-3,
            weight_decay=1e-5,
        ),
        ExperimentConfig(
            name="seq50_dim128_blocks3_drop02",
            max_seq_len=50,
            embedding_dim=128,
            num_heads=2,
            num_blocks=3,
            dropout=0.2,
            learning_rate=1e-3,
            weight_decay=1e-5,
        ),
    ],
    "objective_alignment": [
        ExperimentConfig(
            name="seq50_dim128_blocks3_drop02_ce",
            max_seq_len=50,
            embedding_dim=128,
            num_heads=2,
            num_blocks=3,
            dropout=0.2,
            learning_rate=1e-3,
            weight_decay=1e-5,
            loss_type="cross_entropy",
        ),
        ExperimentConfig(
            name="seq50_dim128_blocks3_drop02_bce_ns1",
            max_seq_len=50,
            embedding_dim=128,
            num_heads=2,
            num_blocks=3,
            dropout=0.2,
            learning_rate=1e-3,
            weight_decay=1e-5,
            loss_type="bce_negative_sampling",
            num_negative_samples=1,
        ),
    ],
    "bce_repair_round1": [
        ExperimentConfig(
            name="seq50_dim128_blocks3_drop02_ce_reference",
            max_seq_len=50,
            embedding_dim=128,
            num_heads=2,
            num_blocks=3,
            dropout=0.2,
            learning_rate=1e-3,
            weight_decay=1e-5,
            epochs=20,
            early_stop_patience=3,
            loss_type="cross_entropy",
        ),
        ExperimentConfig(
            name="seq50_dim128_blocks3_drop02_bce_ns5_lr5e4",
            max_seq_len=50,
            embedding_dim=128,
            num_heads=2,
            num_blocks=3,
            dropout=0.2,
            learning_rate=5e-4,
            weight_decay=1e-5,
            epochs=30,
            early_stop_patience=5,
            loss_type="bce_negative_sampling",
            num_negative_samples=5,
        ),
        ExperimentConfig(
            name="seq50_dim128_blocks3_drop02_bce_ns10_lr5e4",
            max_seq_len=50,
            embedding_dim=128,
            num_heads=2,
            num_blocks=3,
            dropout=0.2,
            learning_rate=5e-4,
            weight_decay=1e-5,
            epochs=30,
            early_stop_patience=5,
            loss_type="bce_negative_sampling",
            num_negative_samples=10,
        ),
        ExperimentConfig(
            name="seq50_dim128_blocks3_drop02_bce_ns10_lr1e4",
            max_seq_len=50,
            embedding_dim=128,
            num_heads=2,
            num_blocks=3,
            dropout=0.2,
            learning_rate=1e-4,
            weight_decay=1e-5,
            epochs=30,
            early_stop_patience=5,
            loss_type="bce_negative_sampling",
            num_negative_samples=10,
        ),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiple SASRec experiments and summarize the results."
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="baseline_triplet",
        choices=sorted(PRESET_CONFIGS),
        help="Named experiment preset.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=str(DEFAULT_DATASET_PATH),
        help="Path to the cleaned MovieLens interaction CSV.",
    )
    parser.add_argument(
        "--experiments-dir",
        type=str,
        default=str(DEFAULT_EXPERIMENTS_DIR),
        help="Root directory for experiment artifacts.",
    )
    parser.add_argument(
        "--python-executable",
        type=str,
        default=sys.executable,
        help="Python executable used to launch preprocess/train/evaluate scripts.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Torch device passed to training and evaluation scripts.",
    )
    parser.add_argument(
        "--force-preprocess",
        action="store_true",
        help="Rebuild preprocessed sequence artifacts even if matching metadata already exists.",
    )
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Retrain runs even if training metrics already exist.",
    )
    parser.add_argument(
        "--force-eval",
        action="store_true",
        help="Re-run test evaluation even if test metrics already exist.",
    )
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    print("Running command:")
    print(" ".join(f'"{arg}"' if " " in arg else arg for arg in command))
    subprocess.run(command, check=True)


def build_preprocess_dir(experiments_dir: Path, config: ExperimentConfig) -> Path:
    return experiments_dir / "prepared_data" / f"seq_len_{config.max_seq_len}"


def build_run_dir(experiments_dir: Path, config: ExperimentConfig) -> Path:
    return experiments_dir / config.name


def metadata_matches(metadata_path: Path, config: ExperimentConfig) -> bool:
    if not metadata_path.exists():
        return False

    with metadata_path.open("r", encoding="utf-8") as fin:
        metadata = json.load(fin)

    if int(metadata.get("max_seq_len", -1)) != config.max_seq_len:
        return False

    preprocess_dir = metadata_path.parent
    required_files = [
        preprocess_dir / "train_sequences.npz",
        preprocess_dir / "train_sequence_supervision.npz",
        preprocess_dir / "valid_sequences.npz",
        preprocess_dir / "test_sequences.npz",
    ]
    return all(path.exists() for path in required_files)


def maybe_run_preprocess(
    args: argparse.Namespace,
    config: ExperimentConfig,
    experiments_dir: Path,
) -> Path:
    preprocess_dir = build_preprocess_dir(experiments_dir, config)
    metadata_path = preprocess_dir / "metadata.json"

    if not args.force_preprocess and metadata_matches(metadata_path, config):
        print(f"Reusing preprocessed data: {preprocess_dir}")
        return preprocess_dir

    preprocess_dir.mkdir(parents=True, exist_ok=True)
    command = [
        args.python_executable,
        str(SRC_DIR / "preprocess_movielens_1m.py"),
        "--input-path",
        str(Path(args.dataset_path)),
        "--output-dir",
        str(preprocess_dir),
        "--max-seq-len",
        str(config.max_seq_len),
    ]
    run_command(command)
    return preprocess_dir


def maybe_run_training(
    args: argparse.Namespace,
    config: ExperimentConfig,
    preprocess_dir: Path,
    run_dir: Path,
) -> Path:
    metrics_path = run_dir / "metrics" / "training_metrics.json"
    if not args.force_train and metrics_path.exists():
        print(f"Reusing training metrics: {metrics_path}")
        return metrics_path

    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        args.python_executable,
        str(SRC_DIR / "train_sasrec.py"),
        "--data-dir",
        str(preprocess_dir),
        "--output-dir",
        str(run_dir),
        "--epochs",
        str(config.epochs),
        "--batch-size",
        str(config.batch_size),
        "--eval-batch-size",
        str(config.eval_batch_size),
        "--learning-rate",
        str(config.learning_rate),
        "--weight-decay",
        str(config.weight_decay),
        "--embedding-dim",
        str(config.embedding_dim),
        "--num-heads",
        str(config.num_heads),
        "--num-blocks",
        str(config.num_blocks),
        "--dropout",
        str(config.dropout),
        "--selection-metric",
        config.selection_metric,
        "--early-stop-patience",
        str(config.early_stop_patience),
        "--loss-type",
        config.loss_type,
        "--num-negative-samples",
        str(config.num_negative_samples),
        "--device",
        args.device,
        "--seed",
        str(config.seed),
    ]
    run_command(command)
    return metrics_path


def maybe_run_evaluation(
    args: argparse.Namespace,
    config: ExperimentConfig,
    preprocess_dir: Path,
    run_dir: Path,
) -> Path:
    result_path = run_dir / "predictions" / "test_metrics.json"
    if not args.force_eval and result_path.exists():
        print(f"Reusing test metrics: {result_path}")
        return result_path

    checkpoint_path = run_dir / "checkpoints" / "sasrec_best.pt"
    command = [
        args.python_executable,
        str(SRC_DIR / "evaluate_sasrec.py"),
        "--data-dir",
        str(preprocess_dir),
        "--checkpoint-path",
        str(checkpoint_path),
        "--output-dir",
        str(run_dir / "predictions"),
        "--split",
        "test",
        "--batch-size",
        str(config.eval_batch_size),
        "--embedding-dim",
        str(config.embedding_dim),
        "--num-heads",
        str(config.num_heads),
        "--num-blocks",
        str(config.num_blocks),
        "--dropout",
        str(config.dropout),
        "--device",
        args.device,
    ]
    run_command(command)
    return result_path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fin:
        return json.load(fin)


def extract_best_epoch_metrics(training_metrics: dict) -> dict:
    best_epoch = int(training_metrics["best_epoch"])
    for epoch_summary in training_metrics["epoch_metrics"]:
        if int(epoch_summary["epoch"]) == best_epoch:
            return epoch_summary
    raise ValueError(f"Best epoch {best_epoch} not found in epoch metrics.")


def build_summary_row(
    config: ExperimentConfig,
    preprocess_dir: Path,
    run_dir: Path,
    training_metrics: dict,
    test_metrics: dict,
) -> dict[str, object]:
    best_epoch_metrics = extract_best_epoch_metrics(training_metrics)
    return {
        "run_name": config.name,
        "preprocess_dir": str(preprocess_dir),
        "run_dir": str(run_dir),
        "max_seq_len": config.max_seq_len,
        "embedding_dim": config.embedding_dim,
        "num_heads": config.num_heads,
        "num_blocks": config.num_blocks,
        "dropout": config.dropout,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "batch_size": config.batch_size,
        "epochs_requested": config.epochs,
        "loss_type": config.loss_type,
        "num_negative_samples": config.num_negative_samples,
        "epochs_completed": training_metrics["epochs_completed"],
        "stopped_early": training_metrics["stopped_early"],
        "best_epoch": training_metrics["best_epoch"],
        "best_valid_hr_at_10": best_epoch_metrics["valid_hr_at_10"],
        "best_valid_ndcg_at_10": best_epoch_metrics["valid_ndcg_at_10"],
        "test_hr_at_10": test_metrics["hr_at_10"],
        "test_ndcg_at_10": test_metrics["ndcg_at_10"],
        "best_checkpoint_path": training_metrics["best_checkpoint_path"],
    }


def save_summary(rows: list[dict[str, object]], output_dir: Path, preset: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json_path = output_dir / f"{preset}_summary.json"
    summary_csv_path = output_dir / f"{preset}_summary.csv"

    payload = {
        "preset": preset,
        "run_count": len(rows),
        "rows": rows,
    }
    with summary_json_path.open("w", encoding="utf-8") as fout:
        json.dump(payload, fout, ensure_ascii=False, indent=2)

    if rows:
        with summary_csv_path.open("w", encoding="utf-8", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"Summary saved to: {summary_json_path}")
    print(f"CSV summary saved to: {summary_csv_path}")


def main() -> None:
    args = parse_args()
    experiments_dir = Path(args.experiments_dir)
    summary_dir = experiments_dir / "summaries"
    rows: list[dict[str, object]] = []

    for config in PRESET_CONFIGS[args.preset]:
        print("")
        print(f"=== Running experiment: {config.name} ===")
        preprocess_dir = maybe_run_preprocess(args, config, experiments_dir)
        run_dir = build_run_dir(experiments_dir, config)
        training_metrics_path = maybe_run_training(args, config, preprocess_dir, run_dir)
        test_metrics_path = maybe_run_evaluation(args, config, preprocess_dir, run_dir)

        training_metrics = load_json(training_metrics_path)
        test_metrics = load_json(test_metrics_path)
        rows.append(
            build_summary_row(
                config=config,
                preprocess_dir=preprocess_dir,
                run_dir=run_dir,
                training_metrics=training_metrics,
                test_metrics=test_metrics,
            )
        )

    save_summary(rows=rows, output_dir=summary_dir, preset=args.preset)


if __name__ == "__main__":
    main()
