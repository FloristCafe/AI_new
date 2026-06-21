import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_SRC = Path(
    r"D:\Python\Artificial Intelligence\projects\incubator\optiver_realized_volatility_starter\src"
)
DEFAULT_DATA_ROOT = Path(
    r"D:\Python\Datasets\optiver_realized_volatility_prediction\raw_extracted"
)
DEFAULT_SAMPLE_ROOT = Path(
    r"D:\Python\Datasets\optiver_realized_volatility_prediction\samples"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Optiver sandbox pipeline: build a larger sample, generate v2 "
            "features, optionally add KNN features, and train a baseline model."
        )
    )
    parser.add_argument(
        "--python-exe",
        type=str,
        default=sys.executable,
        help="Python executable used to run the sub-scripts.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(DEFAULT_DATA_ROOT),
        help="Directory containing train.csv, book_train.parquet and trade_train.parquet.",
    )
    parser.add_argument(
        "--sample-root",
        type=str,
        default=str(DEFAULT_SAMPLE_ROOT),
        help="Root directory for generated sandbox samples.",
    )
    parser.add_argument(
        "--stock-count",
        type=int,
        default=16,
        help="Number of stock_ids to include when --stock-ids is not provided.",
    )
    parser.add_argument(
        "--time-ids-per-stock",
        type=int,
        default=160,
        help="Number of train rows to keep per stock.",
    )
    parser.add_argument(
        "--stock-ids",
        type=str,
        default="",
        help="Optional explicit comma-separated stock_id list.",
    )
    parser.add_argument(
        "--sort-order",
        type=str,
        default="time_id",
        choices=["time_id", "target"],
        help="How to select rows inside each stock for the sandbox.",
    )
    parser.add_argument(
        "--use-knn",
        action="store_true",
        help="If set, add KNN features on top of the v2 feature table.",
    )
    parser.add_argument(
        "--global-k",
        type=int,
        default=5,
        help="Global K for KNN feature generation.",
    )
    parser.add_argument(
        "--same-stock-k",
        type=int,
        default=3,
        help="Same-stock K for KNN feature generation.",
    )
    parser.add_argument(
        "--min-history",
        type=int,
        default=20,
        help="Minimum history required before computing KNN features.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lightgbm",
        choices=["ridge", "random_forest", "lightgbm", "mlp", "mlp_lightgbm_ensemble"],
        help="Model used in the final training step.",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.2,
        help="Validation ratio for the final training script.",
    )
    parser.add_argument(
        "--lgbm-num-leaves",
        type=int,
        default=15,
        help="LightGBM num_leaves.",
    )
    parser.add_argument(
        "--lgbm-max-depth",
        type=int,
        default=4,
        help="LightGBM max_depth.",
    )
    parser.add_argument(
        "--lgbm-min-child-samples",
        type=int,
        default=40,
        help="LightGBM min_child_samples.",
    )
    parser.add_argument(
        "--lgbm-learning-rate",
        type=float,
        default=0.03,
        help="LightGBM learning_rate.",
    )
    parser.add_argument(
        "--lgbm-n-estimators",
        type=int,
        default=400,
        help="LightGBM n_estimators.",
    )
    parser.add_argument(
        "--mlp-hidden-layers",
        type=str,
        default="128,64",
        help="Comma-separated hidden layer sizes for MLPRegressor.",
    )
    parser.add_argument(
        "--mlp-alpha",
        type=float,
        default=1e-4,
        help="L2 regularization strength for MLPRegressor.",
    )
    parser.add_argument(
        "--mlp-learning-rate-init",
        type=float,
        default=1e-3,
        help="Initial learning rate for MLPRegressor.",
    )
    parser.add_argument(
        "--mlp-max-iter",
        type=int,
        default=500,
        help="Maximum iterations for MLPRegressor.",
    )
    parser.add_argument(
        "--ensemble-lgbm-weight",
        type=float,
        default=0.6,
        help="Prediction weight for LightGBM in the MLP + LightGBM ensemble.",
    )
    return parser.parse_args()


def run_step(command: list[str]) -> None:
    print(f"[RUN] {' '.join(command)}")
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()

    stock_tag = args.stock_ids.replace(",", "-") if args.stock_ids else None
    if not stock_tag:
        stock_tag = "-".join(str(i) for i in range(args.stock_count))

    sample_dir = Path(args.sample_root) / (
        f"optiver_sandbox_stocks_{stock_tag}_times_{args.time_ids_per_stock}"
    )

    build_sandbox_cmd = [
        args.python_exe,
        str(PROJECT_SRC / "build_optiver_sandbox.py"),
        "--data-root",
        args.data_root,
        "--output-dir",
        str(sample_dir),
        "--time-ids-per-stock",
        str(args.time_ids_per_stock),
        "--sort-order",
        args.sort_order,
    ]
    if args.stock_ids:
        build_sandbox_cmd.extend(["--stock-ids", args.stock_ids])
    else:
        build_sandbox_cmd.extend(["--stock-count", str(args.stock_count)])
    run_step(build_sandbox_cmd)

    features_v2_dir = sample_dir / "features_v2"
    run_step(
        [
            args.python_exe,
            str(PROJECT_SRC / "build_optiver_features_v2.py"),
            "--sample-dir",
            str(sample_dir),
            "--output-dir",
            str(features_v2_dir),
        ]
    )

    feature_table_path = features_v2_dir / "optiver_features_v2.parquet"

    if args.use_knn:
        features_knn_dir = sample_dir / "features_knn"
        run_step(
            [
                args.python_exe,
                str(PROJECT_SRC / "build_optiver_knn_features.py"),
                "--feature-table",
                str(feature_table_path),
                "--output-dir",
                str(features_knn_dir),
                "--global-k",
                str(args.global_k),
                "--same-stock-k",
                str(args.same_stock_k),
                "--min-history",
                str(args.min_history),
            ]
        )
        feature_table_path = features_knn_dir / "optiver_features_knn.parquet"

    train_cmd = [
        args.python_exe,
        str(PROJECT_SRC / "train_optiver_baseline.py"),
        "--feature-table",
        str(feature_table_path),
        "--model",
        args.model,
        "--valid-ratio",
        str(args.valid_ratio),
    ]
    if args.model in {"lightgbm", "mlp_lightgbm_ensemble"}:
        train_cmd.extend(
            [
                "--lgbm-num-leaves",
                str(args.lgbm_num_leaves),
                "--lgbm-max-depth",
                str(args.lgbm_max_depth),
                "--lgbm-min-child-samples",
                str(args.lgbm_min_child_samples),
                "--lgbm-learning-rate",
                str(args.lgbm_learning_rate),
                "--lgbm-n-estimators",
                str(args.lgbm_n_estimators),
            ]
        )
    if args.model in {"mlp", "mlp_lightgbm_ensemble"}:
        train_cmd.extend(
            [
                "--mlp-hidden-layers",
                args.mlp_hidden_layers,
                "--mlp-alpha",
                str(args.mlp_alpha),
                "--mlp-learning-rate-init",
                str(args.mlp_learning_rate_init),
                "--mlp-max-iter",
                str(args.mlp_max_iter),
            ]
        )
    if args.model == "mlp_lightgbm_ensemble":
        train_cmd.extend(
            [
                "--ensemble-lgbm-weight",
                str(args.ensemble_lgbm_weight),
            ]
        )
    run_step(train_cmd)

    print(f"Final feature table: {feature_table_path}")
    print(f"Sample directory: {sample_dir}")


if __name__ == "__main__":
    main()
