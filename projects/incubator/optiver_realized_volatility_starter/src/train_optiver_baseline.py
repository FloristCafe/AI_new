import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


DEFAULT_FEATURE_TABLE = Path(
    r"D:\Python\Datasets\optiver_realized_volatility_prediction\samples\optiver_sandbox_stocks_0-1-2_times_20\features_baseline\optiver_baseline_features.parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a first Optiver baseline regressor on the sandbox feature table."
    )
    parser.add_argument(
        "--feature-table",
        type=str,
        default=str(DEFAULT_FEATURE_TABLE),
        help="Path to the baseline feature parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory for metrics output. Defaults to feature table sibling baseline_models.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ridge",
        choices=["ridge", "random_forest"],
        help="Which baseline model to train.",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.2,
        help="Validation ratio using chronological order inside the feature table.",
    )
    return parser.parse_args()


def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.where(y_true == 0, 1e-8, y_true)
    return float(np.sqrt(np.mean(np.square((y_true - y_pred) / denominator))))


def build_model(model_name: str):
    if model_name == "ridge":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", Ridge(alpha=1.0)),
            ]
        )

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200,
                    max_depth=6,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def main() -> None:
    args = parse_args()
    feature_table_path = Path(args.feature_table)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else feature_table_path.parent / "baseline_models"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(feature_table_path)
    df = df.sort_values(["stock_id", "time_id"]).reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in {"stock_id", "time_id", "target"}]
    X = df[feature_cols]
    y = df["target"]

    split_index = max(1, int(len(df) * (1 - args.valid_ratio)))
    train_df = df.iloc[:split_index].copy()
    valid_df = df.iloc[split_index:].copy()

    X_train = train_df[feature_cols]
    y_train = train_df["target"]
    X_valid = valid_df[feature_cols]
    y_valid = valid_df["target"]

    model = build_model(args.model)
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    valid_pred = model.predict(X_valid)

    metrics = {
        "train": {
            "mae": float(mean_absolute_error(y_train, train_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_train, train_pred))),
            "r2": float(r2_score(y_train, train_pred)),
            "rmspe": rmspe(y_train.to_numpy(), train_pred),
        },
        "valid": {
            "mae": float(mean_absolute_error(y_valid, valid_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_valid, valid_pred))),
            "r2": float(r2_score(y_valid, valid_pred)),
            "rmspe": rmspe(y_valid.to_numpy(), valid_pred),
        },
        "metadata": {
            "model": args.model,
            "feature_table_path": str(feature_table_path),
            "feature_count": len(feature_cols),
            "train_rows": int(len(train_df)),
            "valid_rows": int(len(valid_df)),
            "valid_ratio": args.valid_ratio,
            "feature_columns": feature_cols,
        },
    }

    metrics_path = output_dir / f"{args.model}_metrics.json"
    pred_path = output_dir / f"{args.model}_valid_predictions.csv"

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    pd.DataFrame(
        {
            "stock_id": valid_df["stock_id"],
            "time_id": valid_df["time_id"],
            "target": y_valid,
            "prediction": valid_pred,
        }
    ).to_csv(pred_path, index=False)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Metrics saved to: {metrics_path}")
    print(f"Validation predictions saved to: {pred_path}")


if __name__ == "__main__":
    main()
