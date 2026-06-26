import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_STACKING_DIR = Path(
    r"D:\Python\Datasets\optiver_realized_volatility_prediction\samples\optiver_sandbox_stocks_0-1-2-3-4-5-6-7-8-9-10-11-13-14-15-16-17-18-19-20-21-22-23-24_times_200\features_knn\stacking_models"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train only the second-stage meta learner from saved Optiver stacking "
            "OOF and validation prediction CSV files."
        )
    )
    parser.add_argument(
        "--oof-pred",
        type=str,
        default=str(DEFAULT_STACKING_DIR / "stacking_oof_predictions.csv"),
        help="CSV path containing OOF predictions from base learners.",
    )
    parser.add_argument(
        "--valid-pred",
        type=str,
        default=str(DEFAULT_STACKING_DIR / "stacking_valid_predictions.csv"),
        help="CSV path containing validation predictions from base learners.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory for meta-only outputs. Defaults to OOF prediction sibling meta_only_models.",
    )
    parser.add_argument(
        "--meta-model",
        type=str,
        default="ridge",
        choices=["ridge", "lightgbm"],
        help="Meta learner trained on cached base predictions.",
    )
    parser.add_argument(
        "--meta-ridge-alpha",
        type=float,
        default=1.0,
        help="Ridge alpha when --meta-model ridge.",
    )
    parser.add_argument(
        "--use-mlp",
        action="store_true",
        help="If set, include MLP base predictions in the meta features.",
    )
    parser.add_argument(
        "--use-cnn",
        action="store_true",
        help="If set, include CNN base predictions in the meta features.",
    )
    parser.add_argument(
        "--use-augmented-features",
        action="store_true",
        help="If set, add mean/std/gap style meta interaction features.",
    )
    return parser.parse_args()


def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.where(y_true == 0, 1e-8, y_true)
    return float(np.sqrt(np.mean(np.square((y_true - y_pred) / denominator))))


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "rmspe": rmspe(y_true, y_pred),
    }


def build_meta_model(args: argparse.Namespace):
    if args.meta_model == "ridge":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=args.meta_ridge_alpha)),
            ]
        )

    try:
        from lightgbm import LGBMRegressor
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "lightgbm is not installed in the current environment. Install it first with "
            "pip install lightgbm"
        ) from exc

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                LGBMRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=3,
                    num_leaves=15,
                    min_child_samples=20,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=42,
                    verbosity=-1,
                    force_col_wise=True,
                ),
            ),
        ]
    )


def build_meta_feature_frame(
    df: pd.DataFrame,
    use_mlp: bool,
    use_cnn: bool,
    use_augmented: bool,
) -> pd.DataFrame:
    feature_map = {
        "lightgbm": {
            "oof": "prediction_lightgbm_oof",
            "valid": "prediction_lightgbm",
        },
        "mlp": {
            "oof": "prediction_mlp_oof",
            "valid": "prediction_mlp",
        },
        "cnn": {
            "oof": "prediction_cnn_oof",
            "valid": "prediction_cnn",
        },
    }

    selected = ["lightgbm"]
    if use_mlp:
        selected.append("mlp")
    if use_cnn:
        selected.append("cnn")

    rename_map = {}
    for key in selected:
        for column_name in feature_map[key].values():
            if column_name in df.columns:
                rename_map[column_name] = f"pred_{key}"

    missing = [c for c in rename_map if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required prediction columns: {missing}")

    frame = df[list(rename_map.keys())].rename(columns=rename_map).copy()

    if use_augmented and frame.shape[1] >= 2:
        frame["pred_mean"] = frame.mean(axis=1)
        frame["pred_std"] = frame.std(axis=1)
        cols = list(frame.columns[: len(selected)])
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                left = cols[i]
                right = cols[j]
                gap_name = f"{left}_{right}_gap"
                frame[gap_name] = frame[left] - frame[right]
                frame[f"{gap_name}_abs"] = np.abs(frame[gap_name])
    return frame


def main() -> None:
    args = parse_args()
    oof_path = Path(args.oof_pred)
    valid_path = Path(args.valid_pred)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else oof_path.parent / "meta_only_models"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    oof_df = pd.read_csv(oof_path)
    valid_df = pd.read_csv(valid_path)

    meta_train_X = build_meta_feature_frame(
        oof_df,
        use_mlp=args.use_mlp,
        use_cnn=args.use_cnn,
        use_augmented=args.use_augmented_features,
    )
    meta_valid_X = build_meta_feature_frame(
        valid_df,
        use_mlp=args.use_mlp,
        use_cnn=args.use_cnn,
        use_augmented=args.use_augmented_features,
    )

    y_train = oof_df["target"].to_numpy(dtype=np.float32)
    y_valid = valid_df["target"].to_numpy(dtype=np.float32)

    meta_model = build_meta_model(args)
    meta_model.fit(meta_train_X, y_train)

    train_pred = np.clip(meta_model.predict(meta_train_X), 0.0, None)
    valid_pred = np.clip(meta_model.predict(meta_valid_X), 0.0, None)

    metrics = {
        "train": evaluate_predictions(y_train, train_pred),
        "valid": evaluate_predictions(y_valid, valid_pred),
        "metadata": {
            "oof_pred_path": str(oof_path),
            "valid_pred_path": str(valid_path),
            "meta_model": args.meta_model,
            "meta_ridge_alpha": args.meta_ridge_alpha,
            "use_mlp": args.use_mlp,
            "use_cnn": args.use_cnn,
            "use_augmented_features": args.use_augmented_features,
            "meta_feature_columns": list(meta_train_X.columns),
            "train_rows": int(len(oof_df)),
            "valid_rows": int(len(valid_df)),
        },
    }

    metrics_path = output_dir / "meta_only_metrics.json"
    train_pred_path = output_dir / "meta_only_train_predictions.csv"
    valid_pred_path = output_dir / "meta_only_valid_predictions.csv"

    train_out = oof_df[["stock_id", "time_id", "target"]].copy()
    for col in meta_train_X.columns:
        train_out[col] = meta_train_X[col]
    train_out["prediction_meta"] = train_pred

    valid_out = valid_df[["stock_id", "time_id", "target"]].copy()
    for col in meta_valid_X.columns:
        valid_out[col] = meta_valid_X[col]
    valid_out["prediction_meta"] = valid_pred

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    train_out.to_csv(train_pred_path, index=False)
    valid_out.to_csv(valid_pred_path, index=False)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Metrics saved to: {metrics_path}")
    print(f"Train predictions saved to: {train_pred_path}")
    print(f"Validation predictions saved to: {valid_pred_path}")


if __name__ == "__main__":
    main()
