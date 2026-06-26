import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
        choices=["ridge", "random_forest", "lightgbm", "mlp", "mlp_lightgbm_ensemble"],
        help="Which baseline model to train.",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.2,
        help="Validation ratio using chronological order inside the feature table.",
    )
    parser.add_argument(
        "--lgbm-num-leaves",
        type=int,
        default=31,
        help="LightGBM num_leaves.",
    )
    parser.add_argument(
        "--lgbm-max-depth",
        type=int,
        default=-1,
        help="LightGBM max_depth. Use -1 for no limit.",
    )
    parser.add_argument(
        "--lgbm-min-child-samples",
        type=int,
        default=20,
        help="LightGBM min_child_samples.",
    )
    parser.add_argument(
        "--lgbm-learning-rate",
        type=float,
        default=0.05,
        help="LightGBM learning_rate.",
    )
    parser.add_argument(
        "--lgbm-n-estimators",
        type=int,
        default=300,
        help="LightGBM number of boosting rounds.",
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
        help="Maximum training iterations for MLPRegressor.",
    )
    parser.add_argument(
        "--ensemble-lgbm-weight",
        type=float,
        default=0.6,
        help="Prediction weight for LightGBM in the MLP + LightGBM ensemble.",
    )
    parser.add_argument(
        "--bagging-size",
        type=int,
        default=1,
        help="How many bootstrap models to train and average. Use 3 for 3-bagging.",
    )
    return parser.parse_args()


def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.where(y_true == 0, 1e-8, y_true)
    return float(np.sqrt(np.mean(np.square((y_true - y_pred) / denominator))))


def parse_hidden_layers(value: str) -> tuple[int, ...]:
    layers = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    if not layers:
        raise ValueError("mlp_hidden_layers must contain at least one layer size.")
    return layers


def build_ridge_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", Ridge(alpha=1.0)),
        ]
    )


def build_lightgbm_model(args: argparse.Namespace) -> Pipeline:
    try:
        from lightgbm import LGBMRegressor
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "lightgbm is not installed in the current environment. "
            "Install it first, for example with: "
            "pip install lightgbm"
        ) from exc

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                LGBMRegressor(
                    n_estimators=args.lgbm_n_estimators,
                    learning_rate=args.lgbm_learning_rate,
                    max_depth=args.lgbm_max_depth,
                    num_leaves=args.lgbm_num_leaves,
                    min_child_samples=args.lgbm_min_child_samples,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=42,
                    verbosity=-1,
                    force_col_wise=True,
                ),
            ),
        ]
    )


def build_mlp_model(args: argparse.Namespace) -> Pipeline:
    feature_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    mlp = MLPRegressor(
        hidden_layer_sizes=parse_hidden_layers(args.mlp_hidden_layers),
        activation="relu",
        solver="adam",
        alpha=args.mlp_alpha,
        learning_rate_init=args.mlp_learning_rate_init,
        max_iter=args.mlp_max_iter,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
    )
    return TransformedTargetRegressor(
        regressor=Pipeline(
            [
                ("features", feature_pipeline),
                ("model", mlp),
            ]
        ),
        transformer=StandardScaler(),
    )


def build_random_forest_model() -> Pipeline:
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


def build_model(model_name: str, args: argparse.Namespace):
    if model_name == "ridge":
        return build_ridge_model()
    if model_name == "lightgbm":
        return build_lightgbm_model(args)
    if model_name == "mlp":
        return build_mlp_model(args)
    return build_random_forest_model()


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "rmspe": rmspe(y_true.to_numpy(), y_pred),
    }


def make_bootstrap_sample(
    X_train: pd.DataFrame, y_train: pd.Series, seed: int
) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(0, len(X_train), size=len(X_train))
    X_boot = X_train.iloc[sample_idx].reset_index(drop=True)
    y_boot = y_train.iloc[sample_idx].reset_index(drop=True)
    return X_boot, y_boot


def train_bagged_model(
    model_name: str,
    args: argparse.Namespace,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    if args.bagging_size <= 1:
        model = build_model(model_name, args)
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        valid_pred = model.predict(X_valid)
        if model_name == "mlp":
            train_pred = np.clip(train_pred, 0.0, None)
            valid_pred = np.clip(valid_pred, 0.0, None)
        return train_pred, valid_pred

    train_preds: list[np.ndarray] = []
    valid_preds: list[np.ndarray] = []

    for bag_idx in range(args.bagging_size):
        bag_args = copy.deepcopy(args)
        seed = 42 + bag_idx
        X_boot, y_boot = make_bootstrap_sample(X_train, y_train, seed)

        if model_name == "lightgbm":
            bag_args.lgbm_n_estimators = args.lgbm_n_estimators
        model = build_model(model_name, bag_args)
        model.fit(X_boot, y_boot)

        bag_train_pred = model.predict(X_train)
        bag_valid_pred = model.predict(X_valid)
        if model_name == "mlp":
            bag_train_pred = np.clip(bag_train_pred, 0.0, None)
            bag_valid_pred = np.clip(bag_valid_pred, 0.0, None)

        train_preds.append(bag_train_pred)
        valid_preds.append(bag_valid_pred)

    return np.mean(train_preds, axis=0), np.mean(valid_preds, axis=0)


def train_bagged_component(
    model_name: str,
    args: argparse.Namespace,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    train_pred, valid_pred = train_bagged_model(
        model_name=model_name,
        args=args,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
    )
    return train_pred, valid_pred


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

    component_metrics: dict[str, dict[str, dict[str, float]]] = {}
    component_weights: dict[str, float] | None = None

    if args.model == "mlp_lightgbm_ensemble":
        if not 0.0 <= args.ensemble_lgbm_weight <= 1.0:
            raise ValueError("ensemble_lgbm_weight must be between 0 and 1.")
        lgbm_train_pred, lgbm_valid_pred = train_bagged_component(
            model_name="lightgbm",
            args=args,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
        )
        mlp_train_pred, mlp_valid_pred = train_bagged_component(
            model_name="mlp",
            args=args,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
        )

        lgbm_weight = args.ensemble_lgbm_weight
        mlp_weight = 1.0 - lgbm_weight

        train_pred = lgbm_weight * lgbm_train_pred + mlp_weight * mlp_train_pred
        valid_pred = lgbm_weight * lgbm_valid_pred + mlp_weight * mlp_valid_pred

        component_weights = {
            "lightgbm": lgbm_weight,
            "mlp": mlp_weight,
        }
        component_metrics = {
            "lightgbm": {
                "train": evaluate_predictions(y_train, lgbm_train_pred),
                "valid": evaluate_predictions(y_valid, lgbm_valid_pred),
            },
            "mlp": {
                "train": evaluate_predictions(y_train, mlp_train_pred),
                "valid": evaluate_predictions(y_valid, mlp_valid_pred),
            },
        }
    else:
        train_pred, valid_pred = train_bagged_model(
            model_name=args.model,
            args=args,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
        )

    metrics = {
        "train": evaluate_predictions(y_train, train_pred),
        "valid": evaluate_predictions(y_valid, valid_pred),
        "metadata": {
            "model": args.model,
            "feature_table_path": str(feature_table_path),
            "feature_count": len(feature_cols),
            "train_rows": int(len(train_df)),
            "valid_rows": int(len(valid_df)),
            "valid_ratio": args.valid_ratio,
            "feature_columns": feature_cols,
            "lgbm_num_leaves": args.lgbm_num_leaves,
            "lgbm_max_depth": args.lgbm_max_depth,
            "lgbm_min_child_samples": args.lgbm_min_child_samples,
            "lgbm_learning_rate": args.lgbm_learning_rate,
            "lgbm_n_estimators": args.lgbm_n_estimators,
            "mlp_hidden_layers": list(parse_hidden_layers(args.mlp_hidden_layers)),
            "mlp_alpha": args.mlp_alpha,
            "mlp_learning_rate_init": args.mlp_learning_rate_init,
            "mlp_max_iter": args.mlp_max_iter,
            "ensemble_lgbm_weight": args.ensemble_lgbm_weight,
            "bagging_size": args.bagging_size,
        },
    }
    if component_weights is not None:
        metrics["metadata"]["ensemble_component_weights"] = component_weights
        metrics["component_metrics"] = component_metrics

    metrics_path = output_dir / f"{args.model}_metrics.json"
    train_pred_path = output_dir / f"{args.model}_train_predictions.csv"
    pred_path = output_dir / f"{args.model}_valid_predictions.csv"

    train_pred_df = pd.DataFrame(
        {
            "stock_id": train_df["stock_id"],
            "time_id": train_df["time_id"],
            "target": y_train,
            "prediction": train_pred,
        }
    )
    if args.model == "mlp_lightgbm_ensemble":
        train_pred_df["prediction_lightgbm"] = lgbm_train_pred
        train_pred_df["prediction_mlp"] = mlp_train_pred

    valid_pred_df = pd.DataFrame(
        {
            "stock_id": valid_df["stock_id"],
            "time_id": valid_df["time_id"],
            "target": y_valid,
            "prediction": valid_pred,
        }
    )
    if args.model == "mlp_lightgbm_ensemble":
        valid_pred_df["prediction_lightgbm"] = lgbm_valid_pred
        valid_pred_df["prediction_mlp"] = mlp_valid_pred

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    train_pred_df.to_csv(train_pred_path, index=False)
    valid_pred_df.to_csv(pred_path, index=False)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Metrics saved to: {metrics_path}")
    print(f"Train predictions saved to: {train_pred_path}")
    print(f"Validation predictions saved to: {pred_path}")


if __name__ == "__main__":
    main()
