import argparse
import json
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def parse_args() -> argparse.Namespace:
    # Keep the script configurable from the command line for easy experiments.
    parser = argparse.ArgumentParser(
        description="Train a first logistic-regression CTR baseline on preprocessed Criteo data."
    )
    parser.add_argument(
        "--train-path",
        type=str,
        default=r"D:\Python\Artificial Intelligence\projects\recommendation\criteo_ctr_baseline\artifacts\train_processed.parquet",
        help="Path to the preprocessed training parquet.",
    )
    parser.add_argument(
        "--valid-path",
        type=str,
        default=r"D:\Python\Artificial Intelligence\projects\recommendation\criteo_ctr_baseline\artifacts\valid_processed.parquet",
        help="Path to the preprocessed validation parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=r"D:\Python\Artificial Intelligence\projects\recommendation\criteo_ctr_baseline\artifacts\baseline_lr",
        help="Directory for model and metrics outputs.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=300,
        help="Maximum iterations for logistic regression.",
    )
    parser.add_argument(
        "--class-weight",
        type=str,
        default="balanced",
        choices=["balanced", "none"],
        help="Use balanced class weights for imbalanced CTR labels.",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=1.0,
        help="Inverse regularization strength for logistic regression.",
    )
    return parser.parse_args()


def collect_feature_groups(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    # Reuse the same grouping rule as preprocessing so train-time behavior is aligned.
    dense_cols = [c for c in df.columns if c.startswith("integer_feature_")]
    sparse_cols = [c for c in df.columns if c.startswith("categorical_feature_")]
    return dense_cols, sparse_cols


def load_data(train_path: Path, valid_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Fail early if an expected artifact is missing.
    if not train_path.exists():
        raise FileNotFoundError(f"Training parquet not found: {train_path}")
    if not valid_path.exists():
        raise FileNotFoundError(f"Validation parquet not found: {valid_path}")

    train_df = pd.read_parquet(train_path)
    valid_df = pd.read_parquet(valid_path)
    return train_df, valid_df


def build_pipeline(
    dense_cols: list[str],
    sparse_cols: list[str],
    max_iter: int,
    class_weight: str,
    c: float,
) -> Pipeline:
    # Dense features: use the same missing sentinel as preprocessing, then
    # standardize scales for logistic regression.
    dense_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
            ("scaler", StandardScaler()),
        ]
    )

    # Sparse features: treat ids as categories, not as ordered numeric values.
    sparse_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Apply different transforms to different column groups, then concatenate.
    preprocessor = ColumnTransformer(
        transformers=[
            ("dense", dense_transformer, dense_cols),
            ("sparse", sparse_transformer, sparse_cols),
        ]
    )

    # Smaller C means stronger regularization, which is useful when we overfit.
    model = LogisticRegression(
        C=c,
        max_iter=max_iter,
        class_weight=None if class_weight == "none" else class_weight,
        solver="liblinear",
    )

    # Pipeline keeps preprocessing and model training bound together.
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def evaluate_predictions(y_true: pd.Series, y_prob: pd.Series) -> dict[str, float]:
    # CTR tasks care about probability quality and ranking quality.
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
    }


def main() -> None:
    args = parse_args()

    train_path = Path(args.train_path)
    valid_path = Path(args.valid_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, valid_df = load_data(train_path, valid_path)
    dense_cols, sparse_cols = collect_feature_groups(train_df)
    feature_cols = dense_cols + sparse_cols

    X_train = train_df[feature_cols]
    y_train = train_df["label"]
    X_valid = valid_df[feature_cols]
    y_valid = valid_df["label"]

    pipeline = build_pipeline(
        dense_cols=dense_cols,
        sparse_cols=sparse_cols,
        max_iter=args.max_iter,
        class_weight=args.class_weight,
        c=args.c,
    )
    pipeline.fit(X_train, y_train)

    train_prob = pipeline.predict_proba(X_train)[:, 1]
    valid_prob = pipeline.predict_proba(X_valid)[:, 1]

    metrics = {
        "train": evaluate_predictions(y_train, train_prob),
        "valid": evaluate_predictions(y_valid, valid_prob),
        "metadata": {
            "train_rows": int(len(train_df)),
            "valid_rows": int(len(valid_df)),
            "feature_count": len(feature_cols),
            "dense_feature_count": len(dense_cols),
            "sparse_feature_count": len(sparse_cols),
            "class_weight": args.class_weight,
            "max_iter": args.max_iter,
            "c": args.c,
        },
    }

    model_path = output_dir / "logistic_regression_pipeline.joblib"
    metrics_path = output_dir / "metrics.json"
    valid_pred_path = output_dir / "valid_predictions.parquet"

    dump(pipeline, model_path)

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    valid_predictions = valid_df[["label"]].copy()
    valid_predictions["prediction"] = valid_prob
    valid_predictions.to_parquet(valid_pred_path, index=False)

    print("Training finished.")
    print(f"Train data: {train_path}")
    print(f"Valid data: {valid_path}")
    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Validation predictions saved to: {valid_pred_path}")
    print(
        "Valid metrics: "
        f"ROC-AUC={metrics['valid']['roc_auc']:.6f}, "
        f"PR-AUC={metrics['valid']['pr_auc']:.6f}, "
        f"LogLoss={metrics['valid']['log_loss']:.6f}"
    )


if __name__ == "__main__":
    main()
