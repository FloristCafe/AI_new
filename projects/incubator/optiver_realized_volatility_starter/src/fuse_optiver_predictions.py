import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DEFAULT_TABULAR_DIR = Path(
    r"D:\Python\Datasets\optiver_realized_volatility_prediction\samples\optiver_sandbox_stocks_0-1-2-3-4-5-6-7-8-9-10-11-13-14-15-16-17-18-19-20-21-22-23-24_times_200\features_knn\baseline_models"
)
DEFAULT_CNN_DIR = Path(
    r"D:\Python\Datasets\optiver_realized_volatility_prediction\samples\optiver_sandbox_stocks_0-1-2-3-4-5-6-7-8-9-10-11-13-14-15-16-17-18-19-20-21-22-23-24_times_200\cnn_tensors\cnn_models"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an ensembled Optiver prediction result from tabular and CNN train/valid predictions."
    )
    parser.add_argument(
        "--tabular-train-pred",
        type=str,
        default=str(DEFAULT_TABULAR_DIR / "mlp_lightgbm_ensemble_train_predictions.csv"),
        help="CSV path containing tabular train predictions.",
    )
    parser.add_argument(
        "--tabular-valid-pred",
        type=str,
        default=str(DEFAULT_TABULAR_DIR / "mlp_lightgbm_ensemble_valid_predictions.csv"),
        help="CSV path containing tabular valid predictions.",
    )
    parser.add_argument(
        "--cnn-train-pred",
        type=str,
        default=str(DEFAULT_CNN_DIR / "dual_cnn_train_predictions.csv"),
        help="CSV path containing CNN train predictions.",
    )
    parser.add_argument(
        "--cnn-valid-pred",
        type=str,
        default=str(DEFAULT_CNN_DIR / "dual_cnn_valid_predictions.csv"),
        help="CSV path containing CNN valid predictions.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory for ensemble outputs. Defaults to tabular prediction sibling ensemble_models.",
    )
    parser.add_argument(
        "--tabular-weight",
        type=float,
        default=0.85,
        help="Weight for tabular prediction in late fusion.",
    )
    parser.add_argument(
        "--cnn-weight",
        type=float,
        default=0.15,
        help="Weight for CNN prediction in late fusion.",
    )
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="If set, search the best tabular weight on validation predictions.",
    )
    parser.add_argument(
        "--grid-step",
        type=float,
        default=0.05,
        help="Step size for tabular weight search when --grid-search is enabled.",
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


def load_and_align_predictions(
    tabular_path: Path,
    cnn_path: Path,
    split_name: str,
) -> pd.DataFrame:
    tabular_df = pd.read_csv(tabular_path)
    cnn_df = pd.read_csv(cnn_path)

    tabular_df = tabular_df.rename(
        columns={"prediction": "prediction_tabular", "target": "target_tabular"}
    )
    cnn_df = cnn_df.rename(
        columns={"prediction": "prediction_cnn", "target": "target_cnn"}
    )

    merged = tabular_df.merge(
        cnn_df[["stock_id", "time_id", "target_cnn", "prediction_cnn"]],
        on=["stock_id", "time_id"],
        how="inner",
    )
    if merged.empty:
        raise ValueError(f"No overlapping {split_name} rows between tabular and CNN predictions.")

    if not np.allclose(
        merged["target_tabular"].to_numpy(),
        merged["target_cnn"].to_numpy(),
        atol=1e-10,
    ):
        raise ValueError(f"Target mismatch detected in {split_name} predictions.")

    merged = merged.sort_values(["stock_id", "time_id"]).reset_index(drop=True)
    merged = merged.rename(columns={"target_tabular": "target"})
    merged = merged.drop(columns=["target_cnn"])
    return merged


def fuse_predictions(
    df: pd.DataFrame,
    tabular_weight: float,
    cnn_weight: float,
) -> tuple[np.ndarray, dict[str, float], float, float]:
    weight_sum = tabular_weight + cnn_weight
    if weight_sum <= 0:
        raise ValueError("tabular_weight and cnn_weight cannot both be zero or negative.")

    tabular_weight = tabular_weight / weight_sum
    cnn_weight = cnn_weight / weight_sum
    pred = (
        tabular_weight * df["prediction_tabular"].to_numpy()
        + cnn_weight * df["prediction_cnn"].to_numpy()
    )
    metrics = evaluate_predictions(df["target"].to_numpy(), pred)
    return pred, metrics, tabular_weight, cnn_weight


def search_best_weight(df: pd.DataFrame, step: float) -> dict[str, float]:
    if step <= 0 or step > 1:
        raise ValueError("grid_step must be in (0, 1].")

    candidates = np.arange(0.0, 1.0 + 1e-9, step)
    best = None
    for tabular_weight in candidates:
        cnn_weight = 1.0 - tabular_weight
        _, metrics, norm_tab_w, norm_cnn_w = fuse_predictions(df, tabular_weight, cnn_weight)
        record = {
            "tabular_weight": float(norm_tab_w),
            "cnn_weight": float(norm_cnn_w),
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "rmspe": metrics["rmspe"],
        }
        if best is None or record["r2"] > best["r2"]:
            best = record
    return best


def main() -> None:
    args = parse_args()
    tabular_train_path = Path(args.tabular_train_pred)
    tabular_valid_path = Path(args.tabular_valid_pred)
    cnn_train_path = Path(args.cnn_train_pred)
    cnn_valid_path = Path(args.cnn_valid_pred)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else tabular_valid_path.parent / "ensemble_models"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_and_align_predictions(tabular_train_path, cnn_train_path, "train")
    valid_df = load_and_align_predictions(tabular_valid_path, cnn_valid_path, "valid")

    best_weight_record = None
    if args.grid_search:
        best_weight_record = search_best_weight(valid_df, args.grid_step)
        tabular_weight = best_weight_record["tabular_weight"]
        cnn_weight = best_weight_record["cnn_weight"]
    else:
        tabular_weight = args.tabular_weight
        cnn_weight = args.cnn_weight

    train_pred, train_metrics, norm_tab_w, norm_cnn_w = fuse_predictions(
        train_df, tabular_weight, cnn_weight
    )
    valid_pred, valid_metrics, _, _ = fuse_predictions(
        valid_df, tabular_weight, cnn_weight
    )

    result = {
        "train": train_metrics,
        "valid": valid_metrics,
        "metadata": {
            "tabular_train_pred_path": str(tabular_train_path),
            "tabular_valid_pred_path": str(tabular_valid_path),
            "cnn_train_pred_path": str(cnn_train_path),
            "cnn_valid_pred_path": str(cnn_valid_path),
            "train_row_count": int(len(train_df)),
            "valid_row_count": int(len(valid_df)),
            "tabular_weight": float(norm_tab_w),
            "cnn_weight": float(norm_cnn_w),
            "grid_search": args.grid_search,
            "grid_step": args.grid_step,
        },
    }
    if best_weight_record is not None:
        result["best_weight_record"] = best_weight_record

    metrics_path = output_dir / "ensembled_metrics.json"
    train_pred_path = output_dir / "ensembled_train_predictions.csv"
    valid_pred_path = output_dir / "ensembled_valid_predictions.csv"

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    train_out = train_df[["stock_id", "time_id", "target"]].copy()
    train_out["prediction_tabular"] = train_df["prediction_tabular"]
    train_out["prediction_cnn"] = train_df["prediction_cnn"]
    train_out["prediction_ensembled"] = train_pred
    train_out.to_csv(train_pred_path, index=False)

    valid_out = valid_df[["stock_id", "time_id", "target"]].copy()
    valid_out["prediction_tabular"] = valid_df["prediction_tabular"]
    valid_out["prediction_cnn"] = valid_df["prediction_cnn"]
    valid_out["prediction_ensembled"] = valid_pred
    valid_out.to_csv(valid_pred_path, index=False)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Metrics saved to: {metrics_path}")
    print(f"Train predictions saved to: {train_pred_path}")
    print(f"Validation predictions saved to: {valid_pred_path}")


if __name__ == "__main__":
    main()
