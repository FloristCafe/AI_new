import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


DEFAULT_FEATURE_TABLE = Path(
    r"D:\Python\Datasets\optiver_realized_volatility_prediction\samples\optiver_sandbox_stocks_0-1-2-3-4-5-6-7_times_80\features_v2\optiver_features_v2.parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add past-only nearest-neighbor target features to an Optiver feature table."
    )
    parser.add_argument(
        "--feature-table",
        type=str,
        default=str(DEFAULT_FEATURE_TABLE),
        help="Path to the existing Optiver feature parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory for the KNN-augmented feature table. Defaults to feature table sibling features_knn.",
    )
    parser.add_argument(
        "--global-k",
        type=int,
        default=5,
        help="Neighbor count for global past-sample KNN features.",
    )
    parser.add_argument(
        "--same-stock-k",
        type=int,
        default=3,
        help="Neighbor count for same-stock past-sample KNN features.",
    )
    parser.add_argument(
        "--min-history",
        type=int,
        default=20,
        help="Minimum number of past rows before computing global KNN features.",
    )
    return parser.parse_args()


def compute_knn_stats(
    past_X: np.ndarray,
    past_y: np.ndarray,
    current_x: np.ndarray,
    k: int,
) -> tuple[float, float, float, float]:
    distances = np.linalg.norm(past_X - current_x, axis=1)
    k = min(k, len(distances))
    idx = np.argpartition(distances, k - 1)[:k]
    selected_distances = distances[idx]
    selected_targets = past_y[idx]
    return (
        float(selected_targets.mean()),
        float(selected_targets.std(ddof=0)),
        float(selected_distances.mean()),
        float(selected_distances.min()),
    )


def main() -> None:
    args = parse_args()
    feature_table_path = Path(args.feature_table)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else feature_table_path.parent.parent / "features_knn"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(feature_table_path)
    df = df.sort_values(["time_id", "stock_id"]).reset_index(drop=True)

    base_feature_cols = [c for c in df.columns if c not in {"stock_id", "time_id", "target"}]

    global_mean_col = f"knn_global_target_mean_k{args.global_k}"
    global_std_col = f"knn_global_target_std_k{args.global_k}"
    global_dist_mean_col = f"knn_global_dist_mean_k{args.global_k}"
    global_dist_min_col = f"knn_global_dist_min_k{args.global_k}"

    same_mean_col = f"knn_same_stock_target_mean_k{args.same_stock_k}"
    same_std_col = f"knn_same_stock_target_std_k{args.same_stock_k}"
    same_dist_mean_col = f"knn_same_stock_dist_mean_k{args.same_stock_k}"
    same_dist_min_col = f"knn_same_stock_dist_min_k{args.same_stock_k}"

    df[global_mean_col] = np.nan
    df[global_std_col] = np.nan
    df[global_dist_mean_col] = np.nan
    df[global_dist_min_col] = np.nan
    df[same_mean_col] = np.nan
    df[same_std_col] = np.nan
    df[same_dist_mean_col] = np.nan
    df[same_dist_min_col] = np.nan

    for idx, row in df.iterrows():
        current_time_id = row["time_id"]
        current_stock_id = row["stock_id"]

        past_mask = df["time_id"] < current_time_id
        past_df = df.loc[past_mask, ["stock_id", "time_id", "target", *base_feature_cols]].copy()

        if len(past_df) < args.min_history:
            continue

        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()

        past_X = imputer.fit_transform(past_df[base_feature_cols])
        past_X = scaler.fit_transform(past_X)
        current_X = imputer.transform(df.loc[[idx], base_feature_cols])
        current_X = scaler.transform(current_X)[0]
        past_y = past_df["target"].to_numpy()

        (
            df.at[idx, global_mean_col],
            df.at[idx, global_std_col],
            df.at[idx, global_dist_mean_col],
            df.at[idx, global_dist_min_col],
        ) = compute_knn_stats(past_X, past_y, current_X, args.global_k)

        same_stock_df = past_df.loc[past_df["stock_id"] == current_stock_id].copy()
        if len(same_stock_df) >= args.same_stock_k:
            same_imputer = SimpleImputer(strategy="median")
            same_scaler = StandardScaler()
            same_X = same_imputer.fit_transform(same_stock_df[base_feature_cols])
            same_X = same_scaler.fit_transform(same_X)
            current_same_X = same_imputer.transform(df.loc[[idx], base_feature_cols])
            current_same_X = same_scaler.transform(current_same_X)[0]
            same_y = same_stock_df["target"].to_numpy()

            (
                df.at[idx, same_mean_col],
                df.at[idx, same_std_col],
                df.at[idx, same_dist_mean_col],
                df.at[idx, same_dist_min_col],
            ) = compute_knn_stats(
                same_X, same_y, current_same_X, args.same_stock_k
            )

    feature_table_out = output_dir / "optiver_features_knn.parquet"
    summary_out = output_dir / "summary.json"
    df.to_parquet(feature_table_out, index=False)

    summary = {
        "source_feature_table": str(feature_table_path),
        "output_dir": str(output_dir),
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "global_k": args.global_k,
        "same_stock_k": args.same_stock_k,
        "min_history": args.min_history,
        "added_feature_columns": [
            global_mean_col,
            global_std_col,
            global_dist_mean_col,
            global_dist_min_col,
            same_mean_col,
            same_std_col,
            same_dist_mean_col,
            same_dist_min_col,
        ],
        "non_null_counts": {
            global_mean_col: int(df[global_mean_col].notna().sum()),
            same_mean_col: int(df[same_mean_col].notna().sum()),
        },
        "feature_table_path": str(feature_table_out),
    }

    with summary_out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
