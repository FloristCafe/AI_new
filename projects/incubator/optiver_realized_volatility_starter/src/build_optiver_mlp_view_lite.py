import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_FEATURE_TABLE = Path(
    r"D:\Python\Datasets\optiver_realized_volatility_prediction\samples\optiver_sandbox_stocks_0-1-2-3-4-5-6-7-8-9-10-11-13-14-15-16-17-18-19-20-21-22-23-24_times_200\features_knn\optiver_features_knn.parquet"
)

EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a lighter MLP-oriented Optiver feature view: fewer raw features, "
            "more stable derived features, and no high-variance history chains."
        )
    )
    parser.add_argument(
        "--feature-table",
        type=str,
        default=str(DEFAULT_FEATURE_TABLE),
        help="Path to the source Optiver feature parquet, usually features_knn.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory for the lite MLP view table. Defaults to source sibling features_mlp_view_lite.",
    )
    parser.add_argument(
        "--clip-lower-quantile",
        type=float,
        default=0.01,
        help="Lower quantile for winsorized heavy-tail features.",
    )
    parser.add_argument(
        "--clip-upper-quantile",
        type=float,
        default=0.99,
        help="Upper quantile for winsorized heavy-tail features.",
    )
    return parser.parse_args()


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def intersect_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def add_log1p_features(df: pd.DataFrame, columns: list[str], added: list[str]) -> None:
    for col in intersect_columns(df, columns):
        out_col = f"mlp_lite_log1p__{col}"
        df[out_col] = np.log1p(np.clip(df[col], a_min=0.0, a_max=None))
        added.append(out_col)


def add_winsorized_features(
    df: pd.DataFrame,
    columns: list[str],
    lower_q: float,
    upper_q: float,
    added: list[str],
) -> None:
    for col in intersect_columns(df, columns):
        low = float(df[col].quantile(lower_q))
        high = float(df[col].quantile(upper_q))
        out_col = f"mlp_lite_clip__{col}"
        df[out_col] = df[col].clip(lower=low, upper=high)
        added.append(out_col)


def add_group_zscore_features(
    df: pd.DataFrame,
    columns: list[str],
    group_col: str,
    prefix: str,
    added: list[str],
) -> None:
    for col in intersect_columns(df, columns):
        group_mean = df.groupby(group_col)[col].transform("mean")
        group_std = df.groupby(group_col)[col].transform("std")
        group_std = group_std.mask(group_std.abs() < 1e-4, np.nan)
        out_col = f"mlp_lite_{prefix}_z__{col}"
        df[out_col] = ((df[col] - group_mean) / group_std).clip(-10.0, 10.0)
        added.append(out_col)


def add_group_rank_features(
    df: pd.DataFrame,
    columns: list[str],
    group_col: str,
    prefix: str,
    added: list[str],
) -> None:
    for col in intersect_columns(df, columns):
        out_col = f"mlp_lite_{prefix}_rank__{col}"
        df[out_col] = df.groupby(group_col)[col].rank(pct=True)
        added.append(out_col)


def add_asinh_features(df: pd.DataFrame, columns: list[str], added: list[str]) -> None:
    for col in intersect_columns(df, columns):
        out_col = f"mlp_lite_asinh__{col}"
        df[out_col] = np.arcsinh(df[col])
        added.append(out_col)


def add_window_delta_features(df: pd.DataFrame, added: list[str]) -> None:
    specs = [
        ("wap1_last_150_realized_vol", "wap1_last_300_realized_vol", "wap1_vol_short_long"),
        ("mid1_last_150_realized_vol", "mid1_last_300_realized_vol", "mid1_vol_short_long"),
        ("trade_price_last_150_realized_vol", "trade_price_last_300_realized_vol", "trade_vol_short_long"),
        ("wap1_last_150_mean", "wap1_last_300_mean", "wap1_mean_short_long"),
        ("mid1_last_150_mean", "mid1_last_300_mean", "mid1_mean_short_long"),
        ("trade_price_last_150_mean", "trade_price_last_300_mean", "trade_mean_short_long"),
    ]
    for short_col, long_col, name in specs:
        if short_col not in df.columns or long_col not in df.columns:
            continue
        out_col = f"mlp_lite_delta__{name}"
        df[out_col] = df[short_col] - df[long_col]
        added.append(out_col)


def add_book_trade_interactions(df: pd.DataFrame, added: list[str]) -> None:
    specs = [
        (
            "mlp_lite_interact__trade_activity_imbalance",
            lambda x: x["trade_active_ratio"] * x["size_imbalance1_mean"],
        ),
        (
            "mlp_lite_interact__trade_depth_vol",
            lambda x: x["trade_depth_ratio_mean"] * x["realized_vol_wap1"],
        ),
        (
            "mlp_lite_interact__trade_impact_imbalance",
            lambda x: x["trade_impact_mean"] * x["size_imbalance1_mean"],
        ),
        (
            "mlp_lite_delta__trade_price_vs_wap1_vol",
            lambda x: x["realized_vol_trade_price"] - x["realized_vol_wap1"],
        ),
        (
            "mlp_lite_delta__trade_price_vs_mid1_vol",
            lambda x: x["realized_vol_trade_price"] - x["realized_vol_mid1"],
        ),
        (
            "mlp_lite_delta__knn_same_vs_global_target_mean",
            lambda x: x["knn_same_stock_target_mean_k3"] - x["knn_global_target_mean_k5"],
        ),
    ]
    for out_col, fn in specs:
        try:
            df[out_col] = fn(df)
            added.append(out_col)
        except KeyError:
            continue


def main() -> None:
    args = parse_args()
    feature_table_path = Path(args.feature_table)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else feature_table_path.parent.parent / "features_mlp_view_lite"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(feature_table_path)
    df = df.sort_values(["time_id", "stock_id"]).reset_index(drop=True)

    added_columns: list[str] = []

    raw_core_cols = [
        "spread1_mean",
        "size_imbalance1_mean",
        "size_imbalance1_std",
        "realized_vol_wap1",
        "realized_vol_mid1",
        "trade_active_ratio",
        "trade_price_std",
        "trade_impact_mean",
        "trade_depth_ratio_mean",
        "realized_vol_trade_price",
        "knn_global_target_mean_k5",
        "knn_same_stock_target_mean_k3",
    ]

    positive_tail_cols = [
        "spread1_std",
        "realized_vol_wap1",
        "realized_vol_mid1",
        "trade_row_count",
        "trade_unique_seconds",
        "trade_price_std",
        "trade_size_sum",
        "trade_size_mean",
        "trade_size_std",
        "trade_size_max",
        "trade_order_count_sum",
        "trade_order_count_mean",
        "trade_order_count_std",
        "trade_notional_sum",
        "trade_notional_mean",
        "trade_impact_mean",
        "trade_depth_ratio_mean",
        "realized_vol_trade_price",
        "knn_global_dist_mean_k5",
        "knn_same_stock_dist_mean_k3",
    ]

    winsor_cols = [
        "spread1_mean",
        "spread1_std",
        "trade_price_std",
        "trade_size_sum",
        "trade_size_std",
        "trade_order_count_sum",
        "trade_notional_sum",
        "trade_impact_mean",
        "trade_depth_ratio_mean",
        "realized_vol_trade_price",
    ]

    stock_z_cols = [
        "spread1_mean",
        "size_imbalance1_mean",
        "realized_vol_wap1",
        "realized_vol_mid1",
        "trade_active_ratio",
        "trade_impact_mean",
        "trade_depth_ratio_mean",
        "realized_vol_trade_price",
        "knn_global_target_mean_k5",
        "knn_same_stock_target_mean_k3",
    ]

    cross_cols = [
        "spread1_mean",
        "size_imbalance1_mean",
        "realized_vol_wap1",
        "trade_active_ratio",
        "trade_impact_mean",
        "trade_depth_ratio_mean",
        "realized_vol_trade_price",
        "knn_global_target_mean_k5",
    ]

    asinh_cols = [
        "size_imbalance1_mean",
        "size_imbalance1_std",
        "size_imbalance2_mean",
        "size_imbalance2_std",
        "trade_mid_gap_mean",
        "trade_mid_gap_std",
        "spread_order_interaction_mean",
        "imbalance_size_interaction_mean",
    ]

    add_log1p_features(df, positive_tail_cols, added_columns)
    add_winsorized_features(
        df,
        winsor_cols,
        lower_q=args.clip_lower_quantile,
        upper_q=args.clip_upper_quantile,
        added=added_columns,
    )
    add_group_zscore_features(df, stock_z_cols, "stock_id", "stock", added_columns)
    add_group_zscore_features(df, cross_cols, "time_id", "cross", added_columns)
    add_group_rank_features(df, cross_cols, "time_id", "cross", added_columns)
    add_window_delta_features(df, added_columns)
    add_book_trade_interactions(df, added_columns)
    add_asinh_features(df, asinh_cols, added_columns)

    final_raw_cols = intersect_columns(df, raw_core_cols)
    final_cols = ["stock_id", "time_id", "target", *final_raw_cols, *added_columns]
    feature_df = df[final_cols].copy()

    feature_table_out = output_dir / "optiver_features_mlp_view_lite.parquet"
    summary_out = output_dir / "summary.json"
    feature_df.to_parquet(feature_table_out, index=False)

    summary = {
        "source_feature_table": str(feature_table_path),
        "output_dir": str(output_dir),
        "row_count": int(len(feature_df)),
        "column_count": int(feature_df.shape[1]),
        "raw_core_count": int(len(final_raw_cols)),
        "raw_core_columns": final_raw_cols,
        "added_feature_count": int(len(added_columns)),
        "added_feature_columns": added_columns,
        "clip_lower_quantile": args.clip_lower_quantile,
        "clip_upper_quantile": args.clip_upper_quantile,
        "view_version": "lite",
        "feature_table_path": str(feature_table_out),
    }

    with summary_out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
