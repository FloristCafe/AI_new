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
            "Build an MLP-oriented Optiver feature view with smoother, relative and "
            "context-aware features on top of the existing tabular feature table."
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
        help="Directory for the MLP view feature table. Defaults to source sibling features_mlp_view_v4.",
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


def robust_divide(
    numerator: pd.Series,
    denominator: pd.Series,
    min_abs_denom: float,
) -> pd.Series:
    denom = denominator.copy()
    denom = denom.where(denom.abs() >= min_abs_denom, np.nan)
    return safe_divide(numerator, denom)


def clip_feature(series: pd.Series, clip_value: float) -> pd.Series:
    return series.clip(lower=-clip_value, upper=clip_value)


def signed_asinh(series: pd.Series) -> pd.Series:
    return np.arcsinh(series)


def robust_log_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
    min_abs_value: float,
) -> pd.Series:
    num = numerator.where(numerator.abs() >= min_abs_value, np.nan)
    den = denominator.where(denominator.abs() >= min_abs_value, np.nan)
    return np.log(num) - np.log(den)


def robust_zscore(
    value: pd.Series,
    mean: pd.Series,
    std: pd.Series,
    min_std: float,
    clip_value: float,
) -> pd.Series:
    stable_std = std.where(std.abs() >= min_std, np.nan)
    z = (value - mean) / stable_std
    return clip_feature(z, clip_value)


def intersect_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def add_log1p_features(df: pd.DataFrame, columns: list[str], added: list[str]) -> None:
    for col in intersect_columns(df, columns):
        out_col = f"mlp_log1p__{col}"
        df[out_col] = np.log1p(np.clip(df[col], a_min=0.0, a_max=None))
        added.append(out_col)


def add_asinh_features(df: pd.DataFrame, columns: list[str], added: list[str]) -> None:
    for col in intersect_columns(df, columns):
        out_col = f"mlp_asinh__{col}"
        df[out_col] = signed_asinh(df[col])
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
        out_col = f"mlp_clip__{col}"
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
        out_col = f"mlp_{prefix}_z__{col}"
        df[out_col] = robust_zscore(
            df[col],
            group_mean,
            group_std,
            min_std=1e-4,
            clip_value=10.0,
        )
        added.append(out_col)


def add_group_rank_features(
    df: pd.DataFrame,
    columns: list[str],
    group_col: str,
    prefix: str,
    added: list[str],
) -> None:
    for col in intersect_columns(df, columns):
        out_col = f"mlp_{prefix}_rank__{col}"
        df[out_col] = df.groupby(group_col)[col].rank(pct=True)
        added.append(out_col)


def add_group_median_relative_features(
    df: pd.DataFrame,
    columns: list[str],
    group_col: str,
    prefix: str,
    added: list[str],
) -> None:
    for col in intersect_columns(df, columns):
        group_median = df.groupby(group_col)[col].transform("median")
        delta_col = f"mlp_{prefix}_median_delta__{col}"
        ratio_col = f"mlp_{prefix}_median_logratio__{col}"
        df[delta_col] = df[col] - group_median
        df[ratio_col] = clip_feature(
            robust_log_ratio(
                df[col].abs() + EPS,
                group_median.abs() + EPS,
                min_abs_value=1e-4,
            ),
            clip_value=10.0,
        )
        added.extend([delta_col, ratio_col])


def add_history_features(
    df: pd.DataFrame,
    columns: list[str],
    added: list[str],
) -> None:
    for col in intersect_columns(df, columns):
        group = df.groupby("stock_id")[col]
        hist_mean = group.transform(lambda s: s.shift(1).expanding().mean())
        hist_std = group.transform(lambda s: s.shift(1).expanding().std())

        mean_col = f"mlp_hist_mean__{col}"
        delta_col = f"mlp_hist_delta__{col}"
        z_col = f"mlp_hist_z__{col}"

        df[mean_col] = hist_mean
        df[delta_col] = df[col] - hist_mean
        df[z_col] = robust_zscore(
            df[col],
            hist_mean,
            hist_std,
            min_std=1e-4,
            clip_value=10.0,
        )

        added.extend([mean_col, delta_col, z_col])


def add_ema_history_features(
    df: pd.DataFrame,
    columns: list[str],
    spans: list[int],
    added: list[str],
) -> None:
    for col in intersect_columns(df, columns):
        group = df.groupby("stock_id")[col]
        for span in spans:
            ema = group.transform(
                lambda s: s.shift(1).ewm(span=span, adjust=False).mean()
            )
            ema_col = f"mlp_hist_ema{span}__{col}"
            delta_col = f"mlp_hist_ema{span}_delta__{col}"
            ratio_col = f"mlp_hist_ema{span}_logratio__{col}"
            df[ema_col] = ema
            df[delta_col] = df[col] - ema
            df[ratio_col] = clip_feature(
                robust_log_ratio(
                    df[col].abs() + EPS,
                    ema.abs() + EPS,
                    min_abs_value=1e-4,
                ),
                clip_value=10.0,
            )
            added.extend([ema_col, delta_col, ratio_col])


def add_rolling_history_features(
    df: pd.DataFrame,
    columns: list[str],
    windows: list[int],
    added: list[str],
) -> None:
    for col in intersect_columns(df, columns):
        group = df.groupby("stock_id")[col]
        for window in windows:
            rolling_mean = group.transform(
                lambda s: s.shift(1).rolling(window=window, min_periods=2).mean()
            )
            rolling_std = group.transform(
                lambda s: s.shift(1).rolling(window=window, min_periods=2).std()
            )

            mean_col = f"mlp_hist_roll{window}_mean__{col}"
            delta_col = f"mlp_hist_roll{window}_delta__{col}"
            z_col = f"mlp_hist_roll{window}_z__{col}"

            df[mean_col] = rolling_mean
            df[delta_col] = df[col] - rolling_mean
            df[z_col] = robust_zscore(
                df[col],
                rolling_mean,
                rolling_std,
                min_std=1e-4,
                clip_value=10.0,
            )
            added.extend([mean_col, delta_col, z_col])


def add_window_relative_features(df: pd.DataFrame, added: list[str]) -> None:
    candidates = [
        ("wap1_last_150_realized_vol", "wap1_last_300_realized_vol", "wap1_vol_short_long", True),
        ("mid1_last_150_realized_vol", "mid1_last_300_realized_vol", "mid1_vol_short_long", True),
        ("trade_price_last_150_realized_vol", "trade_price_last_300_realized_vol", "trade_vol_short_long", True),
        ("wap1_last_150_mean", "wap1_last_300_mean", "wap1_mean_short_long", False),
        ("mid1_last_150_mean", "mid1_last_300_mean", "mid1_mean_short_long", False),
        ("trade_price_last_150_mean", "trade_price_last_300_mean", "trade_mean_short_long", False),
    ]
    for short_col, long_col, name, allow_ratio in candidates:
        if short_col not in df.columns or long_col not in df.columns:
            continue
        delta_col = f"mlp_delta__{name}"
        df[delta_col] = df[short_col] - df[long_col]
        added.append(delta_col)
        if allow_ratio:
            ratio_col = f"mlp_logratio__{name}"
            df[ratio_col] = clip_feature(
                robust_log_ratio(
                    df[short_col].abs() + EPS,
                    df[long_col].abs() + EPS,
                    min_abs_value=1e-4,
                ),
                clip_value=10.0,
            )
            added.append(ratio_col)


def add_book_trade_interactions(df: pd.DataFrame, added: list[str]) -> None:
    specs = [
        (
            "mlp_logratio__trade_vol_to_book_vol",
            lambda x: clip_feature(
                robust_log_ratio(
                    x["realized_vol_trade_price"] + EPS,
                    x["realized_vol_wap1"] + EPS,
                    min_abs_value=1e-4,
                ),
                clip_value=10.0,
            ),
        ),
        (
            "mlp_logratio__trade_impact_to_spread",
            lambda x: clip_feature(
                robust_log_ratio(
                    x["trade_impact_mean"] + EPS,
                    x["spread1_mean"].abs() + EPS,
                    min_abs_value=1e-4,
                ),
                clip_value=10.0,
            ),
        ),
        (
            "mlp_logratio__notional_to_book_volume",
            lambda x: clip_feature(
                robust_log_ratio(
                    x["trade_notional_mean"] + EPS,
                    x["total_volume1_mean"] + EPS,
                    min_abs_value=1.0,
                ),
                clip_value=10.0,
            ),
        ),
        (
            "mlp_interact__trade_activity_imbalance",
            lambda x: x["trade_active_ratio"] * x["size_imbalance1_mean"],
        ),
        (
            "mlp_interact__trade_depth_vol",
            lambda x: x["trade_depth_ratio_mean"] * x["realized_vol_wap1"],
        ),
        (
            "mlp_interact__trade_impact_imbalance",
            lambda x: x["trade_impact_mean"] * x["size_imbalance1_mean"],
        ),
        (
            "mlp_delta__trade_price_vs_wap1_vol",
            lambda x: x["realized_vol_trade_price"] - x["realized_vol_wap1"],
        ),
        (
            "mlp_delta__trade_price_vs_mid1_vol",
            lambda x: x["realized_vol_trade_price"] - x["realized_vol_mid1"],
        ),
    ]

    for out_col, fn in specs:
        try:
            df[out_col] = fn(df)
            added.append(out_col)
        except KeyError:
            continue


def add_knn_context_features(df: pd.DataFrame, added: list[str]) -> None:
    specs = [
        (
            "mlp_delta__knn_same_vs_global_target_mean",
            lambda x: x["knn_same_stock_target_mean_k3"] - x["knn_global_target_mean_k5"],
        ),
        (
            "mlp_logratio__knn_same_vs_global_dist",
            lambda x: clip_feature(
                robust_log_ratio(
                    x["knn_same_stock_dist_mean_k3"] + EPS,
                    x["knn_global_dist_mean_k5"] + EPS,
                    min_abs_value=1e-3,
                ),
                clip_value=10.0,
            ),
        ),
        (
            "mlp_delta__realized_vol_vs_knn_global",
            lambda x: x["realized_vol_wap1"] - x["knn_global_target_mean_k5"],
        ),
        (
            "mlp_delta__trade_vol_vs_knn_same",
            lambda x: x["realized_vol_trade_price"] - x["knn_same_stock_target_mean_k3"],
        ),
        (
            "mlp_logratio__trade_vol_vs_knn_global",
            lambda x: clip_feature(
                robust_log_ratio(
                    x["realized_vol_trade_price"] + EPS,
                    x["knn_global_target_mean_k5"] + EPS,
                    min_abs_value=1e-4,
                ),
                clip_value=10.0,
            ),
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
        else feature_table_path.parent.parent / "features_mlp_view_v4"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(feature_table_path)
    df = df.sort_values(["time_id", "stock_id"]).reset_index(drop=True)

    added_columns: list[str] = []

    positive_heavy_tail_cols = [
        "book_row_count",
        "book_unique_seconds",
        "seconds_in_bucket_max",
        "wap1_std",
        "wap2_std",
        "spread1_std",
        "spread2_std",
        "mid_price1_std",
        "total_volume1_mean",
        "total_volume2_mean",
        "realized_vol_wap1",
        "realized_vol_wap2",
        "realized_vol_mid1",
        "wap1_last_150_realized_vol",
        "wap1_last_300_realized_vol",
        "mid1_last_150_realized_vol",
        "mid1_last_300_realized_vol",
        "trade_row_count",
        "trade_unique_seconds",
        "trade_active_ratio",
        "trade_price_std",
        "trade_size_sum",
        "trade_size_mean",
        "trade_size_std",
        "trade_size_max",
        "trade_log_size_mean",
        "trade_log_size_std",
        "trade_order_count_sum",
        "trade_order_count_mean",
        "trade_order_count_std",
        "trade_log_order_count_mean",
        "size_per_order_mean",
        "size_per_order_std",
        "trade_notional_sum",
        "trade_notional_mean",
        "trade_abs_return_mean",
        "trade_impact_mean",
        "trade_impact_std",
        "trade_mid_gap_std",
        "trade_depth_ratio_mean",
        "trade_depth_ratio_std",
        "realized_vol_trade_price",
        "trade_price_last_150_realized_vol",
        "trade_price_last_300_realized_vol",
        "knn_global_dist_mean_k5",
        "knn_global_dist_min_k5",
        "knn_same_stock_dist_mean_k3",
        "knn_same_stock_dist_min_k3",
    ]

    clip_cols = [
        "spread1_mean",
        "spread1_std",
        "spread2_mean",
        "spread2_std",
        "total_volume1_mean",
        "total_volume2_mean",
        "realized_vol_wap1",
        "realized_vol_wap2",
        "realized_vol_mid1",
        "trade_price_std",
        "trade_size_sum",
        "trade_size_std",
        "trade_size_max",
        "trade_order_count_sum",
        "trade_order_count_std",
        "trade_notional_sum",
        "trade_impact_mean",
        "trade_impact_std",
        "trade_depth_ratio_mean",
        "realized_vol_trade_price",
        "trade_price_last_150_realized_vol",
        "trade_price_last_300_realized_vol",
    ]

    stock_relative_cols = [
        "spread1_mean",
        "spread1_std",
        "size_imbalance1_mean",
        "size_imbalance1_std",
        "size_imbalance2_mean",
        "size_imbalance2_std",
        "realized_vol_wap1",
        "realized_vol_wap2",
        "realized_vol_mid1",
        "trade_active_ratio",
        "trade_price_std",
        "trade_impact_mean",
        "trade_depth_ratio_mean",
        "realized_vol_trade_price",
        "knn_global_target_mean_k5",
        "knn_same_stock_target_mean_k3",
    ]

    cross_section_cols = [
        "spread1_mean",
        "size_imbalance1_mean",
        "realized_vol_wap1",
        "realized_vol_mid1",
        "trade_active_ratio",
        "trade_impact_mean",
        "trade_depth_ratio_mean",
        "realized_vol_trade_price",
        "knn_global_target_mean_k5",
    ]

    cross_section_median_cols = [
        "spread1_mean",
        "realized_vol_wap1",
        "realized_vol_trade_price",
        "trade_impact_mean",
        "trade_depth_ratio_mean",
        "knn_global_target_mean_k5",
        "knn_same_stock_target_mean_k3",
    ]

    history_cols = [
        "spread1_mean",
        "size_imbalance1_mean",
        "realized_vol_wap1",
        "realized_vol_trade_price",
        "trade_active_ratio",
        "trade_impact_mean",
        "trade_depth_ratio_mean",
        "knn_same_stock_target_mean_k3",
    ]

    ema_history_cols = [
        "realized_vol_wap1",
        "realized_vol_trade_price",
        "trade_active_ratio",
        "trade_impact_mean",
        "trade_depth_ratio_mean",
        "knn_global_target_mean_k5",
        "knn_same_stock_target_mean_k3",
    ]

    rolling_history_cols = [
        "realized_vol_wap1",
        "realized_vol_trade_price",
        "trade_active_ratio",
        "trade_impact_mean",
        "knn_same_stock_target_mean_k3",
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
        "mlp_interact__trade_activity_imbalance",
        "mlp_interact__trade_impact_imbalance",
        "mlp_delta__trade_price_vs_wap1_vol",
        "mlp_delta__trade_price_vs_mid1_vol",
    ]

    add_log1p_features(df, positive_heavy_tail_cols, added_columns)
    add_winsorized_features(
        df,
        clip_cols,
        lower_q=args.clip_lower_quantile,
        upper_q=args.clip_upper_quantile,
        added=added_columns,
    )
    add_group_zscore_features(df, stock_relative_cols, "stock_id", "stock", added_columns)
    add_group_zscore_features(df, cross_section_cols, "time_id", "cross", added_columns)
    add_group_rank_features(df, cross_section_cols, "time_id", "cross", added_columns)
    add_group_median_relative_features(
        df,
        cross_section_median_cols,
        "time_id",
        "cross",
        added_columns,
    )
    add_history_features(df, history_cols, added_columns)
    add_ema_history_features(df, ema_history_cols, spans=[5, 10], added=added_columns)
    add_rolling_history_features(
        df,
        rolling_history_cols,
        windows=[5, 10],
        added=added_columns,
    )
    add_window_relative_features(df, added_columns)
    add_book_trade_interactions(df, added_columns)
    add_knn_context_features(df, added_columns)
    add_asinh_features(df, asinh_cols, added_columns)

    feature_table_out = output_dir / "optiver_features_mlp_view_v4.parquet"
    summary_out = output_dir / "summary.json"
    df.to_parquet(feature_table_out, index=False)

    summary = {
        "source_feature_table": str(feature_table_path),
        "output_dir": str(output_dir),
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "added_feature_count": int(len(added_columns)),
        "added_feature_columns": added_columns,
        "clip_lower_quantile": args.clip_lower_quantile,
        "clip_upper_quantile": args.clip_upper_quantile,
        "view_version": "v4",
        "feature_table_path": str(feature_table_out),
    }

    with summary_out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
