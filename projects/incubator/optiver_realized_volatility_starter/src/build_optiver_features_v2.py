import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SAMPLE_DIR = Path(
    r"D:\Python\Datasets\optiver_realized_volatility_prediction\samples\optiver_sandbox_stocks_0-1-2_times_20"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an expanded Optiver feature table with richer microstructure features."
    )
    parser.add_argument(
        "--sample-dir",
        type=str,
        default=str(DEFAULT_SAMPLE_DIR),
        help="Directory containing train_sample.csv, book_sample.parquet and trade_sample.parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory to save the merged feature table. Defaults to sample_dir/features_v2.",
    )
    return parser.parse_args()


def realized_volatility(series: pd.Series) -> float:
    clean = series.dropna()
    if clean.empty:
        return 0.0
    return float(np.sqrt(np.sum(clean.to_numpy() ** 2)))


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def compute_window_features(df: pd.DataFrame, value_col: str, prefix: str) -> pd.DataFrame:
    windows = {
        f"{prefix}_last_150": df["seconds_in_bucket"] >= 450,
        f"{prefix}_last_300": df["seconds_in_bucket"] >= 300,
    }
    frames: list[pd.DataFrame] = []

    for feature_prefix, mask in windows.items():
        sub = df.loc[mask].copy()
        grouped = (
            sub.groupby(["stock_id", "time_id"])
            .agg(
                **{
                    f"{feature_prefix}_mean": (value_col, "mean"),
                    f"{feature_prefix}_std": (value_col, "std"),
                    f"{feature_prefix}_realized_vol": (value_col, realized_volatility),
                }
            )
            .reset_index()
        )
        frames.append(grouped)

    result = frames[0]
    for frame in frames[1:]:
        result = result.merge(frame, on=["stock_id", "time_id"], how="outer")
    return result


def build_book_features_v2(book_df: pd.DataFrame) -> pd.DataFrame:
    df = book_df.copy()
    df["wap1"] = (
        df["bid_price1"] * df["ask_size1"] + df["ask_price1"] * df["bid_size1"]
    ) / (df["bid_size1"] + df["ask_size1"])
    df["wap2"] = (
        df["bid_price2"] * df["ask_size2"] + df["ask_price2"] * df["bid_size2"]
    ) / (df["bid_size2"] + df["ask_size2"])
    df["spread1"] = df["ask_price1"] - df["bid_price1"]
    df["spread2"] = df["ask_price2"] - df["bid_price2"]
    df["mid_price1"] = (df["ask_price1"] + df["bid_price1"]) / 2.0
    df["mid_price2"] = (df["ask_price2"] + df["bid_price2"]) / 2.0
    df["total_volume1"] = df["ask_size1"] + df["bid_size1"]
    df["total_volume2"] = df["ask_size2"] + df["bid_size2"]
    df["size_imbalance1"] = safe_divide(
        df["bid_size1"] - df["ask_size1"], df["bid_size1"] + df["ask_size1"]
    )
    df["size_imbalance2"] = safe_divide(
        df["bid_size2"] - df["ask_size2"], df["bid_size2"] + df["ask_size2"]
    )
    df["price_spread_ratio1"] = safe_divide(df["spread1"], df["mid_price1"])
    df["price_spread_ratio2"] = safe_divide(df["spread2"], df["mid_price2"])

    df = df.sort_values(["stock_id", "time_id", "seconds_in_bucket"]).reset_index(drop=True)
    df["log_return_wap1"] = (
        df.groupby(["stock_id", "time_id"])["wap1"]
        .transform(lambda s: np.log(s).diff())
    )
    df["log_return_wap2"] = (
        df.groupby(["stock_id", "time_id"])["wap2"]
        .transform(lambda s: np.log(s).diff())
    )
    df["log_return_mid1"] = (
        df.groupby(["stock_id", "time_id"])["mid_price1"]
        .transform(lambda s: np.log(s).diff())
    )

    grouped = (
        df.groupby(["stock_id", "time_id"])
        .agg(
            book_row_count=("seconds_in_bucket", "size"),
            book_unique_seconds=("seconds_in_bucket", "nunique"),
            seconds_in_bucket_max=("seconds_in_bucket", "max"),
            wap1_mean=("wap1", "mean"),
            wap1_std=("wap1", "std"),
            wap2_mean=("wap2", "mean"),
            wap2_std=("wap2", "std"),
            spread1_mean=("spread1", "mean"),
            spread1_std=("spread1", "std"),
            spread2_mean=("spread2", "mean"),
            spread2_std=("spread2", "std"),
            mid_price1_std=("mid_price1", "std"),
            total_volume1_mean=("total_volume1", "mean"),
            total_volume2_mean=("total_volume2", "mean"),
            size_imbalance1_mean=("size_imbalance1", "mean"),
            size_imbalance1_std=("size_imbalance1", "std"),
            size_imbalance2_mean=("size_imbalance2", "mean"),
            size_imbalance2_std=("size_imbalance2", "std"),
            price_spread_ratio1_mean=("price_spread_ratio1", "mean"),
            price_spread_ratio2_mean=("price_spread_ratio2", "mean"),
            realized_vol_wap1=("log_return_wap1", realized_volatility),
            realized_vol_wap2=("log_return_wap2", realized_volatility),
            realized_vol_mid1=("log_return_mid1", realized_volatility),
        )
        .reset_index()
    )

    book_wap_windows = compute_window_features(df, "log_return_wap1", "wap1")
    book_mid_windows = compute_window_features(df, "log_return_mid1", "mid1")

    features = grouped.merge(book_wap_windows, on=["stock_id", "time_id"], how="left")
    features = features.merge(book_mid_windows, on=["stock_id", "time_id"], how="left")
    return features


def build_book_second_context(book_df: pd.DataFrame) -> pd.DataFrame:
    df = book_df.copy()
    df["mid_price1"] = (df["ask_price1"] + df["bid_price1"]) / 2.0
    df["spread1"] = df["ask_price1"] - df["bid_price1"]
    df["total_volume1"] = df["ask_size1"] + df["bid_size1"]
    df["size_imbalance1"] = safe_divide(
        df["bid_size1"] - df["ask_size1"], df["bid_size1"] + df["ask_size1"]
    )
    df = df.sort_values(["stock_id", "time_id", "seconds_in_bucket"]).reset_index(drop=True)
    return (
        df.groupby(["stock_id", "time_id", "seconds_in_bucket"])
        .tail(1)[
            [
                "stock_id",
                "time_id",
                "seconds_in_bucket",
                "mid_price1",
                "spread1",
                "total_volume1",
                "size_imbalance1",
            ]
        ]
        .reset_index(drop=True)
    )


def build_trade_features_v2(trade_df: pd.DataFrame, book_df: pd.DataFrame) -> pd.DataFrame:
    df = trade_df.copy()
    df = df.sort_values(["stock_id", "time_id", "seconds_in_bucket"]).reset_index(drop=True)
    df["log_return_trade_price"] = (
        df.groupby(["stock_id", "time_id"])["price"]
        .transform(lambda s: np.log(s).diff())
    )
    df["size_per_order"] = safe_divide(df["size"], df["order_count"])
    df["trade_notional"] = df["price"] * df["size"]
    df["log_size"] = np.log1p(df["size"])
    df["log_order_count"] = np.log1p(df["order_count"])
    df["trade_abs_return"] = df["log_return_trade_price"].abs()

    book_second = build_book_second_context(book_df)
    df = df.merge(
        book_second,
        on=["stock_id", "time_id", "seconds_in_bucket"],
        how="left",
    )
    df["trade_mid_gap"] = df["price"] - df["mid_price1"]
    df["trade_depth_ratio"] = safe_divide(df["size"], df["total_volume1"])
    df["trade_impact"] = df["trade_abs_return"] * df["size"]
    df["spread_order_interaction"] = df["spread1"] * df["order_count"]
    df["imbalance_size_interaction"] = df["size_imbalance1"] * df["size"]

    grouped = (
        df.groupby(["stock_id", "time_id"])
        .agg(
            trade_row_count=("seconds_in_bucket", "size"),
            trade_unique_seconds=("seconds_in_bucket", "nunique"),
            trade_active_ratio=("seconds_in_bucket", lambda s: s.nunique() / 600.0),
            trade_price_mean=("price", "mean"),
            trade_price_std=("price", "std"),
            trade_size_sum=("size", "sum"),
            trade_size_mean=("size", "mean"),
            trade_size_std=("size", "std"),
            trade_size_max=("size", "max"),
            trade_log_size_mean=("log_size", "mean"),
            trade_log_size_std=("log_size", "std"),
            trade_order_count_sum=("order_count", "sum"),
            trade_order_count_mean=("order_count", "mean"),
            trade_order_count_std=("order_count", "std"),
            trade_log_order_count_mean=("log_order_count", "mean"),
            size_per_order_mean=("size_per_order", "mean"),
            size_per_order_std=("size_per_order", "std"),
            trade_notional_sum=("trade_notional", "sum"),
            trade_notional_mean=("trade_notional", "mean"),
            trade_abs_return_mean=("trade_abs_return", "mean"),
            trade_impact_mean=("trade_impact", "mean"),
            trade_impact_std=("trade_impact", "std"),
            trade_mid_gap_mean=("trade_mid_gap", "mean"),
            trade_mid_gap_std=("trade_mid_gap", "std"),
            trade_depth_ratio_mean=("trade_depth_ratio", "mean"),
            trade_depth_ratio_std=("trade_depth_ratio", "std"),
            spread_order_interaction_mean=("spread_order_interaction", "mean"),
            imbalance_size_interaction_mean=("imbalance_size_interaction", "mean"),
            realized_vol_trade_price=("log_return_trade_price", realized_volatility),
        )
        .reset_index()
    )

    trade_windows = compute_window_features(
        df, "log_return_trade_price", "trade_price"
    )
    return grouped.merge(trade_windows, on=["stock_id", "time_id"], how="left")


def main() -> None:
    args = parse_args()
    sample_dir = Path(args.sample_dir)
    output_dir = Path(args.output_dir) if args.output_dir else sample_dir / "features_v2"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = sample_dir / "train_sample.csv"
    book_path = sample_dir / "book_sample.parquet"
    trade_path = sample_dir / "trade_sample.parquet"

    train_df = pd.read_csv(train_path)
    book_df = pd.read_parquet(book_path)
    trade_df = pd.read_parquet(trade_path)

    book_features = build_book_features_v2(book_df)
    trade_features = build_trade_features_v2(trade_df, book_df)

    feature_df = train_df.merge(book_features, on=["stock_id", "time_id"], how="left")
    feature_df = feature_df.merge(trade_features, on=["stock_id", "time_id"], how="left")

    feature_table_path = output_dir / "optiver_features_v2.parquet"
    summary_path = output_dir / "summary.json"
    feature_df.to_parquet(feature_table_path, index=False)

    summary = {
        "sample_dir": str(sample_dir),
        "output_dir": str(output_dir),
        "feature_table_shape": list(feature_df.shape),
        "feature_columns": feature_df.columns.tolist(),
        "feature_table_path": str(feature_table_path),
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
