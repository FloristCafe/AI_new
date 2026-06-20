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
        description="Build a first Optiver baseline feature table from a sandbox sample."
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
        help="Directory to save the merged feature table. Defaults to sample_dir/features_baseline.",
    )
    return parser.parse_args()


def realized_volatility(series: pd.Series) -> float:
    clean = series.dropna()
    if clean.empty:
        return 0.0
    return float(np.sqrt(np.sum(clean.to_numpy() ** 2)))


def build_book_features(book_df: pd.DataFrame) -> pd.DataFrame:
    df = book_df.copy()

    df["wap1"] = (
        df["bid_price1"] * df["ask_size1"] + df["ask_price1"] * df["bid_size1"]
    ) / (df["bid_size1"] + df["ask_size1"])
    df["wap2"] = (
        df["bid_price2"] * df["ask_size2"] + df["ask_price2"] * df["bid_size2"]
    ) / (df["bid_size2"] + df["ask_size2"])
    df["spread1"] = df["ask_price1"] - df["bid_price1"]
    df["spread2"] = df["ask_price2"] - df["bid_price2"]
    df["total_volume1"] = df["ask_size1"] + df["bid_size1"]
    df["total_volume2"] = df["ask_size2"] + df["bid_size2"]
    df["size_imbalance1"] = (df["bid_size1"] - df["ask_size1"]) / (
        df["bid_size1"] + df["ask_size1"]
    )
    df["size_imbalance2"] = (df["bid_size2"] - df["ask_size2"]) / (
        df["bid_size2"] + df["ask_size2"]
    )

    df = df.sort_values(["stock_id", "time_id", "seconds_in_bucket"]).reset_index(drop=True)
    df["log_return_wap1"] = (
        df.groupby(["stock_id", "time_id"])["wap1"]
        .transform(lambda s: np.log(s).diff())
    )
    df["log_return_wap2"] = (
        df.groupby(["stock_id", "time_id"])["wap2"]
        .transform(lambda s: np.log(s).diff())
    )

    book_features = (
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
            total_volume1_mean=("total_volume1", "mean"),
            total_volume2_mean=("total_volume2", "mean"),
            size_imbalance1_mean=("size_imbalance1", "mean"),
            size_imbalance2_mean=("size_imbalance2", "mean"),
            realized_vol_wap1=("log_return_wap1", realized_volatility),
            realized_vol_wap2=("log_return_wap2", realized_volatility),
        )
        .reset_index()
    )

    return book_features


def build_trade_features(trade_df: pd.DataFrame) -> pd.DataFrame:
    df = trade_df.copy()
    df = df.sort_values(["stock_id", "time_id", "seconds_in_bucket"]).reset_index(drop=True)
    df["log_return_trade_price"] = (
        df.groupby(["stock_id", "time_id"])["price"]
        .transform(lambda s: np.log(s).diff())
    )

    trade_features = (
        df.groupby(["stock_id", "time_id"])
        .agg(
            trade_row_count=("seconds_in_bucket", "size"),
            trade_unique_seconds=("seconds_in_bucket", "nunique"),
            trade_price_mean=("price", "mean"),
            trade_price_std=("price", "std"),
            trade_size_sum=("size", "sum"),
            trade_size_mean=("size", "mean"),
            trade_size_std=("size", "std"),
            trade_size_max=("size", "max"),
            trade_order_count_sum=("order_count", "sum"),
            trade_order_count_mean=("order_count", "mean"),
            realized_vol_trade_price=("log_return_trade_price", realized_volatility),
        )
        .reset_index()
    )

    return trade_features


def main() -> None:
    args = parse_args()
    sample_dir = Path(args.sample_dir)
    output_dir = Path(args.output_dir) if args.output_dir else sample_dir / "features_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = sample_dir / "train_sample.csv"
    book_path = sample_dir / "book_sample.parquet"
    trade_path = sample_dir / "trade_sample.parquet"

    train_df = pd.read_csv(train_path)
    book_df = pd.read_parquet(book_path)
    trade_df = pd.read_parquet(trade_path)

    book_features = build_book_features(book_df)
    trade_features = build_trade_features(trade_df)

    feature_df = train_df.merge(book_features, on=["stock_id", "time_id"], how="left")
    feature_df = feature_df.merge(trade_features, on=["stock_id", "time_id"], how="left")

    feature_table_path = output_dir / "optiver_baseline_features.parquet"
    summary_path = output_dir / "summary.json"

    feature_df.to_parquet(feature_table_path, index=False)

    summary = {
        "sample_dir": str(sample_dir),
        "output_dir": str(output_dir),
        "train_shape": list(train_df.shape),
        "book_shape": list(book_df.shape),
        "trade_shape": list(trade_df.shape),
        "book_feature_shape": list(book_features.shape),
        "trade_feature_shape": list(trade_features.shape),
        "feature_table_shape": list(feature_df.shape),
        "feature_columns": feature_df.columns.tolist(),
        "feature_table_path": str(feature_table_path),
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
