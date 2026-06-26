import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SAMPLE_DIR = Path(
    r"D:\Python\Datasets\optiver_realized_volatility_prediction\samples\optiver_sandbox_stocks_0-1-2-3-4-5-6-7-8-9-10-11-13-14-15-16-17-18-19-20-21-22-23-24_times_200"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fixed-length book/trade tensors for a dual-branch Optiver CNN."
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
        help="Directory to save CNN tensors. Defaults to sample_dir/cnn_tensors.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=600,
        help="Fixed seconds_in_bucket length.",
    )
    return parser.parse_args()


def build_book_tensor(
    book_df: pd.DataFrame,
    sample_keys: pd.DataFrame,
    sequence_length: int,
) -> tuple[np.ndarray, list[str]]:
    df = book_df.copy()
    df["wap1"] = (
        df["bid_price1"] * df["ask_size1"] + df["ask_price1"] * df["bid_size1"]
    ) / (df["bid_size1"] + df["ask_size1"])
    df["wap2"] = (
        df["bid_price2"] * df["ask_size2"] + df["ask_price2"] * df["bid_size2"]
    ) / (df["bid_size2"] + df["ask_size2"])
    df["spread1"] = df["ask_price1"] - df["bid_price1"]
    df["spread2"] = df["ask_price2"] - df["bid_price2"]

    book_cols = [
        "wap1",
        "wap2",
        "spread1",
        "spread2",
        "bid_size1",
        "ask_size1",
    ]
    tensor = np.zeros((len(sample_keys), len(book_cols), sequence_length), dtype=np.float32)

    key_to_idx = {
        (int(row.stock_id), int(row.time_id)): idx
        for idx, row in sample_keys.iterrows()
    }

    for (stock_id, time_id), group in df.groupby(["stock_id", "time_id"]):
        row_idx = key_to_idx[(int(stock_id), int(time_id))]
        group = group.sort_values("seconds_in_bucket").copy()
        valid_mask = (group["seconds_in_bucket"] >= 0) & (
            group["seconds_in_bucket"] < sequence_length
        )
        group = group.loc[valid_mask]
        sec_idx = group["seconds_in_bucket"].to_numpy(dtype=int)
        aligned = group[book_cols].to_numpy(dtype=np.float32)
        tensor[row_idx][:, sec_idx] = aligned.T

    return tensor, book_cols


def build_trade_tensor(
    trade_df: pd.DataFrame,
    book_df: pd.DataFrame,
    sample_keys: pd.DataFrame,
    sequence_length: int,
) -> tuple[np.ndarray, list[str]]:
    book_context = book_df.copy()
    book_context["mid_price1"] = (book_context["ask_price1"] + book_context["bid_price1"]) / 2.0
    book_context["spread1"] = book_context["ask_price1"] - book_context["bid_price1"]
    book_context["total_volume1"] = book_context["ask_size1"] + book_context["bid_size1"]
    book_context["size_imbalance1"] = (
        (book_context["bid_size1"] - book_context["ask_size1"])
        / (book_context["bid_size1"] + book_context["ask_size1"]).replace(0, np.nan)
    )
    book_context = (
        book_context.sort_values(["stock_id", "time_id", "seconds_in_bucket"])
        .groupby(["stock_id", "time_id", "seconds_in_bucket"])
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

    df = trade_df.copy()
    df = df.sort_values(["stock_id", "time_id", "seconds_in_bucket"]).reset_index(drop=True)
    df["log_price"] = np.log(df["price"])
    df["trade_return"] = (
        df.groupby(["stock_id", "time_id"])["log_price"].transform(lambda s: s.diff())
    )
    df["size_per_order"] = df["size"] / df["order_count"].replace(0, np.nan)
    df = df.merge(
        book_context,
        on=["stock_id", "time_id", "seconds_in_bucket"],
        how="left",
    )
    df["trade_mid_gap"] = df["price"] - df["mid_price1"]
    df["trade_depth_ratio"] = df["size"] / df["total_volume1"].replace(0, np.nan)
    df["trade_impact"] = df["trade_return"].abs() * df["size"]

    trade_cols = [
        "trade_return",
        "size",
        "order_count",
        "size_per_order",
        "trade_mid_gap",
        "trade_depth_ratio",
        "trade_impact",
        "size_imbalance1",
    ]
    tensor = np.zeros((len(sample_keys), len(trade_cols), sequence_length), dtype=np.float32)

    key_to_idx = {
        (int(row.stock_id), int(row.time_id)): idx
        for idx, row in sample_keys.iterrows()
    }

    aggregated = (
        df.groupby(["stock_id", "time_id", "seconds_in_bucket"])[trade_cols]
        .mean()
        .reset_index()
    )

    for (stock_id, time_id), group in aggregated.groupby(["stock_id", "time_id"]):
        row_idx = key_to_idx[(int(stock_id), int(time_id))]
        group = group.sort_values("seconds_in_bucket").copy()
        valid_mask = (group["seconds_in_bucket"] >= 0) & (
            group["seconds_in_bucket"] < sequence_length
        )
        group = group.loc[valid_mask]
        sec_idx = group["seconds_in_bucket"].to_numpy(dtype=int)
        aligned = group[trade_cols].fillna(0.0).to_numpy(dtype=np.float32)
        tensor[row_idx][:, sec_idx] = aligned.T

    return tensor, trade_cols


def main() -> None:
    args = parse_args()
    sample_dir = Path(args.sample_dir)
    output_dir = Path(args.output_dir) if args.output_dir else sample_dir / "cnn_tensors"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(sample_dir / "train_sample.csv")
    book_df = pd.read_parquet(sample_dir / "book_sample.parquet")
    trade_df = pd.read_parquet(sample_dir / "trade_sample.parquet")
    sample_keys = train_df[["stock_id", "time_id"]].reset_index(drop=True)

    book_tensor, book_cols = build_book_tensor(
        book_df=book_df,
        sample_keys=sample_keys,
        sequence_length=args.sequence_length,
    )
    trade_tensor, trade_cols = build_trade_tensor(
        trade_df=trade_df,
        book_df=book_df,
        sample_keys=sample_keys,
        sequence_length=args.sequence_length,
    )

    target = train_df["target"].to_numpy(dtype=np.float32)
    stock_id = train_df["stock_id"].to_numpy(dtype=np.int32)
    time_id = train_df["time_id"].to_numpy(dtype=np.int32)

    np.save(output_dir / "book_tensor.npy", book_tensor)
    np.save(output_dir / "trade_tensor.npy", trade_tensor)
    np.save(output_dir / "target.npy", target)
    np.save(output_dir / "stock_id.npy", stock_id)
    np.save(output_dir / "time_id.npy", time_id)

    summary = {
        "sample_dir": str(sample_dir),
        "output_dir": str(output_dir),
        "sequence_length": args.sequence_length,
        "row_count": int(len(train_df)),
        "book_tensor_shape": list(book_tensor.shape),
        "trade_tensor_shape": list(trade_tensor.shape),
        "book_channels": book_cols,
        "trade_channels": trade_cols,
        "target_shape": list(target.shape),
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
