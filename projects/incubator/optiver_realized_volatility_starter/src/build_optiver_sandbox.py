import argparse
import json
from pathlib import Path

import pandas as pd


DATA_ROOT = Path(r"D:\Python\Datasets\optiver_realized_volatility_prediction\raw_extracted")
SAMPLE_ROOT = Path(r"D:\Python\Datasets\optiver_realized_volatility_prediction\samples")


def parse_int_list(value: str | None) -> list[int] | None:
    if value is None or value.strip() == "":
        return None
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def format_stock_tag(stock_ids: list[int]) -> str:
    if not stock_ids:
        raise ValueError("stock_ids cannot be empty when building the output tag.")

    sorted_ids = sorted(stock_ids)
    is_contiguous = all(
        right - left == 1 for left, right in zip(sorted_ids[:-1], sorted_ids[1:])
    )
    if is_contiguous:
        return f"{sorted_ids[0]}-{sorted_ids[-1]}"
    return "-".join(str(x) for x in sorted_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a small Optiver train/book/trade sandbox sample."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(DATA_ROOT),
        help="Directory containing train.csv, book_train.parquet and trade_train.parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output directory. Defaults to a generated folder under the dataset samples directory.",
    )
    parser.add_argument(
        "--stock-count",
        type=int,
        default=3,
        help="How many stock_ids to keep when --stock-ids is not given.",
    )
    parser.add_argument(
        "--time-ids-per-stock",
        type=int,
        default=20,
        help="How many time_id samples to keep per stock when building the sandbox.",
    )
    parser.add_argument(
        "--stock-ids",
        type=str,
        default="",
        help="Optional comma-separated explicit stock_id list.",
    )
    parser.add_argument(
        "--sort-order",
        type=str,
        default="time_id",
        choices=["time_id", "target"],
        help="How to pick the per-stock sandbox rows from train.csv.",
    )
    return parser.parse_args()


def select_train_rows(
    train_df: pd.DataFrame,
    explicit_stock_ids: list[int] | None,
    stock_count: int,
    time_ids_per_stock: int,
    sort_order: str,
) -> pd.DataFrame:
    if explicit_stock_ids is None:
        selected_stock_ids = sorted(train_df["stock_id"].unique())[:stock_count]
    else:
        selected_stock_ids = explicit_stock_ids

    selected_frames: list[pd.DataFrame] = []
    for stock_id in selected_stock_ids:
        stock_df = train_df.loc[train_df["stock_id"] == stock_id].copy()
        stock_df = stock_df.sort_values(sort_order, ascending=True)
        selected_frames.append(stock_df.head(time_ids_per_stock))

    sample_train = pd.concat(selected_frames, ignore_index=True)
    return sample_train.sort_values(["stock_id", "time_id"]).reset_index(drop=True)


def filter_parquet_by_keys(
    parquet_path: Path, stock_ids: list[int], sample_keys: pd.DataFrame
) -> pd.DataFrame:
    time_ids = sorted(sample_keys["time_id"].unique().tolist())
    df = pd.read_parquet(
        parquet_path,
        filters=[
            ("stock_id", "in", stock_ids),
            ("time_id", "in", time_ids),
        ],
    )
    merged = df.merge(sample_keys, on=["stock_id", "time_id"], how="inner")
    return merged.sort_values(["stock_id", "time_id", "seconds_in_bucket"]).reset_index(
        drop=True
    )


def build_output_dir(
    explicit_output_dir: str, stock_ids: list[int], time_ids_per_stock: int
) -> Path:
    if explicit_output_dir:
        output_dir = Path(explicit_output_dir)
    else:
        stock_tag = format_stock_tag(stock_ids)
        output_dir = SAMPLE_ROOT / f"optiver_sandbox_stocks_{stock_tag}_times_{time_ids_per_stock}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    train_csv = data_root / "train.csv"
    book_train = data_root / "book_train.parquet"
    trade_train = data_root / "trade_train.parquet"

    train_df = pd.read_csv(train_csv)
    explicit_stock_ids = parse_int_list(args.stock_ids)
    sample_train = select_train_rows(
        train_df=train_df,
        explicit_stock_ids=explicit_stock_ids,
        stock_count=args.stock_count,
        time_ids_per_stock=args.time_ids_per_stock,
        sort_order=args.sort_order,
    )

    selected_stock_ids = sorted(sample_train["stock_id"].unique().tolist())
    sample_keys = sample_train[["stock_id", "time_id"]].drop_duplicates().copy()

    sample_book = filter_parquet_by_keys(book_train, selected_stock_ids, sample_keys)
    sample_trade = filter_parquet_by_keys(trade_train, selected_stock_ids, sample_keys)

    output_dir = build_output_dir(
        args.output_dir, selected_stock_ids, args.time_ids_per_stock
    )
    train_out = output_dir / "train_sample.csv"
    book_out = output_dir / "book_sample.parquet"
    trade_out = output_dir / "trade_sample.parquet"
    summary_out = output_dir / "summary.json"

    sample_train.to_csv(train_out, index=False)
    sample_book.to_parquet(book_out, index=False)
    sample_trade.to_parquet(trade_out, index=False)

    summary = {
        "selected_stock_ids": selected_stock_ids,
        "time_ids_per_stock": args.time_ids_per_stock,
        "sort_order": args.sort_order,
        "sample_train_shape": list(sample_train.shape),
        "sample_book_shape": list(sample_book.shape),
        "sample_trade_shape": list(sample_trade.shape),
        "sample_time_id_count": int(sample_keys.shape[0]),
        "target_mean": float(sample_train["target"].mean()),
        "target_std": float(sample_train["target"].std()),
        "train_out": str(train_out),
        "book_out": str(book_out),
        "trade_out": str(trade_out),
    }
    with summary_out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
