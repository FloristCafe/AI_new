import json
from pathlib import Path

import pandas as pd


DATA_ROOT = Path(r"D:\Python\Datasets\optiver_realized_volatility_prediction\raw_extracted")
TRAIN_CSV = DATA_ROOT / "train.csv"
BOOK_TRAIN = DATA_ROOT / "book_train.parquet"
TRADE_TRAIN = DATA_ROOT / "trade_train.parquet"


def load_csv_preview(path: Path, nrows: int = 5) -> tuple[pd.DataFrame, list[str], tuple[int, int]]:
    df = pd.read_csv(path, nrows=nrows)
    full_shape = pd.read_csv(path, usecols=[0]).shape[0]
    return df, df.columns.tolist(), (full_shape, len(df.columns))


def load_parquet_preview(path: Path, nrows: int = 5) -> tuple[pd.DataFrame, list[str], tuple[int, int]]:
    df = pd.read_parquet(path)
    preview = df.head(nrows).copy()
    return preview, df.columns.tolist(), df.shape


def summarize_common_keys(
    train_df: pd.DataFrame, book_preview: pd.DataFrame, trade_preview: pd.DataFrame
) -> dict:
    train_cols = set(train_df.columns)
    book_cols = set(book_preview.columns)
    trade_cols = set(trade_preview.columns)

    return {
        "train_book_common_columns": sorted(train_cols.intersection(book_cols)),
        "train_trade_common_columns": sorted(train_cols.intersection(trade_cols)),
        "book_trade_common_columns": sorted(book_cols.intersection(trade_cols)),
    }


def main() -> None:
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"train.csv not found: {TRAIN_CSV}")
    if not BOOK_TRAIN.exists():
        raise FileNotFoundError(f"book_train.parquet not found: {BOOK_TRAIN}")
    if not TRADE_TRAIN.exists():
        raise FileNotFoundError(f"trade_train.parquet not found: {TRADE_TRAIN}")

    train_preview, train_columns, train_shape = load_csv_preview(TRAIN_CSV)
    book_preview, book_columns, book_shape = load_parquet_preview(BOOK_TRAIN)
    trade_preview, trade_columns, trade_shape = load_parquet_preview(TRADE_TRAIN)

    common_keys = summarize_common_keys(train_preview, book_preview, trade_preview)

    report = {
        "train_csv": {
            "path": str(TRAIN_CSV),
            "shape": list(train_shape),
            "columns": train_columns,
            "head": train_preview.to_dict(orient="records"),
        },
        "book_train_parquet": {
            "path": str(BOOK_TRAIN),
            "shape": list(book_shape),
            "columns": book_columns,
            "head": book_preview.to_dict(orient="records"),
        },
        "trade_train_parquet": {
            "path": str(TRADE_TRAIN),
            "shape": list(trade_shape),
            "columns": trade_columns,
            "head": trade_preview.to_dict(orient="records"),
        },
        "common_keys": common_keys,
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
