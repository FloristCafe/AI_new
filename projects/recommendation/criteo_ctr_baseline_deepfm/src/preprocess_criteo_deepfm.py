import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess Criteo sample for a DeepFM baseline."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=r"D:\Python\Datasets\criteo_display_ad_challenge\samples\criteo_micro_2000.parquet",
        help="Input parquet file path.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=r"D:\Python\Artificial Intelligence\projects\recommendation\criteo_ctr_baseline_deepfm\artifacts",
        help="Directory for processed outputs.",
    )
    parser.add_argument(
        "--rare-threshold",
        type=int,
        default=5,
        help="Categories with frequency lower than this are mapped to UNK.",
    )
    parser.add_argument(
        "--valid-size",
        type=float,
        default=0.2,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/valid split.",
    )
    parser.add_argument(
        "--dense-bucket-count",
        type=int,
        default=32,
        help="Number of non-missing quantile buckets for each dense feature.",
    )
    return parser.parse_args()


def collect_feature_groups(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    dense_cols = [c for c in df.columns if c.startswith("integer_feature_")]
    sparse_cols = [c for c in df.columns if c.startswith("categorical_feature_")]
    return dense_cols, sparse_cols


def fill_missing_values(
    df: pd.DataFrame, dense_cols: list[str], sparse_cols: list[str]
) -> pd.DataFrame:
    df = df.copy()
    df[dense_cols] = df[dense_cols].fillna(-1)
    df[sparse_cols] = df[sparse_cols].fillna("missing").astype(str)
    return df


def fit_dense_normalizer(
    train_df: pd.DataFrame, dense_cols: list[str]
) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for col in dense_cols:
        mean = float(train_df[col].mean())
        std = float(train_df[col].std())
        if std == 0.0:
            std = 1.0
        stats[col] = {"mean": mean, "std": std}
    return stats


def apply_dense_normalizer(
    df: pd.DataFrame, dense_cols: list[str], stats: dict[str, dict[str, float]]
) -> pd.DataFrame:
    df = df.copy()
    for col in dense_cols:
        mean = stats[col]["mean"]
        std = stats[col]["std"]
        df[col] = ((df[col] - mean) / std).astype("float32")
    return df


def fit_dense_bucket_rules(
    train_df: pd.DataFrame, dense_cols: list[str], bucket_count: int
) -> dict[str, dict[str, list[float] | int]]:
    bucket_rules: dict[str, dict[str, list[float] | int]] = {}

    for col in dense_cols:
        non_missing = train_df.loc[train_df[col] != -1, col].astype("float64")
        if non_missing.empty:
            edges: list[float] = []
        else:
            transformed = np.log1p(np.clip(non_missing.to_numpy(), a_min=0.0, a_max=None))
            quantiles = np.linspace(0.0, 1.0, bucket_count + 1)[1:-1]
            raw_edges = np.quantile(transformed, quantiles)
            unique_edges = np.unique(raw_edges)
            edges = [float(x) for x in unique_edges.tolist()]

        bucket_rules[col] = {
            "edges": edges,
            # bucket 0 is reserved for missing dense values
            "vocab_size": len(edges) + 2,
        }

    return bucket_rules


def apply_dense_bucket_rules(
    df: pd.DataFrame, dense_cols: list[str], bucket_rules: dict[str, dict[str, list[float] | int]]
) -> pd.DataFrame:
    df = df.copy()

    for col in dense_cols:
        bucket_col = f"{col}_bucket"
        values = df[col].astype("float64")
        bucket_ids = np.zeros(len(df), dtype=np.int32)
        non_missing_mask = values.to_numpy() != -1

        if non_missing_mask.any():
            clipped = np.clip(values.to_numpy()[non_missing_mask], a_min=0.0, a_max=None)
            transformed = np.log1p(clipped)
            edges = np.array(bucket_rules[col]["edges"], dtype=np.float64)
            if edges.size == 0:
                bucket_ids[non_missing_mask] = 1
            else:
                bucket_ids[non_missing_mask] = np.digitize(transformed, edges, right=False) + 1

        df[bucket_col] = bucket_ids.astype("int32")

    return df


def fit_rare_category_rules(
    train_df: pd.DataFrame, sparse_cols: list[str], rare_threshold: int
) -> dict[str, set[str]]:
    rare_rules: dict[str, set[str]] = {}
    for col in sparse_cols:
        value_counts = train_df[col].value_counts()
        rare_values = set(value_counts[value_counts < rare_threshold].index.tolist())
        rare_rules[col] = rare_values
    return rare_rules


def apply_rare_category_rules(
    df: pd.DataFrame, sparse_cols: list[str], rare_rules: dict[str, set[str]]
) -> pd.DataFrame:
    df = df.copy()
    for col in sparse_cols:
        df.loc[df[col].isin(rare_rules[col]), col] = "UNK"
    return df


def fit_sparse_mappings(
    train_df: pd.DataFrame, sparse_cols: list[str]
) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    mappings: dict[str, dict[str, int]] = {}
    vocab_sizes: dict[str, int] = {}

    for col in sparse_cols:
        uniques = sorted(set(train_df[col].unique()).union({"UNK"}))
        mapping = {value: index for index, value in enumerate(uniques)}
        mappings[col] = mapping
        vocab_sizes[col] = len(mapping)

    return mappings, vocab_sizes


def apply_sparse_mappings(
    df: pd.DataFrame, sparse_cols: list[str], mappings: dict[str, dict[str, int]]
) -> pd.DataFrame:
    df = df.copy()
    for col in sparse_cols:
        mapping = mappings[col]
        unk_id = mapping["UNK"]
        df[col] = df[col].map(mapping).fillna(unk_id).astype("int32")
    return df


def build_feature_config(
    dense_cols: list[str],
    dense_bucket_cols: list[str],
    sparse_cols: list[str],
    vocab_sizes: dict[str, int],
    dense_stats: dict[str, dict[str, float]],
    dense_bucket_rules: dict[str, dict[str, list[float] | int]],
) -> dict:
    return {
        "dense_features": dense_cols,
        "dense_bucket_features": dense_bucket_cols,
        "sparse_features": sparse_cols,
        "sparse_vocab_sizes": vocab_sizes,
        "dense_normalization": dense_stats,
        "dense_bucket_rules": dense_bucket_rules,
        "label_col": "label",
    }


def build_summary(
    raw_df: pd.DataFrame,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    dense_cols: list[str],
    dense_bucket_cols: list[str],
    sparse_cols: list[str],
    vocab_sizes: dict[str, int],
    rare_threshold: int,
    dense_bucket_count: int,
) -> dict:
    dense_missing_rate = raw_df[dense_cols].isna().mean().sort_values(ascending=False)
    top_dense_missing = {k: float(v) for k, v in dense_missing_rate.head(5).items()}
    top_vocab_sizes = dict(
        sorted(vocab_sizes.items(), key=lambda item: item[1], reverse=True)[:5]
    )

    return {
        "row_count": int(len(raw_df)),
        "column_count": int(raw_df.shape[1]),
        "train_shape": [int(train_df.shape[0]), int(train_df.shape[1])],
        "valid_shape": [int(valid_df.shape[0]), int(valid_df.shape[1])],
        "train_label_rate": float(train_df["label"].mean()),
        "valid_label_rate": float(valid_df["label"].mean()),
        "dense_feature_count": len(dense_cols),
        "dense_bucket_feature_count": len(dense_bucket_cols),
        "sparse_feature_count": len(sparse_cols),
        "rare_threshold": rare_threshold,
        "dense_bucket_count": dense_bucket_count,
        "top_dense_missing_rate_before_fill": top_dense_missing,
        "top_sparse_vocab_sizes": top_vocab_sizes,
        "deepfm_ready": True,
    }


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_path}")

    raw_df = pd.read_parquet(input_path)
    dense_cols, sparse_cols = collect_feature_groups(raw_df)
    cleaned_df = fill_missing_values(raw_df, dense_cols, sparse_cols)

    train_df, valid_df = train_test_split(
        cleaned_df,
        test_size=args.valid_size,
        random_state=args.random_state,
        stratify=cleaned_df["label"],
    )

    rare_rules = fit_rare_category_rules(train_df, sparse_cols, args.rare_threshold)
    train_df = apply_rare_category_rules(train_df, sparse_cols, rare_rules)
    valid_df = apply_rare_category_rules(valid_df, sparse_cols, rare_rules)

    dense_bucket_rules = fit_dense_bucket_rules(
        train_df, dense_cols, args.dense_bucket_count
    )
    train_df = apply_dense_bucket_rules(train_df, dense_cols, dense_bucket_rules)
    valid_df = apply_dense_bucket_rules(valid_df, dense_cols, dense_bucket_rules)
    dense_bucket_cols = [f"{col}_bucket" for col in dense_cols]

    dense_stats = fit_dense_normalizer(train_df, dense_cols)
    train_df = apply_dense_normalizer(train_df, dense_cols, dense_stats)
    valid_df = apply_dense_normalizer(valid_df, dense_cols, dense_stats)

    mappings, vocab_sizes = fit_sparse_mappings(train_df, sparse_cols)
    train_df = apply_sparse_mappings(train_df, sparse_cols, mappings)
    valid_df = apply_sparse_mappings(valid_df, sparse_cols, mappings)

    feature_config = build_feature_config(
        dense_cols,
        dense_bucket_cols,
        sparse_cols,
        vocab_sizes,
        dense_stats,
        dense_bucket_rules,
    )
    summary = build_summary(
        raw_df=raw_df,
        train_df=train_df,
        valid_df=valid_df,
        dense_cols=dense_cols,
        dense_bucket_cols=dense_bucket_cols,
        sparse_cols=sparse_cols,
        vocab_sizes=vocab_sizes,
        rare_threshold=args.rare_threshold,
        dense_bucket_count=args.dense_bucket_count,
    )

    train_path = output_dir / "train_deepfm.parquet"
    valid_path = output_dir / "valid_deepfm.parquet"
    feature_config_path = output_dir / "feature_config.json"
    sparse_mapping_path = output_dir / "sparse_mappings.json"
    summary_path = output_dir / "preprocess_summary.json"

    train_df.to_parquet(train_path, index=False)
    valid_df.to_parquet(valid_path, index=False)

    with feature_config_path.open("w", encoding="utf-8") as f:
        json.dump(feature_config, f, ensure_ascii=False, indent=2)

    with sparse_mapping_path.open("w", encoding="utf-8") as f:
        json.dump(mappings, f, ensure_ascii=False, indent=2)

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("DeepFM preprocess finished.")
    print(f"Train saved to: {train_path}")
    print(f"Valid saved to: {valid_path}")
    print(f"Feature config saved to: {feature_config_path}")
    print(f"Sparse mappings saved to: {sparse_mapping_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"Train shape: {train_df.shape}")
    print(f"Valid shape: {valid_df.shape}")


if __name__ == "__main__":
    main()
