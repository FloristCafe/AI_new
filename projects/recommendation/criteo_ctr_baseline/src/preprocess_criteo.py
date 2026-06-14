import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    # Expose file paths and thresholds as CLI args so we can rerun experiments
    # without editing source code each time.
    parser = argparse.ArgumentParser(
        description="Preprocess a Criteo parquet sample for a first CTR baseline."
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
        default=r"D:\Python\Artificial Intelligence\projects\recommendation\criteo_ctr_baseline\artifacts",
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
    return parser.parse_args()


def collect_feature_groups(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    # Use naming rules instead of hardcoding all 39 feature names.
    dense_cols = [c for c in df.columns if c.startswith("integer_feature_")]
    sparse_cols = [c for c in df.columns if c.startswith("categorical_feature_")]
    return dense_cols, sparse_cols


def fill_missing_values(
    df: pd.DataFrame, dense_cols: list[str], sparse_cols: list[str]
) -> pd.DataFrame:
    # Work on a copy so the raw frame stays intact for debugging and summaries.
    df = df.copy()

    # For CTR dense features, -1 is a clearer sentinel than 0 because many
    # dense fields are count-like or non-negative. This lets the model
    # distinguish "missing" from a real observed zero in the first baseline.
    df[dense_cols] = df[dense_cols].fillna(-1)

    # Sparse features are normalized to strings so category handling is stable.
    df[sparse_cols] = df[sparse_cols].fillna("missing").astype(str)
    return df


def fit_rare_category_rules(
    train_df: pd.DataFrame, sparse_cols: list[str], rare_threshold: int
) -> dict[str, set[str]]:
    # Learn rare-category rules from the training split only.
    rare_rules: dict[str, set[str]] = {}

    for col in sparse_cols:
        value_counts = train_df[col].value_counts()
        rare_values = set(value_counts[value_counts < rare_threshold].index.tolist())
        rare_rules[col] = rare_values

    return rare_rules


def apply_rare_category_rules(
    df: pd.DataFrame, sparse_cols: list[str], rare_rules: dict[str, set[str]]
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    # Apply the training-derived rare-category rules to any split.
    df = df.copy()
    rare_stats: dict[str, dict[str, int]] = {}

    for col in sparse_cols:
        before_unique = int(df[col].nunique())
        rare_values = rare_rules[col]
        df.loc[df[col].isin(rare_values), col] = "UNK"
        after_unique = int(df[col].nunique())
        rare_stats[col] = {
            "unique_before": before_unique,
            "unique_after": after_unique,
            "rare_value_count_from_train": int(len(rare_values)),
        }

    return df, rare_stats


def fit_category_mappings(
    train_df: pd.DataFrame, sparse_cols: list[str]
) -> dict[str, dict[str, int]]:
    # Learn category-to-id mappings from the training split only.
    feature_maps: dict[str, dict[str, int]] = {}

    for col in sparse_cols:
        # Always reserve an UNK bucket so validation-time unseen categories
        # have a stable fallback id even when train never produced UNK here.
        uniques = sorted(set(train_df[col].unique()).union({"UNK"}))
        feature_maps[col] = {value: index for index, value in enumerate(uniques)}

    return feature_maps


def apply_category_mappings(
    df: pd.DataFrame,
    sparse_cols: list[str],
    feature_maps: dict[str, dict[str, int]],
) -> pd.DataFrame:
    # When a validation value was unseen in training, map it to the UNK id.
    df = df.copy()

    for col in sparse_cols:
        mapping = feature_maps[col]
        if "UNK" not in mapping:
            raise ValueError(f"Column {col} is missing an UNK category in feature map.")
        unk_id = mapping["UNK"]
        df[col] = df[col].map(mapping).fillna(unk_id).astype("int32")

    return df


def build_summary(
    raw_df: pd.DataFrame,
    dense_cols: list[str],
    sparse_cols: list[str],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    rare_threshold: int,
    train_rare_stats: dict[str, dict[str, int]],
    valid_rare_stats: dict[str, dict[str, int]],
) -> dict:
    # Keep a compact experiment record for later comparison.
    dense_missing_rate = raw_df[dense_cols].isna().mean().sort_values(ascending=False)
    top_dense_missing = {k: float(v) for k, v in dense_missing_rate.head(5).items()}

    top_sparse_cardinality_train = {
        k: int(v)
        for k, v in train_df[sparse_cols]
        .nunique()
        .sort_values(ascending=False)
        .head(5)
        .items()
    }

    top_sparse_cardinality_valid = {
        k: int(v)
        for k, v in valid_df[sparse_cols]
        .nunique()
        .sort_values(ascending=False)
        .head(5)
        .items()
    }

    return {
        "row_count": int(len(raw_df)),
        "column_count": int(raw_df.shape[1]),
        "dense_feature_count": len(dense_cols),
        "sparse_feature_count": len(sparse_cols),
        "label_rate": float(raw_df["label"].mean()),
        "train_shape": [int(train_df.shape[0]), int(train_df.shape[1])],
        "valid_shape": [int(valid_df.shape[0]), int(valid_df.shape[1])],
        "train_label_rate": float(train_df["label"].mean()),
        "valid_label_rate": float(valid_df["label"].mean()),
        "rare_threshold": rare_threshold,
        "split_before_category_fit": True,
        "top_dense_missing_rate_before_fill": top_dense_missing,
        "top_sparse_cardinality_train_after_encoding": top_sparse_cardinality_train,
        "top_sparse_cardinality_valid_after_encoding": top_sparse_cardinality_valid,
        "train_rare_fold_examples": dict(list(train_rare_stats.items())[:5]),
        "valid_rare_fold_examples": dict(list(valid_rare_stats.items())[:5]),
    }


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_path}")

    # Step 1: load raw parquet into an in-memory table.
    raw_df = pd.read_parquet(input_path)
    dense_cols, sparse_cols = collect_feature_groups(raw_df)

    # Step 2: run basic missing-value cleanup before splitting.
    cleaned_df = fill_missing_values(raw_df, dense_cols, sparse_cols)

    # Step 3: split first, so later category rules are learned only on training data.
    train_df, valid_df = train_test_split(
        cleaned_df,
        test_size=args.valid_size,
        random_state=args.random_state,
        stratify=cleaned_df["label"],
    )

    # Step 4: learn rare-category rules on train, then apply to both splits.
    rare_rules = fit_rare_category_rules(train_df, sparse_cols, args.rare_threshold)
    train_folded_df, train_rare_stats = apply_rare_category_rules(
        train_df, sparse_cols, rare_rules
    )
    valid_folded_df, valid_rare_stats = apply_rare_category_rules(
        valid_df, sparse_cols, rare_rules
    )

    # Step 5: learn integer mappings on train, then apply to both splits.
    feature_maps = fit_category_mappings(train_folded_df, sparse_cols)
    train_encoded_df = apply_category_mappings(
        train_folded_df, sparse_cols, feature_maps
    )
    valid_encoded_df = apply_category_mappings(
        valid_folded_df, sparse_cols, feature_maps
    )

    summary = build_summary(
        raw_df,
        dense_cols,
        sparse_cols,
        train_encoded_df,
        valid_encoded_df,
        args.rare_threshold,
        train_rare_stats,
        valid_rare_stats,
    )

    train_path = output_dir / "train_processed.parquet"
    valid_path = output_dir / "valid_processed.parquet"
    feature_map_path = output_dir / "feature_maps.json"
    summary_path = output_dir / "preprocess_summary.json"

    train_encoded_df.to_parquet(train_path, index=False)
    valid_encoded_df.to_parquet(valid_path, index=False)

    with feature_map_path.open("w", encoding="utf-8") as f:
        json.dump(feature_maps, f, ensure_ascii=False, indent=2)

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Preprocess finished.")
    print(f"Input: {input_path}")
    print(f"Train saved to: {train_path}")
    print(f"Valid saved to: {valid_path}")
    print(f"Feature maps saved to: {feature_map_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"Train shape: {train_encoded_df.shape}")
    print(f"Valid shape: {valid_encoded_df.shape}")
    print(f"Train label rate: {train_encoded_df['label'].mean():.6f}")
    print(f"Valid label rate: {valid_encoded_df['label'].mean():.6f}")


if __name__ == "__main__":
    main()
