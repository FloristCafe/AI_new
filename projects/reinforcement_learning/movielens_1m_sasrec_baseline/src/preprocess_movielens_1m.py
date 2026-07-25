import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess MovieLens-1M interactions into SASRec sequence artifacts."
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default=r"D:\Python\Datasets\movielens_1m\processed\interactions.csv",
        help="Path to cleaned interaction CSV with user_id, movie_id, timestamp.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=r"D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_baseline\artifacts\preprocessed",
        help="Directory for preprocessed sequence artifacts.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=50,
        help="Maximum sequence length used by SASRec.",
    )
    parser.add_argument(
        "--min-user-interactions",
        type=int,
        default=5,
        help="Minimum interactions required to keep a user.",
    )
    parser.add_argument(
        "--min-item-interactions",
        type=int,
        default=1,
        help="Minimum interactions required to keep an item.",
    )
    parser.add_argument(
        "--sample-user-limit",
        type=int,
        default=0,
        help="Optional cap on the number of users for faster debugging. Set 0 to keep all users.",
    )
    return parser.parse_args()


def load_interactions(input_path: Path) -> list[tuple[int, int, int]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    rows: list[tuple[int, int, int]] = []
    with input_path.open("r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)
        required_columns = {"user_id", "movie_id", "timestamp"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                "Input CSV must contain columns: user_id, movie_id, timestamp"
            )

        for row in reader:
            rows.append(
                (
                    int(row["user_id"]),
                    int(row["movie_id"]),
                    int(row["timestamp"]),
                )
            )

    if not rows:
        raise ValueError(f"No interactions found in: {input_path}")

    return rows


def build_sorted_user_histories(
    rows: list[tuple[int, int, int]]
) -> dict[int, list[tuple[int, int]]]:
    user_histories: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for user_id, movie_id, timestamp in rows:
        user_histories[user_id].append((timestamp, movie_id))

    sorted_histories: dict[int, list[tuple[int, int]]] = {}
    for user_id, events in user_histories.items():
        events.sort(key=lambda x: (x[0], x[1]))
        sorted_histories[user_id] = events

    return sorted_histories


def filter_histories(
    user_histories: dict[int, list[tuple[int, int]]],
    min_user_interactions: int,
    min_item_interactions: int,
) -> tuple[dict[int, list[tuple[int, int]]], dict[str, int]]:
    item_counts: dict[int, int] = defaultdict(int)
    for events in user_histories.values():
        for _, movie_id in events:
            item_counts[movie_id] += 1

    filtered_histories: dict[int, list[tuple[int, int]]] = {}
    removed_user_count = 0
    for user_id, events in user_histories.items():
        kept_events = [
            (timestamp, movie_id)
            for timestamp, movie_id in events
            if item_counts[movie_id] >= min_item_interactions
        ]
        if len(kept_events) >= min_user_interactions:
            filtered_histories[user_id] = kept_events
        else:
            removed_user_count += 1

    stats = {
        "removed_user_count": removed_user_count,
        "kept_user_count": len(filtered_histories),
        "kept_item_count": len(
            {
                movie_id
                for events in filtered_histories.values()
                for _, movie_id in events
            }
        ),
    }
    return filtered_histories, stats


def maybe_limit_users(
    user_histories: dict[int, list[tuple[int, int]]],
    sample_user_limit: int,
) -> dict[int, list[tuple[int, int]]]:
    if sample_user_limit <= 0 or sample_user_limit >= len(user_histories):
        return user_histories

    limited_histories: dict[int, list[tuple[int, int]]] = {}
    for user_id in sorted(user_histories)[:sample_user_limit]:
        limited_histories[user_id] = user_histories[user_id]
    return limited_histories


def remap_ids(
    user_histories: dict[int, list[tuple[int, int]]]
) -> tuple[dict[int, list[int]], dict[int, int], dict[int, int]]:
    user_mapping: dict[int, int] = {}
    item_mapping: dict[int, int] = {}
    remapped_histories: dict[int, list[int]] = {}

    next_user_id = 1
    next_item_id = 1
    for raw_user_id in sorted(user_histories):
        user_mapping[raw_user_id] = next_user_id
        next_user_id += 1

        remapped_items: list[int] = []
        for _, raw_item_id in user_histories[raw_user_id]:
            if raw_item_id not in item_mapping:
                item_mapping[raw_item_id] = next_item_id
                next_item_id += 1
            remapped_items.append(item_mapping[raw_item_id])

        remapped_histories[user_mapping[raw_user_id]] = remapped_items

    return remapped_histories, user_mapping, item_mapping


def pad_sequence(sequence: list[int], max_seq_len: int) -> np.ndarray:
    padded = np.zeros(max_seq_len, dtype=np.int64)
    if not sequence:
        return padded

    truncated = sequence[-max_seq_len:]
    padded[-len(truncated) :] = np.asarray(truncated, dtype=np.int64)
    return padded


def pad_aligned_sequences(
    input_sequence: list[int],
    positive_sequence: list[int],
    max_seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(input_sequence) != len(positive_sequence):
        raise ValueError("Input and positive sequences must have the same length.")

    padded_input = np.zeros(max_seq_len, dtype=np.int64)
    padded_positive = np.zeros(max_seq_len, dtype=np.int64)
    if not input_sequence:
        return padded_input, padded_positive

    truncated_input = input_sequence[-max_seq_len:]
    truncated_positive = positive_sequence[-max_seq_len:]
    padded_input[-len(truncated_input) :] = np.asarray(truncated_input, dtype=np.int64)
    padded_positive[-len(truncated_positive) :] = np.asarray(
        truncated_positive,
        dtype=np.int64,
    )
    return padded_input, padded_positive


def build_training_samples(
    user_sequences: dict[int, list[int]],
    max_seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    input_sequences: list[np.ndarray] = []
    target_items: list[int] = []
    sample_user_ids: list[int] = []

    for user_id, items in user_sequences.items():
        train_items = items[:-2]
        if len(train_items) < 2:
            continue

        for next_index in range(1, len(train_items)):
            history = train_items[:next_index]
            target_item = train_items[next_index]
            input_sequences.append(pad_sequence(history, max_seq_len))
            target_items.append(target_item)
            sample_user_ids.append(user_id)

    if not input_sequences:
        raise ValueError(
            "No training samples were created. Check the interaction thresholds."
        )

    return (
        np.stack(input_sequences).astype(np.int64),
        np.asarray(target_items, dtype=np.int64),
        np.asarray(sample_user_ids, dtype=np.int64),
    )


def build_sequence_training_windows(
    user_sequences: dict[int, list[int]],
    max_seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    input_sequences: list[np.ndarray] = []
    positive_sequences: list[np.ndarray] = []
    sample_user_ids: list[int] = []

    for user_id, items in user_sequences.items():
        train_items = items[:-2]
        if len(train_items) < 2:
            continue

        for start_idx in range(0, len(train_items) - 1, max_seq_len):
            window = train_items[start_idx : start_idx + max_seq_len + 1]
            if len(window) < 2:
                continue

            input_sequence = window[:-1]
            positive_sequence = window[1:]
            padded_input, padded_positive = pad_aligned_sequences(
                input_sequence=input_sequence,
                positive_sequence=positive_sequence,
                max_seq_len=max_seq_len,
            )
            input_sequences.append(padded_input)
            positive_sequences.append(padded_positive)
            sample_user_ids.append(user_id)

    if not input_sequences:
        raise ValueError(
            "No sequence-level training windows were created. "
            "Check the interaction thresholds."
        )

    return (
        np.stack(input_sequences).astype(np.int64),
        np.stack(positive_sequences).astype(np.int64),
        np.asarray(sample_user_ids, dtype=np.int64),
    )


def build_eval_split(
    user_sequences: dict[int, list[int]],
    max_seq_len: int,
    split: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    history_sequences: list[np.ndarray] = []
    target_items: list[int] = []
    user_ids: list[int] = []

    for user_id, items in user_sequences.items():
        if split == "valid":
            history = items[:-2]
            target_item = items[-2]
        elif split == "test":
            history = items[:-1]
            target_item = items[-1]
        else:
            raise ValueError(f"Unsupported split: {split}")

        history_sequences.append(pad_sequence(history, max_seq_len))
        target_items.append(target_item)
        user_ids.append(user_id)

    return (
        np.stack(history_sequences).astype(np.int64),
        np.asarray(target_items, dtype=np.int64),
        np.asarray(user_ids, dtype=np.int64),
    )


def save_npz(
    path: Path,
    sequences: np.ndarray,
    targets: np.ndarray,
    user_ids: np.ndarray,
) -> None:
    np.savez_compressed(
        path,
        input_ids=sequences,
        target_ids=targets,
        user_ids=user_ids,
    )


def save_sequence_supervision_npz(
    path: Path,
    input_ids: np.ndarray,
    positive_ids: np.ndarray,
    user_ids: np.ndarray,
) -> None:
    np.savez_compressed(
        path,
        input_ids=input_ids,
        positive_ids=positive_ids,
        user_ids=user_ids,
    )


def save_mapping_csv(path: Path, header: tuple[str, str], mapping: dict[int, int]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(header)
        for raw_id, mapped_id in mapping.items():
            writer.writerow([raw_id, mapped_id])


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_interactions(input_path)
    initial_user_count = len({user_id for user_id, _, _ in rows})
    initial_item_count = len({movie_id for _, movie_id, _ in rows})

    user_histories = build_sorted_user_histories(rows)
    user_histories, filter_stats = filter_histories(
        user_histories=user_histories,
        min_user_interactions=args.min_user_interactions,
        min_item_interactions=args.min_item_interactions,
    )
    user_histories = maybe_limit_users(
        user_histories=user_histories,
        sample_user_limit=args.sample_user_limit,
    )

    remapped_sequences, user_mapping, item_mapping = remap_ids(user_histories)

    train_input_ids, train_target_ids, train_user_ids = build_training_samples(
        user_sequences=remapped_sequences,
        max_seq_len=args.max_seq_len,
    )
    (
        train_sequence_input_ids,
        train_sequence_positive_ids,
        train_sequence_user_ids,
    ) = build_sequence_training_windows(
        user_sequences=remapped_sequences,
        max_seq_len=args.max_seq_len,
    )
    valid_input_ids, valid_target_ids, valid_user_ids = build_eval_split(
        user_sequences=remapped_sequences,
        max_seq_len=args.max_seq_len,
        split="valid",
    )
    test_input_ids, test_target_ids, test_user_ids = build_eval_split(
        user_sequences=remapped_sequences,
        max_seq_len=args.max_seq_len,
        split="test",
    )

    train_path = output_dir / "train_sequences.npz"
    train_sequence_supervision_path = output_dir / "train_sequence_supervision.npz"
    valid_path = output_dir / "valid_sequences.npz"
    test_path = output_dir / "test_sequences.npz"
    metadata_path = output_dir / "metadata.json"
    user_mapping_path = output_dir / "user_id_mapping.csv"
    item_mapping_path = output_dir / "item_id_mapping.csv"

    save_npz(train_path, train_input_ids, train_target_ids, train_user_ids)
    save_sequence_supervision_npz(
        train_sequence_supervision_path,
        train_sequence_input_ids,
        train_sequence_positive_ids,
        train_sequence_user_ids,
    )
    save_npz(valid_path, valid_input_ids, valid_target_ids, valid_user_ids)
    save_npz(test_path, test_input_ids, test_target_ids, test_user_ids)
    save_mapping_csv(user_mapping_path, ("raw_user_id", "mapped_user_id"), user_mapping)
    save_mapping_csv(item_mapping_path, ("raw_movie_id", "mapped_movie_id"), item_mapping)

    metadata = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "max_seq_len": args.max_seq_len,
        "min_user_interactions": args.min_user_interactions,
        "min_item_interactions": args.min_item_interactions,
        "sample_user_limit": args.sample_user_limit,
        "raw_interaction_count": len(rows),
        "raw_user_count": initial_user_count,
        "raw_item_count": initial_item_count,
        "kept_user_count": len(remapped_sequences),
        "kept_item_count": len(item_mapping),
        "train_sample_count": int(len(train_target_ids)),
        "train_sequence_window_count": int(len(train_sequence_user_ids)),
        "train_supervision_position_count": int(
            np.count_nonzero(train_sequence_positive_ids)
        ),
        "valid_user_count": int(len(valid_target_ids)),
        "test_user_count": int(len(test_target_ids)),
        "filter_stats": filter_stats,
        "artifacts": {
            "train_sequences": str(train_path),
            "train_sequence_supervision": str(train_sequence_supervision_path),
            "valid_sequences": str(valid_path),
            "test_sequences": str(test_path),
            "user_mapping": str(user_mapping_path),
            "item_mapping": str(item_mapping_path),
        },
    }

    with metadata_path.open("w", encoding="utf-8") as fout:
        json.dump(metadata, fout, ensure_ascii=False, indent=2)

    print("Preprocessing finished.")
    print(f"Train samples: {len(train_target_ids)}")
    print(f"Validation users: {len(valid_target_ids)}")
    print(f"Test users: {len(test_target_ids)}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
