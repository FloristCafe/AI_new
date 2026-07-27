from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ml_1m_genre_utils import (
    DEFAULT_ITEM_MAPPING_PATH,
    DEFAULT_MOVIES_PATH,
    compute_recommendation_reward,
    load_mapped_movie_genres,
)


DEFAULT_INTERACTIONS_PATH = Path(
    r"D:\Python\Datasets\movielens_1m\processed\interactions.csv"
)
DEFAULT_USER_MAPPING_PATH = Path(
    r"D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_baseline\artifacts\preprocessed\user_id_mapping.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    r"D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_dqn_offline_rl\artifacts\offline_buffer"
)

STATE_DTYPE = np.uint16
USER_DTYPE = np.uint16
STEP_DTYPE = np.uint16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build offline RL replay buffers from MovieLens-1M user sequences."
    )
    parser.add_argument(
        "--interactions-path",
        type=str,
        default=str(DEFAULT_INTERACTIONS_PATH),
        help="Path to cleaned interaction CSV with user_id, movie_id, timestamp.",
    )
    parser.add_argument(
        "--user-mapping-path",
        type=str,
        default=str(DEFAULT_USER_MAPPING_PATH),
        help="Path to upstream SASRec user mapping CSV.",
    )
    parser.add_argument(
        "--item-mapping-path",
        type=str,
        default=str(DEFAULT_ITEM_MAPPING_PATH),
        help="Path to upstream SASRec item mapping CSV.",
    )
    parser.add_argument(
        "--movies-path",
        type=str,
        default=str(DEFAULT_MOVIES_PATH),
        help="Path to MovieLens-1M movies.dat metadata file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for replay buffer artifacts.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=50,
        help="Maximum sequence length for state and next_state tensors.",
    )
    parser.add_argument(
        "--sample-user-limit",
        type=int,
        default=0,
        help="Optional user cap for smoke tests. Set 0 to keep all users.",
    )
    return parser.parse_args()


def load_interactions(interactions_path: Path) -> list[tuple[int, int, int]]:
    if not interactions_path.exists():
        raise FileNotFoundError(f"Interactions file not found: {interactions_path}")

    rows: list[tuple[int, int, int]] = []
    with interactions_path.open("r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)
        required_columns = {"user_id", "movie_id", "timestamp"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                "Interactions CSV must contain columns: user_id, movie_id, timestamp"
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
        raise ValueError(f"No interactions found in: {interactions_path}")
    return rows


def load_mapping_csv(
    mapping_path: Path,
    raw_column: str,
    mapped_column: str,
) -> dict[int, int]:
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")

    mapping: dict[int, int] = {}
    with mapping_path.open("r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)
        required_columns = {raw_column, mapped_column}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                f"Mapping CSV must contain columns: {raw_column}, {mapped_column}"
            )

        for row in reader:
            mapping[int(row[raw_column])] = int(row[mapped_column])
    return mapping


def build_sorted_raw_histories(
    rows: list[tuple[int, int, int]]
) -> dict[int, list[tuple[int, int]]]:
    user_histories: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for raw_user_id, raw_movie_id, timestamp in rows:
        user_histories[raw_user_id].append((timestamp, raw_movie_id))

    sorted_histories: dict[int, list[tuple[int, int]]] = {}
    for raw_user_id, events in user_histories.items():
        events.sort(key=lambda value: (value[0], value[1]))
        sorted_histories[raw_user_id] = events
    return sorted_histories


def remap_histories(
    sorted_raw_histories: dict[int, list[tuple[int, int]]],
    user_mapping: dict[int, int],
    item_mapping: dict[int, int],
    sample_user_limit: int,
) -> dict[int, list[int]]:
    remapped_histories: dict[int, list[int]] = {}
    raw_user_ids = sorted(user_mapping.keys())
    if sample_user_limit > 0:
        raw_user_ids = raw_user_ids[:sample_user_limit]

    for raw_user_id in raw_user_ids:
        events = sorted_raw_histories.get(raw_user_id)
        if not events:
            continue

        mapped_items = [
            item_mapping[raw_movie_id]
            for _, raw_movie_id in events
            if raw_movie_id in item_mapping
        ]
        if len(mapped_items) >= 2:
            remapped_histories[user_mapping[raw_user_id]] = mapped_items
    return remapped_histories


def pad_sequence(sequence: list[int], max_seq_len: int) -> np.ndarray:
    padded = np.zeros(max_seq_len, dtype=STATE_DTYPE)
    if not sequence:
        return padded

    truncated = sequence[-max_seq_len:]
    padded[-len(truncated) :] = np.asarray(truncated, dtype=STATE_DTYPE)
    return padded


def build_train_like_transitions(
    user_id: int,
    items: list[int],
    max_seq_len: int,
    mapped_movie_genres: dict[int, set[str]],
    split_name: str,
) -> list[dict[str, object]]:
    transitions: list[dict[str, object]] = []
    if len(items) < 2:
        return transitions

    for next_index in range(1, len(items)):
        history = items[:next_index]
        action = items[next_index]
        next_history = items[: next_index + 1]
        reward = compute_recommendation_reward(
            recommended_item_id=action,
            target_item_id=action,
            mapped_movie_genres=mapped_movie_genres,
        )
        transitions.append(
            {
                "state": pad_sequence(history, max_seq_len),
                "action": action,
                "reward": reward,
                "next_state": pad_sequence(next_history, max_seq_len),
                "done": next_index == len(items) - 1,
                "user_id": user_id,
                "step_index": next_index,
                "split": split_name,
            }
        )
    return transitions


def build_eval_transition(
    user_id: int,
    history: list[int],
    action: int,
    next_history: list[int],
    max_seq_len: int,
    mapped_movie_genres: dict[int, set[str]],
    step_index: int,
    split_name: str,
) -> dict[str, object]:
    reward = compute_recommendation_reward(
        recommended_item_id=action,
        target_item_id=action,
        mapped_movie_genres=mapped_movie_genres,
    )
    return {
        "state": pad_sequence(history, max_seq_len),
        "action": action,
        "reward": reward,
        "next_state": pad_sequence(next_history, max_seq_len),
        "done": True,
        "user_id": user_id,
        "step_index": step_index,
        "split": split_name,
    }


def build_split_transitions(
    user_sequences: dict[int, list[int]],
    max_seq_len: int,
    mapped_movie_genres: dict[int, set[str]],
) -> dict[str, list[dict[str, object]]]:
    split_transitions = {
        "train": [],
        "valid": [],
        "test": [],
        "all": [],
    }

    for user_id, items in user_sequences.items():
        if len(items) < 3:
            continue

        train_items = items[:-2]
        if len(train_items) >= 2:
            split_transitions["train"].extend(
                build_train_like_transitions(
                    user_id=user_id,
                    items=train_items,
                    max_seq_len=max_seq_len,
                    mapped_movie_genres=mapped_movie_genres,
                    split_name="train",
                )
            )

        split_transitions["valid"].append(
            build_eval_transition(
                user_id=user_id,
                history=items[:-2],
                action=items[-2],
                next_history=items[:-1],
                max_seq_len=max_seq_len,
                mapped_movie_genres=mapped_movie_genres,
                step_index=len(items) - 2,
                split_name="valid",
            )
        )
        split_transitions["test"].append(
            build_eval_transition(
                user_id=user_id,
                history=items[:-1],
                action=items[-1],
                next_history=items[:],
                max_seq_len=max_seq_len,
                mapped_movie_genres=mapped_movie_genres,
                step_index=len(items) - 1,
                split_name="test",
            )
        )
        split_transitions["all"].extend(
            build_train_like_transitions(
                user_id=user_id,
                items=items,
                max_seq_len=max_seq_len,
                mapped_movie_genres=mapped_movie_genres,
                split_name="all",
            )
        )

    return split_transitions


def stack_transition_payload(
    transitions: list[dict[str, object]]
) -> dict[str, np.ndarray]:
    if not transitions:
        raise ValueError("No transitions were created for this split.")

    return {
        "states": np.stack([row["state"] for row in transitions]).astype(STATE_DTYPE),
        "actions": np.asarray([row["action"] for row in transitions], dtype=STATE_DTYPE),
        "rewards": np.asarray([row["reward"] for row in transitions], dtype=np.float32),
        "next_states": np.stack([row["next_state"] for row in transitions]).astype(STATE_DTYPE),
        "dones": np.asarray([row["done"] for row in transitions], dtype=np.bool_),
        "user_ids": np.asarray([row["user_id"] for row in transitions], dtype=USER_DTYPE),
        "step_indices": np.asarray(
            [row["step_index"] for row in transitions],
            dtype=STEP_DTYPE,
        ),
    }


def save_replay_buffer(path: Path, payload: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **payload)


def build_split_metadata(
    split_name: str,
    payload: dict[str, np.ndarray],
) -> dict[str, object]:
    rewards = payload["rewards"]
    dones = payload["dones"]
    return {
        "split": split_name,
        "transition_count": int(rewards.shape[0]),
        "state_shape": list(payload["states"].shape),
        "next_state_shape": list(payload["next_states"].shape),
        "reward_min": float(rewards.min()),
        "reward_max": float(rewards.max()),
        "reward_mean": float(rewards.mean()),
        "done_count": int(dones.sum()),
        "user_count": int(np.unique(payload["user_ids"]).shape[0]),
    }


def main() -> None:
    args = parse_args()

    interactions_path = Path(args.interactions_path)
    user_mapping_path = Path(args.user_mapping_path)
    item_mapping_path = Path(args.item_mapping_path)
    movies_path = Path(args.movies_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_interactions(interactions_path)
    user_mapping = load_mapping_csv(
        user_mapping_path,
        raw_column="raw_user_id",
        mapped_column="mapped_user_id",
    )
    item_mapping = load_mapping_csv(
        item_mapping_path,
        raw_column="raw_movie_id",
        mapped_column="mapped_movie_id",
    )
    mapped_movie_genres = load_mapped_movie_genres(
        movies_path=movies_path,
        item_mapping_path=item_mapping_path,
    )

    sorted_raw_histories = build_sorted_raw_histories(rows)
    user_sequences = remap_histories(
        sorted_raw_histories=sorted_raw_histories,
        user_mapping=user_mapping,
        item_mapping=item_mapping,
        sample_user_limit=args.sample_user_limit,
    )
    split_transitions = build_split_transitions(
        user_sequences=user_sequences,
        max_seq_len=args.max_seq_len,
        mapped_movie_genres=mapped_movie_genres,
    )

    summary = {
        "project": "movielens_1m_sasrec_dqn_offline_rl",
        "input_paths": {
            "interactions": str(interactions_path),
            "user_mapping": str(user_mapping_path),
            "item_mapping": str(item_mapping_path),
            "movies": str(movies_path),
        },
        "output_dir": str(output_dir),
        "max_seq_len": args.max_seq_len,
        "sample_user_limit": args.sample_user_limit,
        "kept_user_count": len(user_sequences),
        "kept_item_count": len(item_mapping),
        "reward_definition": {
            "exact_hit_reward": 1.0,
            "genre_match_reward": 0.1,
            "mismatch_reward": -0.1,
        },
        "logged_action_note": (
            "Offline buffer stores logged next-click actions. "
            "Under this logged-action setup, the immediate reward values in the "
            "buffer are all exact-hit rewards (+1.0). Genre-based rewards remain "
            "relevant for later counterfactual evaluation when the agent chooses "
            "actions other than the logged next click."
        ),
        "storage_dtypes": {
            "states": str(STATE_DTYPE),
            "actions": str(STATE_DTYPE),
            "next_states": str(STATE_DTYPE),
            "user_ids": str(USER_DTYPE),
            "step_indices": str(STEP_DTYPE),
            "rewards": "float32",
            "dones": "bool",
        },
        "splits": {},
    }

    for split_name, transitions in split_transitions.items():
        payload = stack_transition_payload(transitions)
        output_path = output_dir / f"{split_name}_replay_buffer.npz"
        save_replay_buffer(output_path, payload)

        split_metadata = build_split_metadata(split_name, payload)
        split_metadata["artifact_path"] = str(output_path)
        summary["splits"][split_name] = split_metadata

        print(
            f"Saved {split_name} replay buffer: {output_path} "
            f"(transitions={split_metadata['transition_count']})"
        )

    metadata_path = output_dir / "offline_buffer_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as fout:
        json.dump(summary, fout, ensure_ascii=False, indent=2)

    print("Offline replay buffer construction finished.")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
