from __future__ import annotations

import csv
from pathlib import Path


DEFAULT_MOVIES_PATH = Path(
    r"D:\Python\Datasets\movielens_1m\raw_extracted\ml-1m\movies.dat"
)
DEFAULT_ITEM_MAPPING_PATH = Path(
    r"D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_baseline\artifacts\preprocessed\item_id_mapping.csv"
)


def load_item_mapping(item_mapping_path: Path = DEFAULT_ITEM_MAPPING_PATH) -> dict[int, int]:
    if not item_mapping_path.exists():
        raise FileNotFoundError(f"Item mapping file not found: {item_mapping_path}")

    mapping: dict[int, int] = {}
    with item_mapping_path.open("r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)
        required_columns = {"raw_movie_id", "mapped_movie_id"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                "Item mapping CSV must contain columns: raw_movie_id, mapped_movie_id"
            )

        for row in reader:
            mapping[int(row["raw_movie_id"])] = int(row["mapped_movie_id"])
    return mapping


def load_raw_movie_genres(
    movies_path: Path = DEFAULT_MOVIES_PATH,
) -> dict[int, set[str]]:
    if not movies_path.exists():
        raise FileNotFoundError(f"Movie metadata file not found: {movies_path}")

    raw_movie_genres: dict[int, set[str]] = {}
    with movies_path.open("r", encoding="latin-1") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            raw_movie_id_text, _, genre_text = line.split("::", maxsplit=2)
            genres = set(genre_text.split("|")) if genre_text else set()
            raw_movie_genres[int(raw_movie_id_text)] = genres
    return raw_movie_genres


def load_mapped_movie_genres(
    movies_path: Path = DEFAULT_MOVIES_PATH,
    item_mapping_path: Path = DEFAULT_ITEM_MAPPING_PATH,
) -> dict[int, set[str]]:
    item_mapping = load_item_mapping(item_mapping_path)
    raw_movie_genres = load_raw_movie_genres(movies_path)

    mapped_movie_genres: dict[int, set[str]] = {}
    for raw_movie_id, mapped_movie_id in item_mapping.items():
        mapped_movie_genres[mapped_movie_id] = raw_movie_genres.get(raw_movie_id, set())
    return mapped_movie_genres


def compute_recommendation_reward(
    recommended_item_id: int,
    target_item_id: int,
    mapped_movie_genres: dict[int, set[str]],
    exact_hit_reward: float = 1.0,
    genre_match_reward: float = 0.1,
    mismatch_reward: float = -0.1,
) -> float:
    if recommended_item_id == target_item_id:
        return exact_hit_reward

    recommended_genres = mapped_movie_genres.get(recommended_item_id, set())
    target_genres = mapped_movie_genres.get(target_item_id, set())
    if recommended_genres & target_genres:
        return genre_match_reward
    return mismatch_reward
