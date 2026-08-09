from __future__ import annotations

import csv
from pathlib import Path

import gymnasium as gym
import numpy as np


DEFAULT_MOVIES_PATH = Path(
    r"D:\Python\Datasets\movielens_1m\raw_extracted\ml-1m\movies.dat"
)
DEFAULT_ITEM_MAPPING_PATH = Path(
    r"D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_baseline\artifacts\preprocessed\item_id_mapping.csv"
)


def load_item_mapping(
    item_mapping_path: Path = DEFAULT_ITEM_MAPPING_PATH,
) -> dict[int, int]:
    if not item_mapping_path.exists():
        raise FileNotFoundError(f"Item mapping file not found: {item_mapping_path}")

    mapping: dict[int, int] = {}
    with item_mapping_path.open("r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            mapping[int(row["mapped_movie_id"])] = int(row["raw_movie_id"])
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
    mapped_to_raw = load_item_mapping(item_mapping_path)
    raw_movie_genres = load_raw_movie_genres(movies_path)

    mapped_movie_genres: dict[int, set[str]] = {}
    for mapped_movie_id, raw_movie_id in mapped_to_raw.items():
        mapped_movie_genres[mapped_movie_id] = raw_movie_genres.get(raw_movie_id, set())
    return mapped_movie_genres


class MicroRecSimEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(5)
        observation_low = np.zeros(11, dtype=np.float32)
        observation_high = np.array([1.0] * 10 + [10.0], dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=observation_low,
            high=observation_high,
            dtype=np.float32,
        )
        self._rng = np.random.default_rng()
        self._true_preference = np.zeros(5, dtype=np.float32)
        self._fatigue = np.zeros(5, dtype=np.float32)
        self._patience = 10.0

    def _get_observation(self) -> np.ndarray:
        return np.concatenate(
            [
                self._fatigue,
                self._true_preference,
                np.array([self._patience], dtype=np.float32),
            ]
        ).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._rng = self.np_random
        self._true_preference = self._rng.uniform(0.2, 0.9, size=5).astype(np.float32)
        self._fatigue = np.zeros(5, dtype=np.float32)
        self._patience = 10.0
        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        click_probability = float(
            np.clip(
                self._true_preference[action] * (1.0 - self._fatigue[action]),
                0.0,
                1.0,
            )
        )
        clicked = bool(self._rng.random() < click_probability)

        self._fatigue[action] = np.clip(self._fatigue[action] + 0.2, 0.0, 1.0)
        other_indices = np.arange(5) != action
        self._fatigue[other_indices] = np.clip(
            self._fatigue[other_indices] * 0.9,
            0.0,
            1.0,
        )

        if not clicked:
            self._patience = max(0.0, self._patience - 1.0)

        terminated = self._patience <= 0.0
        truncated = False

        if terminated:
            reward = -10.0
        elif clicked:
            reward = 1.0
        else:
            reward = 0.0

        observation = self._get_observation()
        info = {
            "clicked": clicked,
            "click_probability": click_probability,
            "true_preference": self._true_preference.copy(),
        }
        return observation, reward, terminated, truncated, info


class SlateRecSimEnv(MicroRecSimEnv):
    metadata = {"render_modes": []}

    def __init__(
        self,
        slate_size: int = 5,
        num_items: int = 3706,
        max_seq_len: int = 50,
        max_episode_steps: int = 10,
        max_patience: float = 10.0,
        fatigue_decay: float = 0.9,
        clicked_fatigue_increase: float = 0.25,
        movies_path: Path = DEFAULT_MOVIES_PATH,
        item_mapping_path: Path = DEFAULT_ITEM_MAPPING_PATH,
    ) -> None:
        super().__init__()
        self.slate_size = slate_size
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.max_episode_steps = max_episode_steps
        self.max_patience = max_patience
        self.fatigue_decay = fatigue_decay
        self.clicked_fatigue_increase = clicked_fatigue_increase

        mapped_movie_genres = load_mapped_movie_genres(
            movies_path=movies_path,
            item_mapping_path=item_mapping_path,
        )
        if len(mapped_movie_genres) < num_items:
            raise ValueError(
                f"Mapped movie genres only cover {len(mapped_movie_genres)} items, "
                f"but num_items={num_items} was requested."
            )

        self.genre_vocab = sorted(
            {
                genre
                for mapped_item_id in range(1, num_items + 1)
                for genre in mapped_movie_genres.get(mapped_item_id, set())
            }
        )
        self.genre_to_index = {
            genre_name: genre_index
            for genre_index, genre_name in enumerate(self.genre_vocab)
        }
        self.item_genres = {
            item_id: mapped_movie_genres.get(item_id, set())
            for item_id in range(1, num_items + 1)
        }
        self.item_type_mapping = np.zeros(num_items + 1, dtype=np.int64)
        for item_id in range(1, num_items + 1):
            genres = sorted(self.item_genres.get(item_id, set()))
            primary_genre = genres[0] if genres else self.genre_vocab[0]
            self.item_type_mapping[item_id] = self.genre_to_index[primary_genre]

        self.num_item_types = len(self.genre_vocab)
        self.action_space = gym.spaces.Box(
            low=1,
            high=num_items,
            shape=(slate_size,),
            dtype=np.int64,
        )
        self.observation_space = gym.spaces.Dict(
            {
                "input_ids": gym.spaces.Box(
                    low=0,
                    high=num_items,
                    shape=(max_seq_len,),
                    dtype=np.int64,
                ),
                "fatigue": gym.spaces.Box(
                    low=np.zeros(self.num_item_types, dtype=np.float32),
                    high=np.ones(self.num_item_types, dtype=np.float32),
                    dtype=np.float32,
                ),
                "base_preference": gym.spaces.Box(
                    low=np.zeros(self.num_item_types, dtype=np.float32),
                    high=np.ones(self.num_item_types, dtype=np.float32),
                    dtype=np.float32,
                ),
                "effective_preference": gym.spaces.Box(
                    low=np.zeros(self.num_item_types, dtype=np.float32),
                    high=np.ones(self.num_item_types, dtype=np.float32),
                    dtype=np.float32,
                ),
                "patience": gym.spaces.Box(
                    low=np.array([0.0], dtype=np.float32),
                    high=np.array([max_patience], dtype=np.float32),
                    dtype=np.float32,
                ),
                "slate_step": gym.spaces.Box(
                    low=np.array([0], dtype=np.int64),
                    high=np.array([max_episode_steps], dtype=np.int64),
                    dtype=np.int64,
                ),
            }
        )

        self._history_input_ids = np.zeros(max_seq_len, dtype=np.int64)
        self._base_preference = np.zeros(self.num_item_types, dtype=np.float32)
        self._fatigue = np.zeros(self.num_item_types, dtype=np.float32)
        self._patience = max_patience
        self._slate_step = 0
        self._current_user_id: int | None = None

    def _prepare_history(self, input_ids: np.ndarray | list[int]) -> np.ndarray:
        history = np.asarray(input_ids, dtype=np.int64).reshape(-1)
        if history.size >= self.max_seq_len:
            return history[-self.max_seq_len :].astype(np.int64)

        padded_history = np.zeros(self.max_seq_len, dtype=np.int64)
        padded_history[-history.size :] = history
        return padded_history

    def _initialize_true_preference(
        self,
        history_input_ids: np.ndarray,
        provided_preference: np.ndarray | list[float] | None = None,
    ) -> np.ndarray:
        if provided_preference is not None:
            preference = np.asarray(provided_preference, dtype=np.float32).reshape(-1)
            if preference.size != self.num_item_types:
                raise ValueError(
                    f"Provided preference vector has size {preference.size}, "
                    f"expected {self.num_item_types}."
                )
            return np.clip(preference, 0.05, 0.95).astype(np.float32)

        genre_counts = np.zeros(self.num_item_types, dtype=np.float32)
        for item_id in history_input_ids:
            if item_id <= 0:
                continue
            genre_index = int(self.item_type_mapping[int(item_id)])
            genre_counts[genre_index] += 1.0

        base_preference = np.full(self.num_item_types, 0.15, dtype=np.float32)
        max_count = float(genre_counts.max())
        if max_count > 0.0:
            base_preference += 0.7 * (genre_counts / max_count)

        noise = self._rng.uniform(-0.05, 0.05, size=self.num_item_types).astype(np.float32)
        return np.clip(base_preference + noise, 0.05, 0.95).astype(np.float32)

    def _get_effective_preference(self) -> np.ndarray:
        return np.clip(self._base_preference * (1.0 - self._fatigue), 0.0, 1.0).astype(
            np.float32
        )

    def _append_clicked_items(self, clicked_item_ids: list[int]) -> None:
        if not clicked_item_ids:
            return

        history = self._history_input_ids.tolist()
        for item_id in clicked_item_ids:
            history.append(int(item_id))
        self._history_input_ids = np.asarray(history[-self.max_seq_len :], dtype=np.int64)

    def _update_fatigue(self, clicked_item_ids: list[int]) -> None:
        self._fatigue = np.clip(self._fatigue * self.fatigue_decay, 0.0, 1.0)
        for item_id in clicked_item_ids:
            item_type = int(self.item_type_mapping[int(item_id)])
            self._fatigue[item_type] = np.clip(
                self._fatigue[item_type] + self.clicked_fatigue_increase,
                0.0,
                1.0,
            )

    def _get_observation(self) -> dict[str, np.ndarray]:
        return {
            "input_ids": self._history_input_ids.astype(np.int64),
            "fatigue": self._fatigue.astype(np.float32),
            "base_preference": self._base_preference.astype(np.float32),
            "effective_preference": self._get_effective_preference(),
            "patience": np.array([self._patience], dtype=np.float32),
            "slate_step": np.array([self._slate_step], dtype=np.int64),
        }

    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        self._rng = self.np_random
        options = options or {}

        input_ids = options.get("input_ids")
        if input_ids is None:
            input_ids = np.zeros(self.max_seq_len, dtype=np.int64)

        self._history_input_ids = self._prepare_history(input_ids)
        self._base_preference = self._initialize_true_preference(
            self._history_input_ids,
            provided_preference=options.get("true_preference"),
        )
        self._fatigue = np.zeros(self.num_item_types, dtype=np.float32)
        self._patience = self.max_patience
        self._slate_step = 0
        self._current_user_id = options.get("user_id")
        observation = self._get_observation()
        info = {
            "user_id": self._current_user_id,
            "genre_vocab": self.genre_vocab,
        }
        return observation, info

    def step(self, action_slate):
        action_slate = np.asarray(action_slate, dtype=np.int64).reshape(-1)
        if action_slate.shape != (self.slate_size,):
            raise ValueError(
                f"Slate action must have shape ({self.slate_size},), got {action_slate.shape}."
            )
        if np.any(action_slate < 1) or np.any(action_slate > self.num_items):
            raise ValueError(
                f"Slate action contains item ids outside [1, {self.num_items}]."
            )

        effective_preference = self._get_effective_preference()
        clicked_item_ids: list[int] = []
        clicked_flags: list[bool] = []
        exposed_flags: list[bool] = []
        click_probabilities: list[float] = []
        conditional_click_probabilities: list[float] = []
        position_biases: list[float] = []
        diversity_penalties: list[float] = []
        item_types: list[int] = []
        seen_item_types: set[int] = set()

        for slate_position, item_id in enumerate(action_slate, start=1):
            item_type = int(self.item_type_mapping[int(item_id)])
            position_bias = float(1.0 / np.log2(slate_position + 1.0))
            diversity_penalty = 0.5 if item_type in seen_item_types else 1.0
            conditional_click_probability = float(
                np.clip(
                    effective_preference[item_type] * diversity_penalty,
                    0.0,
                    1.0,
                )
            )
            exposed = bool(self._rng.random() < position_bias)
            clicked = exposed and bool(
                self._rng.random() < conditional_click_probability
            )
            click_probability = float(position_bias * conditional_click_probability)

            item_types.append(item_type)
            position_biases.append(position_bias)
            diversity_penalties.append(diversity_penalty)
            conditional_click_probabilities.append(conditional_click_probability)
            click_probabilities.append(click_probability)
            exposed_flags.append(exposed)
            clicked_flags.append(clicked)

            if clicked:
                clicked_item_ids.append(int(item_id))

            seen_item_types.add(item_type)

        total_reward = float(sum(clicked_flags))
        self._append_clicked_items(clicked_item_ids)
        self._update_fatigue(clicked_item_ids)
        self._slate_step += 1
        if total_reward <= 0.0:
            self._patience = max(0.0, self._patience - 1.0)

        terminated = bool(
            self._patience <= 0.0 or self._slate_step >= self.max_episode_steps
        )
        truncated = False
        observation = self._get_observation()
        info = {
            "user_id": self._current_user_id,
            "action_slate": action_slate.astype(np.int64),
            "item_types": np.asarray(item_types, dtype=np.int64),
            "clicked_flags": np.asarray(clicked_flags, dtype=np.bool_),
            "exposed_flags": np.asarray(exposed_flags, dtype=np.bool_),
            "clicked_item_ids": np.asarray(clicked_item_ids, dtype=np.int64),
            "click_probabilities": np.asarray(click_probabilities, dtype=np.float32),
            "conditional_click_probabilities": np.asarray(
                conditional_click_probabilities,
                dtype=np.float32,
            ),
            "position_biases": np.asarray(position_biases, dtype=np.float32),
            "diversity_penalties": np.asarray(diversity_penalties, dtype=np.float32),
            "base_preference": self._base_preference.copy(),
            "effective_preference": effective_preference.copy(),
            "fatigue": self._fatigue.copy(),
            "total_clicks": int(total_reward),
        }
        return observation, total_reward, terminated, truncated, info


if __name__ == "__main__":
    env = MicroRecSimEnv()
    obs, info = env.reset(seed=42)

    total_reward = 0.0
    print("Initial observation:", obs)
    print("Observed user preference:", obs[5:10])

    for i in range(100):
        random_action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(random_action)
        total_reward += reward
        print(
            f"Step {i:02d} | Action {random_action} | "
            f"Reward {reward:>5} | Clicked {info['clicked']} | "
            f"ClickProb {info['click_probability']:.3f} | "
            f"Patience {obs[-1]:.1f} | Fatigue {obs[:5]} | Preference {obs[5:10]}"
        )

        if terminated or truncated:
            print(f"User left the app. Total reward: {total_reward}")
            break
