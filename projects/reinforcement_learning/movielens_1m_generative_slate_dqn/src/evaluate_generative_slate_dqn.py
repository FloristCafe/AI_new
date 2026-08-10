from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

CURRENT_PROJECT_DIR = Path(
    r"D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_generative_slate_dqn"
)
DEFAULT_TRAIN_SEQUENCE_PATH = Path(
    r"D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_baseline\artifacts\preprocessed\train_sequence_supervision.npz"
)
DEFAULT_VALID_SEQUENCE_PATH = Path(
    r"D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_baseline\artifacts\preprocessed\valid_sequences.npz"
)
DEFAULT_TEST_SEQUENCE_PATH = Path(
    r"D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_baseline\artifacts\preprocessed\test_sequences.npz"
)
DEFAULT_METADATA_PATH = Path(
    r"D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_baseline\artifacts\preprocessed\metadata.json"
)
DEFAULT_CHECKPOINT_PATH = Path(
    r"D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_generative_slate_dqn\artifacts\checkpoints\slate_dqn_best.pt"
)
DEFAULT_OUTPUT_DIR = CURRENT_PROJECT_DIR / "artifacts" / "predictions"
RECSIM_ENV_DIR = Path(
    r"D:\Python\Artificial Intelligence\projects\recommendation\recommender_mdp_gymnasium"
)

if str(RECSIM_ENV_DIR) not in sys.path:
    sys.path.append(str(RECSIM_ENV_DIR))

from micro_recsim_env import SlateRecSimEnv
from slate_dqn_model import DEFAULT_SASREC_CHECKPOINT_PATH, SlateDQN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the MovieLens-1M Generative Slate-DQN with batched Monte Carlo rollouts."
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=str(DEFAULT_CHECKPOINT_PATH),
        help="Path to the trained Slate-DQN checkpoint.",
    )
    parser.add_argument(
        "--sasrec-checkpoint-path",
        type=str,
        default=str(DEFAULT_SASREC_CHECKPOINT_PATH),
        help="Path to the pretrained SASRec checkpoint used to initialize the user encoder.",
    )
    parser.add_argument(
        "--train-sequence-path",
        type=str,
        default=str(DEFAULT_TRAIN_SEQUENCE_PATH),
        help="Path to the training sequence pool used to build popularity baselines.",
    )
    parser.add_argument(
        "--eval-sequence-path",
        type=str,
        default=str(DEFAULT_VALID_SEQUENCE_PATH),
        help="Path to the validation or test sequence pool.",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default=str(DEFAULT_METADATA_PATH),
        help="Path to MovieLens preprocessing metadata.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory used to save the final evaluation report.",
    )
    parser.add_argument(
        "--split-name",
        type=str,
        default="valid",
        choices=["valid", "test", "custom"],
        help="Logical name of the evaluated split.",
    )
    parser.add_argument(
        "--policy-names",
        type=str,
        default="slate_dqn,sasrec_topk,popularity_topk,random_unique",
        help="Comma-separated policies to evaluate.",
    )
    parser.add_argument(
        "--slate-size",
        type=int,
        default=5,
        help="Number of items per slate.",
    )
    parser.add_argument(
        "--candidate-pool-size",
        type=int,
        default=50,
        help=(
            "Size of the SASRec candidate pool used to truncate the Slate-DQN action space. "
            "Set 0 to disable candidate truncation."
        ),
    )
    parser.add_argument(
        "--blend-lambda",
        type=float,
        default=0.0,
        help=(
            "Residual logit blending weight used only for slate_dqn inference: "
            "Q_final = Q_slate_dqn + blend_lambda * logits_sasrec."
        ),
    )
    parser.add_argument(
        "--mc-rollouts",
        type=int,
        default=0,
        help=(
            "Number of Monte Carlo rollouts per initial user state. "
            "Set 0 to use split-aware defaults: valid=1, test=3."
        ),
    )
    parser.add_argument(
        "--rollout-batch-size",
        type=int,
        default=128,
        help="Number of repeated episodes processed together during evaluation.",
    )
    parser.add_argument(
        "--max-eval-users",
        type=int,
        default=1000,
        help=(
            "Maximum number of unique evaluation users/states. "
            "Set 0 to use the full split."
        ),
    )
    parser.add_argument(
        "--max-states",
        type=int,
        default=-1,
        help=(
            "Deprecated alias of --max-eval-users. "
            "Set >= 0 to override --max-eval-users."
        ),
    )
    parser.add_argument(
        "--max-slates-per-episode",
        type=int,
        default=0,
        help=(
            "Maximum number of slates generated in a single episode. "
            "Set 0 to use split-aware defaults: valid=5, test=10."
        ),
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Evaluate only slate_dqn and skip sasrec/popularity/random baselines.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Torch device selection.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_mc_rollouts(split_name: str, mc_rollouts: int) -> int:
    if mc_rollouts > 0:
        return mc_rollouts
    if split_name == "test":
        return 3
    return 1


def resolve_max_eval_users(max_eval_users: int, max_states: int) -> int:
    if max_states >= 0:
        return max_states
    return max_eval_users


def resolve_max_slates_per_episode(split_name: str, max_slates_per_episode: int) -> int:
    if max_slates_per_episode > 0:
        return max_slates_per_episode
    if split_name == "test":
        return 10
    return 5


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_metadata(metadata_path: Path) -> dict:
    with metadata_path.open("r", encoding="utf-8") as fin:
        return json.load(fin)


def load_sequence_pool(npz_path: Path) -> dict[str, np.ndarray]:
    data = np.load(npz_path)
    return {key: data[key] for key in data.files}


def build_model(
    metadata: dict,
    checkpoint_path: Path,
    sasrec_checkpoint_path: Path,
    slate_size: int,
    device: torch.device,
) -> SlateDQN:
    model = SlateDQN(
        num_items=int(metadata["kept_item_count"]),
        max_seq_len=int(metadata["max_seq_len"]),
        slate_size=slate_size,
        embedding_dim=128,
        num_heads=2,
        num_blocks=3,
        dropout=0.2,
        context_dropout=0.1,
    ).to(device)
    model.load_sasrec_encoder_weights(sasrec_checkpoint_path, map_location=device)
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.freeze_user_encoder()
    model.eval()
    return model


def build_env(metadata: dict, slate_size: int, max_slates_per_episode: int) -> SlateRecSimEnv:
    return SlateRecSimEnv(
        slate_size=slate_size,
        num_items=int(metadata["kept_item_count"]),
        max_seq_len=int(metadata["max_seq_len"]),
        max_episode_steps=max_slates_per_episode,
    )


def clone_env(env: SlateRecSimEnv) -> SlateRecSimEnv:
    return SlateRecSimEnv(
        slate_size=env.slate_size,
        num_items=env.num_items,
        max_seq_len=env.max_seq_len,
        max_episode_steps=env.max_episode_steps,
        max_patience=env.max_patience,
        fatigue_decay=env.fatigue_decay,
        clicked_fatigue_increase=env.clicked_fatigue_increase,
    )


def normalize_policy_names(policy_names: str) -> list[str]:
    normalized = [policy_name.strip() for policy_name in policy_names.split(",")]
    normalized = [policy_name for policy_name in normalized if policy_name]
    if not normalized:
        raise ValueError("At least one policy name must be provided.")
    return normalized


def compute_item_popularity(
    train_sequence_pool: dict[str, np.ndarray],
    num_items: int,
) -> np.ndarray:
    item_counts = np.zeros(num_items + 1, dtype=np.int64)

    if "input_ids" in train_sequence_pool:
        input_ids = train_sequence_pool["input_ids"].astype(np.int64).reshape(-1)
        valid_input_ids = input_ids[input_ids > 0]
        item_counts += np.bincount(valid_input_ids, minlength=num_items + 1)

    if "target_ids" in train_sequence_pool:
        target_ids = train_sequence_pool["target_ids"].astype(np.int64).reshape(-1)
        valid_target_ids = target_ids[target_ids > 0]
        item_counts += np.bincount(valid_target_ids, minlength=num_items + 1)

    if "positive_ids" in train_sequence_pool:
        positive_ids = train_sequence_pool["positive_ids"].astype(np.int64).reshape(-1)
        valid_positive_ids = positive_ids[positive_ids > 0]
        item_counts += np.bincount(valid_positive_ids, minlength=num_items + 1)

    ranked_items = np.argsort(-item_counts[1:]) + 1
    return ranked_items.astype(np.int64)


def compute_sasrec_item_scores_batch(
    model: SlateDQN,
    state_input_ids_batch: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    state_tensor = torch.from_numpy(state_input_ids_batch).long().to(device)
    user_state = model.encode_user_state(state_tensor)
    item_weights = model.user_encoder.item_embedding.weight[1:]
    item_scores = user_state @ item_weights.transpose(0, 1)
    history_rows, history_cols = torch.nonzero(state_tensor > 0, as_tuple=True)
    if history_rows.numel() > 0:
        item_scores[history_rows, state_tensor[history_rows, history_cols] - 1] = float("-inf")
    return item_scores


def generate_random_unique_slates(
    batch_size: int,
    slate_size: int,
    num_items: int,
    policy_rng: np.random.Generator,
) -> np.ndarray:
    slates = np.zeros((batch_size, slate_size), dtype=np.int64)
    for row_index in range(batch_size):
        slates[row_index] = policy_rng.choice(
            np.arange(1, num_items + 1, dtype=np.int64),
            size=slate_size,
            replace=False,
        )
    return slates


def generate_popularity_slates(
    batch_size: int,
    slate_size: int,
    popularity_ranking: np.ndarray,
) -> np.ndarray:
    top_slate = popularity_ranking[:slate_size].astype(np.int64)
    return np.repeat(top_slate.reshape(1, -1), batch_size, axis=0)


def generate_sasrec_topk_slates(
    model: SlateDQN,
    state_input_ids_batch: np.ndarray,
    slate_size: int,
    device: torch.device,
) -> np.ndarray:
    with torch.no_grad():
        item_scores = compute_sasrec_item_scores_batch(
            model=model,
            state_input_ids_batch=state_input_ids_batch,
            device=device,
        )
        topk_indices = torch.topk(item_scores, k=slate_size, dim=1).indices + 1
    return topk_indices.cpu().numpy().astype(np.int64)


def build_candidate_pool_batch(
    model: SlateDQN,
    state_input_ids_batch: np.ndarray,
    candidate_pool_size: int,
    device: torch.device,
) -> np.ndarray | None:
    if candidate_pool_size <= 0:
        return None
    if candidate_pool_size < model.slate_size:
        raise ValueError(
            f"candidate_pool_size={candidate_pool_size} must be >= slate_size={model.slate_size}."
        )

    with torch.no_grad():
        item_scores = compute_sasrec_item_scores_batch(
            model=model,
            state_input_ids_batch=state_input_ids_batch,
            device=device,
        )
        topk_size = min(candidate_pool_size, model.num_items)
        topk_indices = torch.topk(item_scores, k=topk_size, dim=1).indices + 1
    return topk_indices.cpu().numpy().astype(np.int64)


def build_sasrec_logits_batch(
    model: SlateDQN,
    state_input_ids_batch: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    with torch.no_grad():
        item_scores = compute_sasrec_item_scores_batch(
            model=model,
            state_input_ids_batch=state_input_ids_batch,
            device=device,
        )
    return item_scores.cpu().numpy().astype(np.float32)


def generate_slate_dqn_slates(
    model: SlateDQN,
    state_input_ids_batch: np.ndarray,
    slate_size: int,
    device: torch.device,
    candidate_pool_batch: np.ndarray | None = None,
    sasrec_logits_batch: np.ndarray | None = None,
    blend_lambda: float = 0.0,
) -> np.ndarray:
    batch_size = state_input_ids_batch.shape[0]
    state_tensor = torch.from_numpy(state_input_ids_batch).long().to(device)
    prefix_tensor = torch.zeros((batch_size, slate_size), dtype=torch.long, device=device)
    blended_logits_tensor = None
    if sasrec_logits_batch is not None and blend_lambda != 0.0:
        blended_logits_tensor = torch.from_numpy(sasrec_logits_batch).float().to(device)

    with torch.no_grad():
        for position_index in range(slate_size):
            q_values = model.get_q_values(state_tensor, prefix_tensor)
            masked_q_values = q_values.clone()
            if candidate_pool_batch is not None:
                candidate_indices = (
                    torch.from_numpy(candidate_pool_batch)
                    .long()
                    .to(device)
                    - 1
                )
                candidate_mask = torch.zeros_like(masked_q_values, dtype=torch.bool)
                candidate_mask.scatter_(1, candidate_indices, True)
                masked_q_values = masked_q_values.masked_fill(~candidate_mask, float("-inf"))
            for previous_position in range(position_index):
                selected_actions = prefix_tensor[:, previous_position]
                valid_rows = selected_actions.gt(0)
                if valid_rows.any():
                    row_indices = torch.nonzero(valid_rows, as_tuple=False).squeeze(-1)
                    column_indices = selected_actions[valid_rows] - 1
                    masked_q_values[row_indices, column_indices] = float("-inf")

            if blended_logits_tensor is not None:
                blended_q_values = masked_q_values + blend_lambda * blended_logits_tensor
            else:
                blended_q_values = masked_q_values

            next_actions = blended_q_values.argmax(dim=1) + 1
            prefix_tensor[:, position_index] = next_actions

    return prefix_tensor.cpu().numpy().astype(np.int64)


def generate_policy_slates(
    policy_name: str,
    model: SlateDQN | None,
    state_input_ids_batch: np.ndarray,
    slate_size: int,
    num_items: int,
    device: torch.device,
    policy_rng: np.random.Generator,
    popularity_ranking: np.ndarray | None = None,
    candidate_pool_batch: np.ndarray | None = None,
    sasrec_logits_batch: np.ndarray | None = None,
    blend_lambda: float = 0.0,
) -> np.ndarray:
    batch_size = state_input_ids_batch.shape[0]
    if policy_name == "random_unique":
        return generate_random_unique_slates(
            batch_size=batch_size,
            slate_size=slate_size,
            num_items=num_items,
            policy_rng=policy_rng,
        )
    if policy_name == "popularity_topk":
        if popularity_ranking is None:
            raise ValueError("Popularity ranking is required for popularity_topk evaluation.")
        return generate_popularity_slates(
            batch_size=batch_size,
            slate_size=slate_size,
            popularity_ranking=popularity_ranking,
        )
    if policy_name == "sasrec_topk":
        if model is None:
            raise ValueError("A model is required for sasrec_topk evaluation.")
        return generate_sasrec_topk_slates(
            model=model,
            state_input_ids_batch=state_input_ids_batch,
            slate_size=slate_size,
            device=device,
        )
    if policy_name == "slate_dqn":
        if model is None:
            raise ValueError("A model is required for slate_dqn evaluation.")
        return generate_slate_dqn_slates(
            model=model,
            state_input_ids_batch=state_input_ids_batch,
            slate_size=slate_size,
            device=device,
            candidate_pool_batch=candidate_pool_batch,
            sasrec_logits_batch=sasrec_logits_batch,
            blend_lambda=blend_lambda,
        )
    raise ValueError(f"Unsupported policy name: {policy_name}")


def compute_genre_entropy(item_types: np.ndarray) -> float:
    _, counts = np.unique(item_types, return_counts=True)
    probabilities = counts.astype(np.float64) / max(item_types.size, 1)
    return float(-np.sum(probabilities * np.log2(probabilities + 1e-12)))


def compute_intra_list_diversity(
    slate_item_ids: np.ndarray,
    item_genres: dict[int, set[str]],
) -> float:
    if slate_item_ids.size <= 1:
        return 0.0

    pairwise_dissimilarities: list[float] = []
    for left_index in range(slate_item_ids.size):
        for right_index in range(left_index + 1, slate_item_ids.size):
            left_genres = item_genres.get(int(slate_item_ids[left_index]), set())
            right_genres = item_genres.get(int(slate_item_ids[right_index]), set())
            union_genres = left_genres | right_genres
            if not union_genres:
                pairwise_dissimilarities.append(0.0)
                continue
            overlap = len(left_genres & right_genres)
            pairwise_dissimilarities.append(1.0 - overlap / len(union_genres))
    return float(np.mean(pairwise_dissimilarities)) if pairwise_dissimilarities else 0.0


class SlateEvaluationAccumulator:
    def __init__(
        self,
        slate_size: int,
        max_slates_per_episode: int,
        num_items: int,
        num_item_types: int,
        item_genres: dict[int, set[str]],
    ) -> None:
        self.slate_size = slate_size
        self.max_slates_per_episode = max_slates_per_episode
        self.num_items = num_items
        self.num_item_types = num_item_types
        self.item_genres = item_genres

        self.total_episodes = 0
        self.total_slates = 0
        self.total_clicks = 0.0
        self.total_slate_reward = 0.0
        self.total_episode_return = 0.0
        self.total_episode_length = 0.0
        self.slate_success_count = 0
        self.multi_click_count = 0

        self.single_slate_count = 0
        self.single_slate_reward_sum = 0.0
        self.single_slate_success_count = 0
        self.single_slate_multi_click_count = 0

        self.reward_decay_sum = np.zeros(max_slates_per_episode, dtype=np.float64)
        self.reward_decay_count = np.zeros(max_slates_per_episode, dtype=np.int64)

        self.position_examine_count = np.zeros(slate_size, dtype=np.int64)
        self.position_click_count = np.zeros(slate_size, dtype=np.int64)

        self.first_click_position_sum = 0.0
        self.first_click_defined_count = 0
        self.position_weighted_gain_sum = 0.0

        self.total_duplicate_items = 0
        self.total_same_genre_repeats = 0
        self.unique_genre_count_sum = 0.0
        self.genre_entropy_sum = 0.0
        self.intra_list_diversity_sum = 0.0

        self.recommended_item_ids: set[int] = set()
        self.recommended_genres: set[str] = set()

        self.target_hit_count = 0
        self.target_ndcg_sum = 0.0
        self.target_mrr_sum = 0.0
        self.genre_hit_count = 0

    def update_episode(self, episode_return: float, episode_length: int) -> None:
        self.total_episodes += 1
        self.total_episode_return += float(episode_return)
        self.total_episode_length += float(episode_length)

    def update_slate(
        self,
        slate_item_ids: np.ndarray,
        item_types: np.ndarray,
        clicked_flags: np.ndarray,
        exposed_flags: np.ndarray,
        reward: float,
        episode_step: int,
        target_id: int | None,
        is_first_slate: bool,
    ) -> None:
        self.total_slates += 1
        self.total_slate_reward += float(reward)
        self.total_clicks += float(np.sum(clicked_flags))
        self.reward_decay_sum[episode_step] += float(reward)
        self.reward_decay_count[episode_step] += 1

        if reward >= 1.0:
            self.slate_success_count += 1
        if reward >= 2.0:
            self.multi_click_count += 1

        if is_first_slate:
            self.single_slate_count += 1
            self.single_slate_reward_sum += float(reward)
            if reward >= 1.0:
                self.single_slate_success_count += 1
            if reward >= 2.0:
                self.single_slate_multi_click_count += 1

        self.position_examine_count += exposed_flags.astype(np.int64)
        self.position_click_count += clicked_flags.astype(np.int64)

        clicked_positions = np.flatnonzero(clicked_flags)
        if clicked_positions.size > 0:
            self.first_click_position_sum += float(clicked_positions[0] + 1)
            self.first_click_defined_count += 1

        weights = np.asarray(
            [1.0 / math.log2(position_index + 2.0) for position_index in range(self.slate_size)],
            dtype=np.float64,
        )
        self.position_weighted_gain_sum += float(np.sum(clicked_flags.astype(np.float64) * weights))

        duplicate_count = int(slate_item_ids.size - np.unique(slate_item_ids).size)
        same_genre_repeat_count = int(item_types.size - np.unique(item_types).size)
        self.total_duplicate_items += duplicate_count
        self.total_same_genre_repeats += same_genre_repeat_count
        self.unique_genre_count_sum += float(np.unique(item_types).size)
        self.genre_entropy_sum += compute_genre_entropy(item_types)
        self.intra_list_diversity_sum += compute_intra_list_diversity(
            slate_item_ids=slate_item_ids,
            item_genres=self.item_genres,
        )

        for item_id in slate_item_ids:
            self.recommended_item_ids.add(int(item_id))
            self.recommended_genres.update(self.item_genres.get(int(item_id), set()))

        if is_first_slate and target_id is not None and target_id > 0:
            target_positions = np.flatnonzero(slate_item_ids == int(target_id))
            if target_positions.size > 0:
                rank = int(target_positions[0]) + 1
                self.target_hit_count += 1
                self.target_ndcg_sum += 1.0 / math.log2(rank + 1.0)
                self.target_mrr_sum += 1.0 / rank

            target_genres = self.item_genres.get(int(target_id), set())
            if target_genres:
                slate_genres: set[str] = set()
                for item_id in slate_item_ids:
                    slate_genres.update(self.item_genres.get(int(item_id), set()))
                if target_genres & slate_genres:
                    self.genre_hit_count += 1

    def finalize(self) -> dict[str, object]:
        safe_episode_count = max(self.total_episodes, 1)
        safe_slate_count = max(self.total_slates, 1)
        safe_first_slate_count = max(self.single_slate_count, 1)

        reward_decay_curve = []
        for step_index in range(self.max_slates_per_episode):
            step_count = int(self.reward_decay_count[step_index])
            mean_reward = (
                float(self.reward_decay_sum[step_index] / step_count) if step_count > 0 else 0.0
            )
            reward_decay_curve.append(
                {
                    "slate_step": step_index + 1,
                    "mean_reward": mean_reward,
                    "count": step_count,
                }
            )

        examine_rate_by_position = (
            self.position_examine_count.astype(np.float64) / safe_slate_count
        )
        click_rate_by_position = (
            self.position_click_count.astype(np.float64) / safe_slate_count
        )
        ctr_given_examine_by_position = np.divide(
            self.position_click_count.astype(np.float64),
            np.maximum(self.position_examine_count, 1),
        )

        return {
            "reward_metrics": {
                "mean_slate_reward": float(self.total_slate_reward / safe_slate_count),
                "mean_clicks_per_slate": float(self.total_clicks / safe_slate_count),
                "slate_success_rate": float(self.slate_success_count / safe_slate_count),
                "multi_click_rate": float(self.multi_click_count / safe_slate_count),
                "single_slate_mean_reward": float(
                    self.single_slate_reward_sum / safe_first_slate_count
                ),
                "single_slate_success_rate": float(
                    self.single_slate_success_count / safe_first_slate_count
                ),
                "single_slate_multi_click_rate": float(
                    self.single_slate_multi_click_count / safe_first_slate_count
                ),
                "mean_episode_return": float(self.total_episode_return / safe_episode_count),
                "mean_episode_length": float(self.total_episode_length / safe_episode_count),
                "reward_decay_curve": reward_decay_curve,
            },
            "position_metrics": {
                "examine_rate_by_position": [
                    float(value) for value in examine_rate_by_position.tolist()
                ],
                "click_rate_by_position": [
                    float(value) for value in click_rate_by_position.tolist()
                ],
                "ctr_given_examine_by_position": [
                    float(value) for value in ctr_given_examine_by_position.tolist()
                ],
                "first_click_position_mean": float(
                    self.first_click_position_sum / max(self.first_click_defined_count, 1)
                ),
                "first_click_defined_rate": float(
                    self.first_click_defined_count / safe_slate_count
                ),
                "position_weighted_gain_mean": float(
                    self.position_weighted_gain_sum / safe_slate_count
                ),
            },
            "diversity_metrics": {
                "duplicate_rate": float(
                    self.total_duplicate_items / max(self.total_slates * self.slate_size, 1)
                ),
                "same_genre_repeat_rate": float(
                    self.total_same_genre_repeats / max(self.total_slates * self.slate_size, 1)
                ),
                "unique_genre_count_mean": float(
                    self.unique_genre_count_sum / safe_slate_count
                ),
                "genre_entropy_mean": float(self.genre_entropy_sum / safe_slate_count),
                "intra_list_diversity_mean": float(
                    self.intra_list_diversity_sum / safe_slate_count
                ),
                "catalog_coverage": float(
                    len(self.recommended_item_ids) / max(self.num_items, 1)
                ),
                "genre_coverage": float(
                    len(self.recommended_genres) / max(self.num_item_types, 1)
                ),
            },
            "offline_bridge_metrics": {
                "target_hit_at_5": float(self.target_hit_count / safe_first_slate_count),
                "target_ndcg_at_5": float(self.target_ndcg_sum / safe_first_slate_count),
                "target_mrr_at_5": float(self.target_mrr_sum / safe_first_slate_count),
                "genre_hit_at_5": float(self.genre_hit_count / safe_first_slate_count),
            },
            "raw_counts": {
                "episode_count": int(self.total_episodes),
                "slate_count": int(self.total_slates),
                "first_slate_count": int(self.single_slate_count),
                "total_clicks": float(self.total_clicks),
                "successful_slates": int(self.slate_success_count),
                "multi_click_slates": int(self.multi_click_count),
                "recommended_item_count": int(len(self.recommended_item_ids)),
                "recommended_genre_count": int(len(self.recommended_genres)),
            },
        }


def build_report_schema() -> dict[str, object]:
    return {
        "schema_version": "generative_slate_eval_v1",
        "top_level_fields": {
            "protocol": "Evaluation protocol metadata and run configuration.",
            "policies": "Per-policy metric reports keyed by policy name.",
            "policy_ranking": "Policy ranking sorted by core reward metrics.",
        },
        "policy_report_fields": {
            "reward_metrics": {
                "mean_slate_reward": "Average total clicks per generated slate.",
                "mean_clicks_per_slate": "Alias of average realized clicks per slate.",
                "slate_success_rate": "Fraction of slates with at least one click.",
                "multi_click_rate": "Fraction of slates with at least two clicks.",
                "single_slate_mean_reward": "Average reward of the first slate only.",
                "single_slate_success_rate": "First-slate hit rate.",
                "single_slate_multi_click_rate": "First-slate multi-click rate.",
                "mean_episode_return": "Average cumulative clicks over a full episode.",
                "mean_episode_length": "Average number of slates served per episode.",
                "reward_decay_curve": "Mean reward by slate step within an episode.",
            },
            "position_metrics": {
                "examine_rate_by_position": "Exposure probability realized at each slate position.",
                "click_rate_by_position": "Actual click-through rate at each slate position.",
                "ctr_given_examine_by_position": "Conditional CTR given exposure at each position.",
                "first_click_position_mean": "Mean 1-based position of the first click among clicked slates.",
                "first_click_defined_rate": "Fraction of slates with at least one click.",
                "position_weighted_gain_mean": "Discounted click gain weighted by 1/log2(k+1).",
            },
            "diversity_metrics": {
                "duplicate_rate": "Fraction of duplicate item slots across all slates.",
                "same_genre_repeat_rate": "Fraction of repeated primary-genre slots.",
                "unique_genre_count_mean": "Average number of distinct primary genres per slate.",
                "genre_entropy_mean": "Average entropy of the primary-genre mix within a slate.",
                "intra_list_diversity_mean": "Average pairwise genre dissimilarity inside a slate.",
                "catalog_coverage": "Fraction of the item catalog recommended at least once.",
                "genre_coverage": "Fraction of genres covered by the generated slates.",
            },
            "offline_bridge_metrics": {
                "target_hit_at_5": "Whether the held-out next item appears in the first slate.",
                "target_ndcg_at_5": "NDCG of the held-out next item in the first slate.",
                "target_mrr_at_5": "MRR of the held-out next item in the first slate.",
                "genre_hit_at_5": "Whether the target movie genre is covered in the first slate.",
            },
            "raw_counts": {
                "episode_count": "Number of Monte Carlo episode rollouts evaluated.",
                "slate_count": "Number of total slates evaluated.",
                "first_slate_count": "Number of first-slate observations used for bridge metrics.",
                "total_clicks": "Total realized clicks across all evaluated slates.",
                "successful_slates": "Slates with at least one click.",
                "multi_click_slates": "Slates with at least two clicks.",
                "recommended_item_count": "Unique item count recommended across the run.",
                "recommended_genre_count": "Unique genre count recommended across the run.",
            },
        },
    }


def build_policy_ranking(policy_reports: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    ranking_rows = []
    for policy_name, report in policy_reports.items():
        reward_metrics = report["reward_metrics"]
        diversity_metrics = report["diversity_metrics"]
        offline_bridge_metrics = report["offline_bridge_metrics"]
        ranking_rows.append(
            {
                "policy_name": policy_name,
                "mean_slate_reward": float(reward_metrics["mean_slate_reward"]),
                "mean_episode_return": float(reward_metrics["mean_episode_return"]),
                "slate_success_rate": float(reward_metrics["slate_success_rate"]),
                "target_hit_at_5": float(offline_bridge_metrics["target_hit_at_5"]),
                "intra_list_diversity_mean": float(
                    diversity_metrics["intra_list_diversity_mean"]
                ),
            }
        )

    ranking_rows.sort(
        key=lambda row: (
            row["mean_slate_reward"],
            row["mean_episode_return"],
            row["target_hit_at_5"],
            row["intra_list_diversity_mean"],
        ),
        reverse=True,
    )
    return ranking_rows


def evaluate_slate_policy(
    policy_name: str,
    model: SlateDQN | None,
    sequence_pool: dict[str, np.ndarray],
    env: SlateRecSimEnv,
    device: torch.device,
    env_rng: np.random.Generator,
    policy_rng: np.random.Generator,
    mc_rollouts: int = 20,
    rollout_batch_size: int = 128,
    max_states: int = 0,
    max_slates_per_episode: int | None = None,
    popularity_ranking: np.ndarray | None = None,
    candidate_pool_size: int = 50,
    blend_lambda: float = 0.0,
) -> dict[str, object]:
    input_ids = sequence_pool["input_ids"].astype(np.int64)
    target_ids = sequence_pool.get("target_ids")
    if target_ids is not None:
        target_ids = target_ids.astype(np.int64)
    user_ids = sequence_pool.get("user_ids")
    if user_ids is not None:
        user_ids = user_ids.astype(np.int64)

    base_state_count = input_ids.shape[0]
    if max_states > 0:
        base_state_count = min(base_state_count, max_states)
        input_ids = input_ids[:base_state_count]
        if target_ids is not None:
            target_ids = target_ids[:base_state_count]
        if user_ids is not None:
            user_ids = user_ids[:base_state_count]

    repeated_state_indices = np.repeat(np.arange(base_state_count, dtype=np.int64), mc_rollouts)
    if repeated_state_indices.size == 0:
        raise ValueError("No evaluation states found in the provided sequence pool.")

    if max_slates_per_episode is None:
        max_slates_per_episode = env.max_episode_steps

    accumulator = SlateEvaluationAccumulator(
        slate_size=env.slate_size,
        max_slates_per_episode=max_slates_per_episode,
        num_items=env.num_items,
        num_item_types=env.num_item_types,
        item_genres=env.item_genres,
    )

    for chunk_start in range(0, repeated_state_indices.size, rollout_batch_size):
        chunk_indices = repeated_state_indices[chunk_start : chunk_start + rollout_batch_size]
        chunk_size = int(chunk_indices.size)
        chunk_input_ids = input_ids[chunk_indices].copy()
        chunk_target_ids = (
            target_ids[chunk_indices].copy() if target_ids is not None else None
        )
        chunk_user_ids = user_ids[chunk_indices].copy() if user_ids is not None else None
        candidate_pool_batch = None
        sasrec_logits_batch = None
        if policy_name == "slate_dqn":
            if model is None:
                raise ValueError("A model is required for slate_dqn evaluation.")
            candidate_pool_batch = build_candidate_pool_batch(
                model=model,
                state_input_ids_batch=chunk_input_ids,
                candidate_pool_size=candidate_pool_size,
                device=device,
            )
            if blend_lambda != 0.0:
                sasrec_logits_batch = build_sasrec_logits_batch(
                    model=model,
                    state_input_ids_batch=chunk_input_ids,
                    device=device,
                )

        envs = [clone_env(env) for _ in range(chunk_size)]
        current_observations: list[dict[str, np.ndarray]] = []
        episode_returns = np.zeros(chunk_size, dtype=np.float64)
        episode_lengths = np.zeros(chunk_size, dtype=np.int64)
        active_mask = np.ones(chunk_size, dtype=bool)

        for row_index in range(chunk_size):
            reset_seed = int(env_rng.integers(0, 1_000_000_000))
            user_id = int(chunk_user_ids[row_index]) if chunk_user_ids is not None else None
            observation, _ = envs[row_index].reset(
                seed=reset_seed,
                options={
                    "input_ids": chunk_input_ids[row_index],
                    "user_id": user_id,
                },
            )
            current_observations.append(observation)

        for episode_step in range(max_slates_per_episode):
            active_indices = np.flatnonzero(active_mask)
            if active_indices.size == 0:
                break

            state_input_ids_batch = np.stack(
                [current_observations[int(active_index)]["input_ids"] for active_index in active_indices]
            ).astype(np.int64)
            active_candidate_pool_batch = None
            active_sasrec_logits_batch = None
            if candidate_pool_batch is not None:
                active_candidate_pool_batch = candidate_pool_batch[active_indices]
            if sasrec_logits_batch is not None:
                active_sasrec_logits_batch = sasrec_logits_batch[active_indices]
            slates = generate_policy_slates(
                policy_name=policy_name,
                model=model,
                state_input_ids_batch=state_input_ids_batch,
                slate_size=env.slate_size,
                num_items=env.num_items,
                device=device,
                policy_rng=policy_rng,
                popularity_ranking=popularity_ranking,
                candidate_pool_batch=active_candidate_pool_batch,
                sasrec_logits_batch=active_sasrec_logits_batch,
                blend_lambda=blend_lambda,
            )

            for local_index, active_index in enumerate(active_indices):
                slate_action = slates[local_index].astype(np.int64)
                if np.unique(slate_action).size != slate_action.size:
                    raise ValueError(
                        f"Policy {policy_name} generated duplicate items inside one slate: {slate_action.tolist()}"
                    )

                next_observation, reward, terminated, truncated, info = envs[int(active_index)].step(
                    slate_action
                )
                target_id = (
                    int(chunk_target_ids[int(active_index)])
                    if chunk_target_ids is not None
                    else None
                )
                accumulator.update_slate(
                    slate_item_ids=slate_action,
                    item_types=info["item_types"].astype(np.int64),
                    clicked_flags=info["clicked_flags"].astype(np.int64),
                    exposed_flags=info["exposed_flags"].astype(np.int64),
                    reward=float(reward),
                    episode_step=episode_step,
                    target_id=target_id,
                    is_first_slate=episode_step == 0,
                )

                episode_returns[int(active_index)] += float(reward)
                episode_lengths[int(active_index)] += 1
                current_observations[int(active_index)] = next_observation
                if terminated or truncated:
                    active_mask[int(active_index)] = False

        for row_index in range(chunk_size):
            accumulator.update_episode(
                episode_return=float(episode_returns[row_index]),
                episode_length=int(episode_lengths[row_index]),
            )

    return accumulator.finalize()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    resolved_mc_rollouts = resolve_mc_rollouts(args.split_name, args.mc_rollouts)
    resolved_max_eval_users = resolve_max_eval_users(
        args.max_eval_users,
        args.max_states,
    )
    resolved_max_slates_per_episode = resolve_max_slates_per_episode(
        args.split_name,
        args.max_slates_per_episode,
    )

    checkpoint_path = Path(args.checkpoint_path)
    sasrec_checkpoint_path = Path(args.sasrec_checkpoint_path)
    train_sequence_path = Path(args.train_sequence_path)
    eval_sequence_path = Path(args.eval_sequence_path)
    metadata_path = Path(args.metadata_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(metadata_path)
    train_sequence_pool = load_sequence_pool(train_sequence_path)
    eval_sequence_pool = load_sequence_pool(eval_sequence_path)
    popularity_ranking = compute_item_popularity(
        train_sequence_pool=train_sequence_pool,
        num_items=int(metadata["kept_item_count"]),
    )

    if args.skip_baselines:
        policy_names = ["slate_dqn"]
    else:
        policy_names = normalize_policy_names(args.policy_names)
    requires_model = any(
        policy_name in {"slate_dqn", "sasrec_topk"} for policy_name in policy_names
    )
    if "slate_dqn" in policy_names and not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Slate-DQN checkpoint not found: {checkpoint_path}"
        )
    model: SlateDQN | None = None
    if requires_model:
        model = build_model(
            metadata=metadata,
            checkpoint_path=checkpoint_path,
            sasrec_checkpoint_path=sasrec_checkpoint_path,
            slate_size=args.slate_size,
            device=device,
        )

    env = build_env(
        metadata=metadata,
        slate_size=args.slate_size,
        max_slates_per_episode=resolved_max_slates_per_episode,
    )

    policy_reports: dict[str, dict[str, object]] = {}
    for policy_index, policy_name in enumerate(policy_names):
        env_rng = np.random.default_rng(args.seed + 10_000)
        policy_rng = np.random.default_rng(args.seed + policy_index + 1)
        policy_reports[policy_name] = evaluate_slate_policy(
            policy_name=policy_name,
            model=model,
            sequence_pool=eval_sequence_pool,
            env=env,
            device=device,
            env_rng=env_rng,
            policy_rng=policy_rng,
            mc_rollouts=resolved_mc_rollouts,
            rollout_batch_size=args.rollout_batch_size,
            max_states=resolved_max_eval_users,
            max_slates_per_episode=resolved_max_slates_per_episode,
            popularity_ranking=popularity_ranking,
            candidate_pool_size=args.candidate_pool_size,
            blend_lambda=args.blend_lambda,
        )

    report = {
        "schema": build_report_schema(),
        "protocol": {
            "split_name": args.split_name,
            "checkpoint_path": str(checkpoint_path),
            "sasrec_checkpoint_path": str(sasrec_checkpoint_path),
            "train_sequence_path": str(train_sequence_path),
            "eval_sequence_path": str(eval_sequence_path),
            "metadata_path": str(metadata_path),
            "device": str(device),
            "seed": args.seed,
            "num_items": int(metadata["kept_item_count"]),
            "max_seq_len": int(metadata["max_seq_len"]),
            "slate_size": args.slate_size,
            "candidate_pool_size": args.candidate_pool_size,
            "blend_lambda": args.blend_lambda,
            "mc_rollouts": resolved_mc_rollouts,
            "rollout_batch_size": args.rollout_batch_size,
            "max_eval_users": resolved_max_eval_users,
            "max_slates_per_episode": resolved_max_slates_per_episode,
            "skip_baselines": args.skip_baselines,
            "policy_names": policy_names,
        },
        "policies": policy_reports,
        "policy_ranking": build_policy_ranking(policy_reports),
    }

    report_path = output_dir / f"{args.split_name}_generative_slate_report.json"
    with report_path.open("w", encoding="utf-8") as fout:
        json.dump(report, fout, ensure_ascii=False, indent=2)

    print(f"Evaluation report saved to: {report_path}")


if __name__ == "__main__":
    main()
