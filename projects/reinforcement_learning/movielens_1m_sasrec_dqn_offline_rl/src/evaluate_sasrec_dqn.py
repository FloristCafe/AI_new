from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ml_1m_genre_utils import (
    DEFAULT_ITEM_MAPPING_PATH,
    DEFAULT_MOVIES_PATH,
    compute_recommendation_reward,
    load_mapped_movie_genres,
)
from sasrec_dqn_model import SASRecDQN
from train_sasrec_dqn import load_offline_buffer_metadata, resolve_device


DEFAULT_OFFLINE_BUFFER_DIR = Path(
    r"D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_dqn_offline_rl\artifacts\offline_buffer"
)
DEFAULT_OUTPUT_DIR = Path(
    r"D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_dqn_offline_rl\artifacts\predictions"
)
DEFAULT_CHECKPOINT_PATH = Path(
    r"D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_dqn_offline_rl\artifacts\checkpoints\sasrec_dqn_best.pt"
)


class OfflineReplayEvalDataset(Dataset):
    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        user_ids: np.ndarray,
        step_indices: np.ndarray,
    ) -> None:
        self.states = torch.from_numpy(states).long()
        self.actions = torch.from_numpy(actions).long()
        self.user_ids = torch.from_numpy(user_ids).long()
        self.step_indices = torch.from_numpy(step_indices).long()

    def __len__(self) -> int:
        return int(self.actions.shape[0])

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.states[index],
            self.actions[index],
            self.user_ids[index],
            self.step_indices[index],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SASRec-DQN with top-k ranking metrics and trajectory rewards."
    )
    parser.add_argument(
        "--offline-buffer-dir",
        type=str,
        default=str(DEFAULT_OFFLINE_BUFFER_DIR),
        help="Directory containing offline replay buffer artifacts.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=str(DEFAULT_CHECKPOINT_PATH),
        help="Trained SASRec-DQN checkpoint path.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for evaluation outputs.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "valid", "test", "all"],
        help="Replay buffer split used for evaluation.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Top-k used for ranking metrics.",
    )
    parser.add_argument(
        "--item-mapping-path",
        type=str,
        default=str(DEFAULT_ITEM_MAPPING_PATH),
        help="Path to upstream item mapping CSV.",
    )
    parser.add_argument(
        "--movies-path",
        type=str,
        default=str(DEFAULT_MOVIES_PATH),
        help="Path to MovieLens-1M movies.dat metadata file.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=128,
        help="Embedding dimension used by the trained SASRec-DQN model.",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=2,
        help="Number of attention heads used by the trained SASRec-DQN model.",
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=3,
        help="Number of transformer blocks used by the trained SASRec-DQN model.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate used by the trained SASRec-DQN model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Torch device selection.",
    )
    return parser.parse_args()


def load_eval_dataset(
    offline_buffer_dir: Path,
    split: str,
) -> OfflineReplayEvalDataset:
    split_path = offline_buffer_dir / f"{split}_replay_buffer.npz"
    if not split_path.exists():
        raise FileNotFoundError(f"Replay buffer split not found: {split_path}")

    data = np.load(split_path)
    return OfflineReplayEvalDataset(
        states=data["states"],
        actions=data["actions"],
        user_ids=data["user_ids"],
        step_indices=data["step_indices"],
    )


def build_model(args: argparse.Namespace, metadata: dict, device: torch.device) -> SASRecDQN:
    model = SASRecDQN(
        num_items=int(metadata["kept_item_count"]),
        max_seq_len=int(metadata["max_seq_len"]),
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
    ).to(device)

    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def compute_hit_and_ndcg(
    topk_indices: torch.Tensor,
    target_indices: torch.Tensor,
) -> tuple[float, float]:
    hits = 0.0
    ndcg = 0.0

    for row_idx in range(topk_indices.size(0)):
        matches = torch.nonzero(
            topk_indices[row_idx] == target_indices[row_idx],
            as_tuple=False,
        )
        if matches.numel() > 0:
            rank = int(matches[0, 0].item())
            hits += 1.0
            ndcg += 1.0 / np.log2(rank + 2.0)

    batch_size = max(topk_indices.size(0), 1)
    return hits / batch_size, ndcg / batch_size


def mask_seen_items(
    q_values: torch.Tensor,
    states: torch.Tensor,
    target_actions: torch.Tensor,
) -> torch.Tensor:
    masked_q_values = q_values.clone()
    history_mask = torch.zeros_like(masked_q_values, dtype=torch.bool)
    valid_positions = states.ne(0)
    if valid_positions.any():
        row_indices = (
            torch.arange(states.size(0), device=states.device)
            .unsqueeze(1)
            .expand_as(states)
        )
        history_rows = row_indices[valid_positions]
        history_cols = states[valid_positions] - 1
        history_mask[history_rows, history_cols] = True

    target_indices = target_actions - 1
    batch_indices = torch.arange(states.size(0), device=states.device)
    history_mask[batch_indices, target_indices] = False
    return masked_q_values.masked_fill(history_mask, float("-inf"))


def evaluate_model(
    model: SASRecDQN,
    dataloader: DataLoader,
    mapped_movie_genres: dict[int, set[str]],
    device: torch.device,
    topk: int,
    num_items: int,
) -> dict[str, object]:
    total_examples = 0
    hr_sum = 0.0
    ndcg_sum = 0.0
    top1_reward_sum = 0.0
    top1_exact_hit_count = 0
    top1_genre_match_count = 0
    per_user_reward_sum: dict[int, float] = {}
    per_user_step_count: dict[int, int] = {}

    model.eval()
    with torch.no_grad():
        for states, actions, user_ids, step_indices in dataloader:
            del step_indices
            states = states.to(device)
            actions = actions.to(device)
            user_ids = user_ids.to(device)

            q_values = model.get_q_values(states)
            masked_q_values = mask_seen_items(q_values, states, actions)

            topk_indices = torch.topk(
                masked_q_values,
                k=min(topk, num_items),
                dim=1,
            ).indices
            target_indices = actions - 1
            hr_batch, ndcg_batch = compute_hit_and_ndcg(topk_indices, target_indices)

            top1_indices = topk_indices[:, 0]
            recommended_action_ids = top1_indices + 1

            batch_size = states.size(0)
            total_examples += batch_size
            hr_sum += hr_batch * batch_size
            ndcg_sum += ndcg_batch * batch_size

            for row_idx in range(batch_size):
                user_id = int(user_ids[row_idx].item())
                target_action = int(actions[row_idx].item())
                recommended_action = int(recommended_action_ids[row_idx].item())

                reward = compute_recommendation_reward(
                    recommended_item_id=recommended_action,
                    target_item_id=target_action,
                    mapped_movie_genres=mapped_movie_genres,
                )
                top1_reward_sum += reward
                if reward == 1.0:
                    top1_exact_hit_count += 1
                elif reward == 0.1:
                    top1_genre_match_count += 1

                per_user_reward_sum[user_id] = per_user_reward_sum.get(user_id, 0.0) + reward
                per_user_step_count[user_id] = per_user_step_count.get(user_id, 0) + 1

    per_user_rewards = list(per_user_reward_sum.values())
    per_user_lengths = list(per_user_step_count.values())
    mean_cumulative_reward = sum(per_user_rewards) / max(len(per_user_rewards), 1)
    mean_episode_length = sum(per_user_lengths) / max(len(per_user_lengths), 1)

    return {
        "hr_at_k": hr_sum / max(total_examples, 1),
        "ndcg_at_k": ndcg_sum / max(total_examples, 1),
        "step_count": total_examples,
        "user_count": len(per_user_rewards),
        "top1_average_reward": top1_reward_sum / max(total_examples, 1),
        "top1_exact_hit_rate": top1_exact_hit_count / max(total_examples, 1),
        "top1_genre_match_rate": top1_genre_match_count / max(total_examples, 1),
        "mean_cumulative_reward_per_user": mean_cumulative_reward,
        "min_cumulative_reward_per_user": min(per_user_rewards) if per_user_rewards else 0.0,
        "max_cumulative_reward_per_user": max(per_user_rewards) if per_user_rewards else 0.0,
        "mean_episode_length": mean_episode_length,
    }


def main() -> None:
    args = parse_args()

    offline_buffer_dir = Path(args.offline_buffer_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_offline_buffer_metadata(offline_buffer_dir)
    dataset = load_eval_dataset(offline_buffer_dir, args.split)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    mapped_movie_genres = load_mapped_movie_genres(
        movies_path=Path(args.movies_path),
        item_mapping_path=Path(args.item_mapping_path),
    )

    device = resolve_device(args.device)
    model = build_model(args, metadata, device)
    metrics = evaluate_model(
        model=model,
        dataloader=dataloader,
        mapped_movie_genres=mapped_movie_genres,
        device=device,
        topk=args.topk,
        num_items=int(metadata["kept_item_count"]),
    )

    result = {
        "split": args.split,
        "offline_buffer_dir": str(offline_buffer_dir),
        "checkpoint_path": str(Path(args.checkpoint_path)),
        "device": str(device),
        "topk": args.topk,
        "hr_at_k": float(metrics["hr_at_k"]),
        "ndcg_at_k": float(metrics["ndcg_at_k"]),
        "step_count": int(metrics["step_count"]),
        "user_count": int(metrics["user_count"]),
        "top1_average_reward": float(metrics["top1_average_reward"]),
        "top1_exact_hit_rate": float(metrics["top1_exact_hit_rate"]),
        "top1_genre_match_rate": float(metrics["top1_genre_match_rate"]),
        "mean_cumulative_reward_per_user": float(metrics["mean_cumulative_reward_per_user"]),
        "min_cumulative_reward_per_user": float(metrics["min_cumulative_reward_per_user"]),
        "max_cumulative_reward_per_user": float(metrics["max_cumulative_reward_per_user"]),
        "mean_episode_length": float(metrics["mean_episode_length"]),
    }

    result_path = output_dir / f"{args.split}_sasrec_dqn_metrics.json"
    with result_path.open("w", encoding="utf-8") as fout:
        json.dump(result, fout, ensure_ascii=False, indent=2)

    print("SASRec-DQN evaluation finished.")
    print(f"Split: {args.split}")
    print(f"HR@{args.topk}: {result['hr_at_k']:.6f}")
    print(f"NDCG@{args.topk}: {result['ndcg_at_k']:.6f}")
    print(f"Top1 average reward: {result['top1_average_reward']:.6f}")
    print(
        "Mean cumulative reward per user: "
        f"{result['mean_cumulative_reward_per_user']:.6f}"
    )
    print(f"Summary saved to: {result_path}")


if __name__ == "__main__":
    main()
