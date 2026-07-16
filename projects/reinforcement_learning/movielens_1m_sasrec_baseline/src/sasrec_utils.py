import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from sasrec_model import SASRec


class SequenceDataset(Dataset):
    def __init__(self, input_ids: np.ndarray, target_ids: np.ndarray) -> None:
        self.input_ids = torch.from_numpy(input_ids).long()
        self.target_ids = torch.from_numpy(target_ids).long()

    def __len__(self) -> int:
        return int(self.target_ids.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[index], self.target_ids[index]


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_metadata(data_dir: Path) -> dict:
    metadata_path = data_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as fin:
        return json.load(fin)


def load_sequence_dataset(data_dir: Path, split: str) -> SequenceDataset:
    split_path = data_dir / f"{split}_sequences.npz"
    if not split_path.exists():
        raise FileNotFoundError(f"Sequence split not found: {split_path}")

    data = np.load(split_path)
    return SequenceDataset(
        input_ids=data["input_ids"],
        target_ids=data["target_ids"],
    )


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


def evaluate_ranking_metrics(
    model: SASRec,
    dataloader: DataLoader,
    num_items: int,
    device: torch.device,
    topk: int = 10,
) -> dict[str, float]:
    total_users = 0
    hr_sum = 0.0
    ndcg_sum = 0.0

    model.eval()
    with torch.no_grad():
        for input_ids, target_ids in dataloader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            last_hidden_state = model.get_last_hidden_state(input_ids)
            logits = model.score_all_items(last_hidden_state)

            history_mask = torch.zeros_like(logits, dtype=torch.bool)
            valid_positions = input_ids.ne(0)
            if valid_positions.any():
                row_indices = (
                    torch.arange(input_ids.size(0), device=device)
                    .unsqueeze(1)
                    .expand_as(input_ids)
                )
                history_rows = row_indices[valid_positions]
                history_cols = input_ids[valid_positions] - 1
                history_mask[history_rows, history_cols] = True

            target_indices = target_ids - 1
            batch_indices = torch.arange(input_ids.size(0), device=device)
            history_mask[batch_indices, target_indices] = False

            logits = logits.masked_fill(history_mask, float("-inf"))

            topk_indices = torch.topk(logits, k=min(topk, num_items), dim=1).indices
            hr_batch, ndcg_batch = compute_hit_and_ndcg(topk_indices, target_indices)

            batch_size = input_ids.size(0)
            total_users += batch_size
            hr_sum += hr_batch * batch_size
            ndcg_sum += ndcg_batch * batch_size

    return {
        "hr_at_10": hr_sum / max(total_users, 1),
        "ndcg_at_10": ndcg_sum / max(total_users, 1),
        "user_count": total_users,
    }
