import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from sasrec_model import SASRec
from train_sasrec import resolve_device


DEFAULT_DATA_DIR = Path(
    r"D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_baseline\artifacts\preprocessed"
)
DEFAULT_OUTPUT_DIR = Path(
    r"D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_baseline\artifacts\predictions"
)
DEFAULT_CHECKPOINT_PATH = Path(
    r"D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_baseline\artifacts\checkpoints\sasrec_best.pt"
)


class EvalDataset(Dataset):
    def __init__(self, input_ids: np.ndarray, target_ids: np.ndarray) -> None:
        self.input_ids = torch.from_numpy(input_ids).long()
        self.target_ids = torch.from_numpy(target_ids).long()

    def __len__(self) -> int:
        return int(self.target_ids.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[index], self.target_ids[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a SASRec baseline with HR@10 and NDCG@10."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="Directory containing preprocessed evaluation artifacts.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=str(DEFAULT_CHECKPOINT_PATH),
        help="Model checkpoint path.",
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
        choices=["valid", "test"],
        help="Evaluation split.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=64,
        help="Embedding dimension used during training.",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=2,
        help="Number of attention heads used during training.",
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=2,
        help="Number of transformer blocks used during training.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate used during training.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Torch device selection.",
    )
    return parser.parse_args()


def load_metadata(data_dir: Path) -> dict:
    metadata_path = data_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as fin:
        return json.load(fin)


def load_eval_dataset(data_dir: Path, split: str) -> EvalDataset:
    split_path = data_dir / f"{split}_sequences.npz"
    if not split_path.exists():
        raise FileNotFoundError(f"Evaluation split not found: {split_path}")

    data = np.load(split_path)
    return EvalDataset(
        input_ids=data["input_ids"],
        target_ids=data["target_ids"],
    )


def build_model(args: argparse.Namespace, metadata: dict, device: torch.device) -> SASRec:
    model = SASRec(
        num_items=metadata["kept_item_count"],
        max_seq_len=metadata["max_seq_len"],
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


def evaluate(
    model: SASRec,
    dataloader: DataLoader,
    num_items: int,
    device: torch.device,
) -> dict[str, float]:
    total_users = 0
    hr_sum = 0.0
    ndcg_sum = 0.0

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
            # Re-enable the target column in case the target item appeared before.
            history_mask[batch_indices, target_indices] = False

            logits = logits.masked_fill(history_mask, float("-inf"))

            topk = torch.topk(logits, k=min(10, num_items), dim=1).indices
            hr_batch, ndcg_batch = compute_hit_and_ndcg(topk, target_indices)

            batch_size = input_ids.size(0)
            total_users += batch_size
            hr_sum += hr_batch * batch_size
            ndcg_sum += ndcg_batch * batch_size

    return {
        "hr_at_10": hr_sum / max(total_users, 1),
        "ndcg_at_10": ndcg_sum / max(total_users, 1),
        "user_count": total_users,
    }


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(data_dir)
    dataset = load_eval_dataset(data_dir, args.split)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    device = resolve_device(args.device)
    model = build_model(args, metadata, device)
    metrics = evaluate(
        model=model,
        dataloader=dataloader,
        num_items=metadata["kept_item_count"],
        device=device,
    )

    result = {
        "split": args.split,
        "data_dir": str(data_dir),
        "checkpoint_path": str(Path(args.checkpoint_path)),
        "device": str(device),
        "hr_at_10": float(metrics["hr_at_10"]),
        "ndcg_at_10": float(metrics["ndcg_at_10"]),
        "user_count": int(metrics["user_count"]),
    }

    result_path = output_dir / f"{args.split}_metrics.json"
    with result_path.open("w", encoding="utf-8") as fout:
        json.dump(result, fout, ensure_ascii=False, indent=2)

    print("Evaluation finished.")
    print(f"Split: {args.split}")
    print(f"HR@10: {result['hr_at_10']:.6f}")
    print(f"NDCG@10: {result['ndcg_at_10']:.6f}")
    print(f"Summary saved to: {result_path}")


if __name__ == "__main__":
    main()
