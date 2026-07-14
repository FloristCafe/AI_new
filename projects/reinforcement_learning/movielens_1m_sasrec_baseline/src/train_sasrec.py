import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from sasrec_model import SASRec


DEFAULT_DATA_DIR = Path(
    r"D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_baseline\artifacts\preprocessed"
)
DEFAULT_OUTPUT_DIR = Path(
    r"D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_baseline\artifacts"
)


class SequenceDataset(Dataset):
    def __init__(self, input_ids: np.ndarray, target_ids: np.ndarray) -> None:
        self.input_ids = torch.from_numpy(input_ids).long()
        self.target_ids = torch.from_numpy(target_ids).long()

    def __len__(self) -> int:
        return int(self.target_ids.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[index], self.target_ids[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a SASRec baseline on MovieLens-1M."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="Directory containing preprocessed sequence artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for checkpoints and metrics.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Training batch size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Adam weight decay.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=64,
        help="Embedding dimension.",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=2,
        help="Number of attention heads.",
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=2,
        help="Number of transformer blocks.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate.",
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_metadata(data_dir: Path) -> dict:
    metadata_path = data_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as fin:
        return json.load(fin)


def load_training_dataset(data_dir: Path) -> SequenceDataset:
    train_path = data_dir / "train_sequences.npz"
    if not train_path.exists():
        raise FileNotFoundError(f"Training sequences not found: {train_path}")

    data = np.load(train_path)
    return SequenceDataset(
        input_ids=data["input_ids"],
        target_ids=data["target_ids"],
    )


def build_model(args: argparse.Namespace, metadata: dict, device: torch.device) -> SASRec:
    return SASRec(
        num_items=metadata["kept_item_count"],
        max_seq_len=metadata["max_seq_len"],
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
    ).to(device)


def train_one_epoch(
    model: SASRec,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    loss_sum = 0.0
    correct_sum = 0
    sample_count = 0

    for input_ids, target_ids in dataloader:
        input_ids = input_ids.to(device)
        # CrossEntropyLoss expects class indices in [0, num_classes - 1].
        target_ids = target_ids.to(device) - 1

        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss = criterion(logits, target_ids)
        loss.backward()
        optimizer.step()

        batch_size = input_ids.size(0)
        loss_sum += float(loss.item()) * batch_size
        predictions = torch.argmax(logits, dim=1)
        correct_sum += int((predictions == target_ids).sum().item())
        sample_count += batch_size

    average_loss = loss_sum / max(sample_count, 1)
    accuracy = correct_sum / max(sample_count, 1)
    return {
        "loss": float(average_loss),
        "accuracy": float(accuracy),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    metrics_dir = output_dir / "metrics"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(data_dir)
    train_dataset = load_training_dataset(data_dir)
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    device = resolve_device(args.device)
    model = build_model(args, metadata, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    epoch_metrics: list[dict[str, float]] = []
    best_loss = float("inf")
    best_checkpoint_path = checkpoints_dir / "sasrec_best.pt"

    for epoch_idx in range(1, args.epochs + 1):
        metrics = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        metrics["epoch"] = float(epoch_idx)
        epoch_metrics.append(metrics)

        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            torch.save(model.state_dict(), best_checkpoint_path)

        print(
            f"Epoch {epoch_idx}/{args.epochs} - "
            f"loss={metrics['loss']:.6f} - "
            f"accuracy={metrics['accuracy']:.6f}"
        )

    final_checkpoint_path = checkpoints_dir / "sasrec_final.pt"
    torch.save(model.state_dict(), final_checkpoint_path)

    training_summary = {
        "data_dir": str(data_dir),
        "device": str(device),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "embedding_dim": args.embedding_dim,
        "num_heads": args.num_heads,
        "num_blocks": args.num_blocks,
        "dropout": args.dropout,
        "train_sample_count": len(train_dataset),
        "num_items": metadata["kept_item_count"],
        "max_seq_len": metadata["max_seq_len"],
        "best_loss": best_loss,
        "best_checkpoint_path": str(best_checkpoint_path),
        "final_checkpoint_path": str(final_checkpoint_path),
        "epoch_metrics": epoch_metrics,
    }

    metrics_path = metrics_dir / "training_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fout:
        json.dump(training_summary, fout, ensure_ascii=False, indent=2)

    print("Training finished.")
    print(f"Best checkpoint saved to: {best_checkpoint_path}")
    print(f"Final checkpoint saved to: {final_checkpoint_path}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
