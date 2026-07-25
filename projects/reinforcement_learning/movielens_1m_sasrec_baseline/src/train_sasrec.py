import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from sasrec_model import SASRec
from sasrec_utils import (
    evaluate_ranking_metrics,
    load_metadata,
    load_sequence_dataset,
    load_sequence_supervision_dataset,
    resolve_device,
    sample_uniform_negative_ids,
)


DEFAULT_DATA_DIR = Path(
    r"D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_baseline\artifacts\preprocessed"
)
DEFAULT_OUTPUT_DIR = Path(
    r"D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_baseline\artifacts"
)


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
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Training batch size.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=256,
        help="Validation batch size.",
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
        "--selection-metric",
        type=str,
        default="ndcg_at_10",
        choices=["hr_at_10", "ndcg_at_10"],
        help="Validation metric used to choose the best checkpoint.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=3,
        help="Stop if the validation metric does not improve for this many epochs. Set 0 to disable.",
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        default="cross_entropy",
        choices=["cross_entropy", "bce_negative_sampling"],
        help="Training objective. Keep cross_entropy for the existing baseline, or switch to BCE with sampled negatives for paper alignment.",
    )
    parser.add_argument(
        "--num-negative-samples",
        type=int,
        default=1,
        help="Number of sampled negatives per valid sequence position when using BCE training.",
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(args: argparse.Namespace, metadata: dict, device: torch.device) -> SASRec:
    return SASRec(
        num_items=metadata["kept_item_count"],
        max_seq_len=metadata["max_seq_len"],
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
    ).to(device)


def train_one_epoch_cross_entropy(
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


def train_one_epoch_bce_negative_sampling(
    model: SASRec,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_items: int,
    num_negative_samples: int,
) -> dict[str, float]:
    model.train()
    loss_sum = 0.0
    correct_sum = 0
    position_count = 0

    for input_ids, positive_ids in dataloader:
        input_ids = input_ids.to(device)
        positive_ids = positive_ids.to(device)
        valid_mask = positive_ids.ne(0)
        valid_positions = int(valid_mask.sum().item())
        if valid_positions == 0:
            continue

        negative_ids = sample_uniform_negative_ids(
            input_ids=input_ids,
            positive_ids=positive_ids,
            num_items=num_items,
            num_negative_samples=num_negative_samples,
        )

        optimizer.zero_grad(set_to_none=True)
        hidden_states = model.encode(input_ids)
        positive_logits = model.score_candidates(hidden_states, positive_ids)
        negative_logits = model.score_candidates(hidden_states, negative_ids)

        positive_loss = F.binary_cross_entropy_with_logits(
            positive_logits,
            torch.ones_like(positive_logits),
            reduction="none",
        )
        if num_negative_samples == 1:
            negative_loss = F.binary_cross_entropy_with_logits(
                negative_logits,
                torch.zeros_like(negative_logits),
                reduction="none",
            )
            comparison = positive_logits > negative_logits
        else:
            negative_loss = F.binary_cross_entropy_with_logits(
                negative_logits,
                torch.zeros_like(negative_logits),
                reduction="none",
            ).mean(dim=-1)
            comparison = (positive_logits.unsqueeze(-1) > negative_logits).all(dim=-1)

        loss = ((positive_loss + negative_loss) * valid_mask.float()).sum() / valid_positions
        loss.backward()
        optimizer.step()

        loss_sum += float(loss.item()) * valid_positions
        correct_sum += int(comparison[valid_mask].sum().item())
        position_count += valid_positions

    average_loss = loss_sum / max(position_count, 1)
    accuracy = correct_sum / max(position_count, 1)
    return {
        "loss": float(average_loss),
        "accuracy": float(accuracy),
    }


def train_one_epoch(
    model: SASRec,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_type: str,
    num_items: int,
    num_negative_samples: int,
    criterion: nn.Module | None = None,
) -> dict[str, float]:
    if loss_type == "cross_entropy":
        if criterion is None:
            raise ValueError("criterion is required for cross_entropy training.")
        return train_one_epoch_cross_entropy(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
    if loss_type == "bce_negative_sampling":
        return train_one_epoch_bce_negative_sampling(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            num_items=num_items,
            num_negative_samples=num_negative_samples,
        )
    raise ValueError(f"Unsupported loss type: {loss_type}")


def is_metric_improved(current_value: float, best_value: float) -> bool:
    return current_value > best_value


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    metrics_dir = output_dir / "metrics"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if args.loss_type == "bce_negative_sampling" and args.num_negative_samples <= 0:
        raise ValueError("--num-negative-samples must be at least 1.")

    metadata = load_metadata(data_dir)
    if args.loss_type == "cross_entropy":
        train_dataset = load_sequence_dataset(data_dir, split="train")
        train_count_summary = {
            "train_dataset_size": len(train_dataset),
            "train_supervision_position_count": len(train_dataset),
            "train_count_unit": "single_target_samples",
        }
    else:
        train_dataset = load_sequence_supervision_dataset(data_dir)
        train_count_summary = {
            "train_dataset_size": len(train_dataset),
            "train_supervision_position_count": int(
                metadata.get("train_supervision_position_count", 0)
            ),
            "train_count_unit": "sequence_positions",
        }
    valid_dataset = load_sequence_dataset(data_dir, split="valid")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
    )

    device = resolve_device(args.device)
    model = build_model(args, metadata, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss() if args.loss_type == "cross_entropy" else None

    epoch_metrics: list[dict[str, float]] = []
    best_metric_value = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False

    best_checkpoint_path = checkpoints_dir / "sasrec_best.pt"
    final_checkpoint_path = checkpoints_dir / "sasrec_final.pt"

    for epoch_idx in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            device=device,
            loss_type=args.loss_type,
            num_items=metadata["kept_item_count"],
            num_negative_samples=args.num_negative_samples,
            criterion=criterion,
        )
        valid_metrics = evaluate_ranking_metrics(
            model=model,
            dataloader=valid_dataloader,
            num_items=metadata["kept_item_count"],
            device=device,
        )

        selected_metric_value = float(valid_metrics[args.selection_metric])
        improved = is_metric_improved(selected_metric_value, best_metric_value)
        if improved:
            best_metric_value = selected_metric_value
            best_epoch = epoch_idx
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_checkpoint_path)
        else:
            epochs_without_improvement += 1

        epoch_summary = {
            "epoch": epoch_idx,
            "train_loss": float(train_metrics["loss"]),
            "train_accuracy": float(train_metrics["accuracy"]),
            "valid_hr_at_10": float(valid_metrics["hr_at_10"]),
            "valid_ndcg_at_10": float(valid_metrics["ndcg_at_10"]),
            "selection_metric": args.selection_metric,
            "selection_metric_value": selected_metric_value,
            "is_best_epoch": improved,
            "epochs_without_improvement": epochs_without_improvement,
        }
        epoch_metrics.append(epoch_summary)

        print(
            f"Epoch {epoch_idx}/{args.epochs} - "
            f"train_loss={epoch_summary['train_loss']:.6f} - "
            f"train_accuracy={epoch_summary['train_accuracy']:.6f} - "
            f"valid_hr@10={epoch_summary['valid_hr_at_10']:.6f} - "
            f"valid_ndcg@10={epoch_summary['valid_ndcg_at_10']:.6f}"
        )

        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            stopped_early = True
            print(
                f"Early stopping triggered at epoch {epoch_idx}. "
                f"No improvement in {args.selection_metric} for "
                f"{args.early_stop_patience} consecutive epochs."
            )
            break

    torch.save(model.state_dict(), final_checkpoint_path)

    training_summary = {
        "data_dir": str(data_dir),
        "device": str(device),
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "epochs_completed": len(epoch_metrics),
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "embedding_dim": args.embedding_dim,
        "num_heads": args.num_heads,
        "num_blocks": args.num_blocks,
        "dropout": args.dropout,
        "loss_type": args.loss_type,
        "num_negative_samples": args.num_negative_samples,
        "selection_metric": args.selection_metric,
        "early_stop_patience": args.early_stop_patience,
        "stopped_early": stopped_early,
        "train_metric_definition": (
            "next_item_classification_accuracy"
            if args.loss_type == "cross_entropy"
            else "positive_logit_greater_than_negative_logit_rate"
        ),
        "valid_user_count": len(valid_dataset),
        "num_items": metadata["kept_item_count"],
        "max_seq_len": metadata["max_seq_len"],
        "best_epoch": best_epoch,
        "best_selection_metric_value": best_metric_value,
        "best_checkpoint_path": str(best_checkpoint_path),
        "final_checkpoint_path": str(final_checkpoint_path),
        "epoch_metrics": epoch_metrics,
    }
    training_summary.update(train_count_summary)

    metrics_path = metrics_dir / "training_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fout:
        json.dump(training_summary, fout, ensure_ascii=False, indent=2)

    print("Training finished.")
    print(
        f"Best epoch: {best_epoch} - "
        f"{args.selection_metric}={best_metric_value:.6f}"
    )
    print(f"Best checkpoint saved to: {best_checkpoint_path}")
    print(f"Final checkpoint saved to: {final_checkpoint_path}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
