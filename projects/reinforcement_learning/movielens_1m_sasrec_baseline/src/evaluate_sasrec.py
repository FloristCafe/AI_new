import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sasrec_model import SASRec
from sasrec_utils import (
    evaluate_ranking_metrics,
    load_metadata,
    load_sequence_dataset,
    resolve_device,
)


DEFAULT_DATA_DIR = Path(
    r"D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_baseline\artifacts\preprocessed"
)
DEFAULT_OUTPUT_DIR = Path(
    r"D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_baseline\artifacts\predictions"
)
DEFAULT_CHECKPOINT_PATH = Path(
    r"D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_baseline\artifacts\checkpoints\sasrec_best.pt"
)


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


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(data_dir)
    dataset = load_sequence_dataset(data_dir, args.split)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    device = resolve_device(args.device)
    model = build_model(args, metadata, device)
    metrics = evaluate_ranking_metrics(
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
