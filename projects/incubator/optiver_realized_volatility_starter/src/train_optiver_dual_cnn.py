import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


DEFAULT_TENSOR_DIR = Path(
    r"D:\Python\Datasets\optiver_realized_volatility_prediction\samples\optiver_sandbox_stocks_0-1-2-3-4-5-6-7-8-9-10-11-13-14-15-16-17-18-19-20-21-22-23-24_times_200\cnn_tensors"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Optiver CNN with book-only, trade-only or dual-branch modes."
    )
    parser.add_argument(
        "--tensor-dir",
        type=str,
        default=str(DEFAULT_TENSOR_DIR),
        help="Directory containing book_tensor.npy, trade_tensor.npy, target.npy, stock_id.npy and time_id.npy.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory for metrics and predictions. Defaults to tensor_dir/cnn_models.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="dual",
        choices=["book_only", "trade_only", "dual"],
        help="Which branch configuration to train.",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.2,
        help="Validation ratio using chronological order.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Mini-batch size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-4,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Adam weight decay.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="Hidden dimension after branch pooling.",
    )
    parser.add_argument(
        "--book-weight",
        type=float,
        default=0.5,
        help="Weight of the book branch contribution in dual mode.",
    )
    parser.add_argument(
        "--trade-weight",
        type=float,
        default=0.5,
        help="Weight of the trade branch contribution in dual mode.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout in the prediction head.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device.",
    )
    return parser.parse_args()


def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.where(y_true == 0, 1e-8, y_true)
    return float(np.sqrt(np.mean(np.square((y_true - y_pred) / denominator))))


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "rmspe": rmspe(y_true, y_pred),
    }


def standardize_tensor(
    train_tensor: np.ndarray,
    valid_tensor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    channel_mean = train_tensor.mean(axis=(0, 2), keepdims=True)
    channel_std = train_tensor.std(axis=(0, 2), keepdims=True)
    channel_std = np.where(channel_std < 1e-6, 1.0, channel_std)
    train_scaled = (train_tensor - channel_mean) / channel_std
    valid_scaled = (valid_tensor - channel_mean) / channel_std
    return train_scaled, valid_scaled, channel_mean.squeeze(), channel_std.squeeze()


class OptiverTensorDataset(Dataset):
    def __init__(
        self,
        book_tensor: np.ndarray,
        trade_tensor: np.ndarray,
        target_scaled: np.ndarray,
    ) -> None:
        self.book_tensor = torch.from_numpy(np.asarray(book_tensor, dtype=np.float32))
        self.trade_tensor = torch.from_numpy(np.asarray(trade_tensor, dtype=np.float32))
        self.target = torch.from_numpy(np.asarray(target_scaled, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, idx: int):
        return self.book_tensor[idx], self.trade_tensor[idx], self.target[idx]


class ConvBranch(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(64, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x).squeeze(-1)
        return self.proj(out)


class DualBranchCNN(nn.Module):
    def __init__(
        self,
        mode: str,
        book_channels: int,
        trade_channels: int,
        hidden_dim: int,
        dropout: float,
        book_weight: float,
        trade_weight: float,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.book_branch = ConvBranch(book_channels, hidden_dim) if book_channels > 0 else None
        self.trade_branch = (
            ConvBranch(trade_channels, hidden_dim) if trade_channels > 0 else None
        )

        self.book_weight = book_weight
        self.trade_weight = trade_weight
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, book_x: torch.Tensor, trade_x: torch.Tensor) -> torch.Tensor:
        if self.mode == "book_only":
            fused = self.book_branch(book_x)
        elif self.mode == "trade_only":
            fused = self.trade_branch(trade_x)
        else:
            book_feat = self.book_branch(book_x)
            trade_feat = self.trade_branch(trade_x)
            fused = self.book_weight * book_feat + self.trade_weight * trade_feat
        return self.head(fused).squeeze(-1)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: str,
) -> tuple[float, np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    model.train(is_train)
    criterion = nn.MSELoss()

    losses: list[float] = []
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    for book_x, trade_x, target in loader:
        book_x = book_x.to(device)
        trade_x = trade_x.to(device)
        target = target.to(device)

        pred = model(book_x, trade_x)
        loss = criterion(pred, target)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        preds.append(pred.detach().cpu().numpy())
        targets.append(target.detach().cpu().numpy())

    return (
        float(np.mean(losses)),
        np.concatenate(preds),
        np.concatenate(targets),
    )


def inverse_target_scale(pred_scaled: np.ndarray, target_mean: float, target_std: float) -> np.ndarray:
    return pred_scaled * target_std + target_mean


def main() -> None:
    args = parse_args()
    if args.mode == "dual":
        if args.book_weight < 0 or args.trade_weight < 0:
            raise ValueError("book_weight and trade_weight must be non-negative.")
        if args.book_weight == 0 and args.trade_weight == 0:
            raise ValueError("book_weight and trade_weight cannot both be zero.")

    tensor_dir = Path(args.tensor_dir)
    output_dir = Path(args.output_dir) if args.output_dir else tensor_dir / "cnn_models"
    output_dir.mkdir(parents=True, exist_ok=True)

    book_tensor = np.load(tensor_dir / "book_tensor.npy").astype(np.float32)
    trade_tensor = np.load(tensor_dir / "trade_tensor.npy").astype(np.float32)
    target = np.load(tensor_dir / "target.npy").astype(np.float32)
    stock_id = np.load(tensor_dir / "stock_id.npy")
    time_id = np.load(tensor_dir / "time_id.npy")

    order = np.lexsort((stock_id, time_id))
    book_tensor = book_tensor[order]
    trade_tensor = trade_tensor[order]
    target = target[order]
    stock_id = stock_id[order]
    time_id = time_id[order]

    split_index = max(1, int(len(target) * (1 - args.valid_ratio)))

    book_train_raw = book_tensor[:split_index]
    book_valid_raw = book_tensor[split_index:]
    trade_train_raw = trade_tensor[:split_index]
    trade_valid_raw = trade_tensor[split_index:]
    y_train = target[:split_index]
    y_valid = target[split_index:]

    book_train, book_valid, book_mean, book_std = standardize_tensor(
        book_train_raw, book_valid_raw
    )
    trade_train, trade_valid, trade_mean, trade_std = standardize_tensor(
        trade_train_raw, trade_valid_raw
    )

    target_mean = float(y_train.mean())
    target_std = float(y_train.std())
    if target_std < 1e-8:
        target_std = 1.0
    y_train_scaled = (y_train - target_mean) / target_std
    y_valid_scaled = (y_valid - target_mean) / target_std

    train_dataset = OptiverTensorDataset(book_train, trade_train, y_train_scaled)
    valid_dataset = OptiverTensorDataset(book_valid, trade_valid, y_valid_scaled)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    train_eval_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    weight_sum = args.book_weight + args.trade_weight
    book_weight = args.book_weight / weight_sum if weight_sum > 0 else 1.0
    trade_weight = args.trade_weight / weight_sum if weight_sum > 0 else 0.0

    model = DualBranchCNN(
        mode=args.mode,
        book_channels=book_tensor.shape[1],
        trade_channels=trade_tensor.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        book_weight=book_weight,
        trade_weight=trade_weight,
    ).to(args.device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_valid_rmse = float("inf")
    best_valid_pred = None
    best_train_pred = None

    for _ in range(args.epochs):
        _, _, _ = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=args.device,
        )
        _, train_pred_scaled, _ = run_epoch(
            model=model,
            loader=train_eval_loader,
            optimizer=None,
            device=args.device,
        )
        _, valid_pred_scaled, _ = run_epoch(
            model=model,
            loader=valid_loader,
            optimizer=None,
            device=args.device,
        )

        train_pred = inverse_target_scale(train_pred_scaled, target_mean, target_std)
        valid_pred = inverse_target_scale(valid_pred_scaled, target_mean, target_std)
        train_pred = np.clip(train_pred, 0.0, None)
        valid_pred = np.clip(valid_pred, 0.0, None)

        valid_rmse = np.sqrt(mean_squared_error(y_valid, valid_pred))
        if valid_rmse < best_valid_rmse:
            best_valid_rmse = valid_rmse
            best_valid_pred = valid_pred.copy()
            best_train_pred = train_pred.copy()

    train_metrics = evaluate_predictions(y_train, best_train_pred)
    valid_metrics = evaluate_predictions(y_valid, best_valid_pred)

    metrics = {
        "train": train_metrics,
        "valid": valid_metrics,
        "metadata": {
            "tensor_dir": str(tensor_dir),
            "mode": args.mode,
            "train_rows": int(split_index),
            "valid_rows": int(len(target) - split_index),
            "valid_ratio": args.valid_ratio,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "book_weight": book_weight,
            "trade_weight": trade_weight,
            "device": args.device,
            "book_channels": int(book_tensor.shape[1]),
            "trade_channels": int(trade_tensor.shape[1]),
            "sequence_length": int(book_tensor.shape[2]),
            "target_mean": target_mean,
            "target_std": target_std,
            "book_channel_mean": book_mean.tolist(),
            "book_channel_std": book_std.tolist(),
            "trade_channel_mean": trade_mean.tolist(),
            "trade_channel_std": trade_std.tolist(),
        },
    }

    metrics_path = output_dir / f"{args.mode}_cnn_metrics.json"
    train_pred_path = output_dir / f"{args.mode}_cnn_train_predictions.csv"
    pred_path = output_dir / f"{args.mode}_cnn_valid_predictions.csv"

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    pd.DataFrame(
        {
            "stock_id": stock_id[:split_index],
            "time_id": time_id[:split_index],
            "target": y_train,
            "prediction": best_train_pred,
        }
    ).to_csv(train_pred_path, index=False)

    pd.DataFrame(
        {
            "stock_id": stock_id[split_index:],
            "time_id": time_id[split_index:],
            "target": y_valid,
            "prediction": best_valid_pred,
        }
    ).to_csv(pred_path, index=False)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Metrics saved to: {metrics_path}")
    print(f"Train predictions saved to: {train_pred_path}")
    print(f"Validation predictions saved to: {pred_path}")


if __name__ == "__main__":
    main()
