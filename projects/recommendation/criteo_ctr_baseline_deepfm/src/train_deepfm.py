import argparse
import json
import math
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from deepfm_model import DeepFM


class CriteoDeepFMDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        dense_cols: list[str],
        sparse_cols: list[str],
        label_col: str = "label",
    ) -> None:
        self.dense_x = torch.tensor(df[dense_cols].values, dtype=torch.float32)
        self.sparse_x = torch.tensor(df[sparse_cols].values, dtype=torch.long)
        self.labels = torch.tensor(df[label_col].values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.dense_x[index], self.sparse_x[index], self.labels[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a minimal DeepFM baseline on preprocessed Criteo data."
    )
    parser.add_argument(
        "--train-path",
        type=str,
        default=r"D:\Python\Artificial Intelligence\projects\recommendation\criteo_ctr_baseline_deepfm\artifacts\train_deepfm.parquet",
        help="Path to the DeepFM-ready training parquet.",
    )
    parser.add_argument(
        "--valid-path",
        type=str,
        default=r"D:\Python\Artificial Intelligence\projects\recommendation\criteo_ctr_baseline_deepfm\artifacts\valid_deepfm.parquet",
        help="Path to the DeepFM-ready validation parquet.",
    )
    parser.add_argument(
        "--feature-config",
        type=str,
        default=r"D:\Python\Artificial Intelligence\projects\recommendation\criteo_ctr_baseline_deepfm\artifacts\feature_config.json",
        help="Path to the feature config json.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=r"D:\Python\Artificial Intelligence\projects\recommendation\criteo_ctr_baseline_deepfm\artifacts\deepfm_run",
        help="Directory for model and metrics outputs.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fm-embedding-init-std",
        type=float,
        default=0.01,
        help="Initialization std for FM embeddings.",
    )
    parser.add_argument(
        "--fm-scale",
        type=float,
        default=0.1,
        help="Scaling factor applied to the FM logit before summation.",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Max norm for gradient clipping. Use 0 or negative to disable.",
    )
    parser.add_argument(
        "--disable-fm",
        action="store_true",
        help="Disable the FM branch for ablation debugging.",
    )
    parser.add_argument(
        "--disable-deep",
        action="store_true",
        help="Disable the deep branch for ablation debugging.",
    )
    parser.add_argument(
        "--learn-global-bias",
        action="store_true",
        help="Allow the global prior bias to keep training instead of staying fixed.",
    )
    parser.add_argument(
        "--debug-logits",
        action="store_true",
        help="Print min/max/mean logits during evaluation for numerical debugging.",
    )
    parser.add_argument(
        "--one-batch-overfit",
        action="store_true",
        help="Use only one training batch repeatedly to test whether the model can overfit a tiny batch.",
    )
    parser.add_argument(
        "--one-batch-steps",
        type=int,
        default=200,
        help="Number of repeated optimization steps in one-batch-overfit mode.",
    )
    parser.add_argument(
        "--save-best-by",
        type=str,
        default="roc_auc",
        choices=["roc_auc", "pr_auc", "log_loss"],
        help="Validation metric used to save the best checkpoint.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_feature_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_prior_bias(train_df: pd.DataFrame, label_col: str = "label") -> float:
    ctr = float(train_df[label_col].mean())
    ctr = min(max(ctr, 1e-6), 1 - 1e-6)
    return math.log(ctr / (1.0 - ctr))


def evaluate_model(
    model: DeepFM,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    debug_logits: bool = False,
) -> tuple[dict[str, float], list[float], list[float]]:
    model.eval()
    total_loss = 0.0
    all_probs: list[float] = []
    all_labels: list[float] = []
    all_logits: list[float] = []

    with torch.no_grad():
        for dense_x, sparse_x, labels in dataloader:
            dense_x = dense_x.to(device)
            sparse_x = sparse_x.to(device)
            labels = labels.to(device)

            if debug_logits:
                logits, components = model(
                    dense_x, sparse_x, return_components=True
                )
            else:
                logits = model(dense_x, sparse_x)
            loss = loss_fn(logits, labels)
            probs = torch.sigmoid(logits)

            total_loss += loss.item() * labels.size(0)
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_logits.extend(logits.cpu().tolist())

    avg_loss = total_loss / len(dataloader.dataset)
    if debug_logits and all_logits:
        logits_tensor = torch.tensor(all_logits, dtype=torch.float32)
        print(
            "Eval logits stats | "
            f"min={logits_tensor.min().item():.6f} | "
            f"max={logits_tensor.max().item():.6f} | "
            f"mean={logits_tensor.mean().item():.6f}"
        )
        for name, values in components.items():
            values_tensor = values.detach().cpu().float()
            print(
                f"{name} stats | "
                f"min={values_tensor.min().item():.6f} | "
                f"max={values_tensor.max().item():.6f} | "
                f"mean={values_tensor.mean().item():.6f}"
            )

    metrics = {
        "roc_auc": float(roc_auc_score(all_labels, all_probs)),
        "pr_auc": float(average_precision_score(all_labels, all_probs)),
        "log_loss": float(log_loss(all_labels, all_probs, labels=[0, 1])),
        "bce_loss": float(avg_loss),
    }
    return metrics, all_probs, all_labels


def run_one_batch_overfit(
    model: DeepFM,
    train_loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    steps: int,
) -> list[dict[str, float]]:
    dense_x, sparse_x, labels = next(iter(train_loader))
    dense_x = dense_x.to(device)
    sparse_x = sparse_x.to(device)
    labels = labels.to(device)
    history: list[dict[str, float]] = []

    print(
        "One-batch mode | "
        f"dense_shape={tuple(dense_x.shape)} | "
        f"sparse_shape={tuple(sparse_x.shape)} | "
        f"label_mean={labels.float().mean().item():.6f}"
    )

    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(dense_x, sparse_x)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        if step == 1 or step % 20 == 0 or step == steps:
            probs = torch.sigmoid(logits)
            logits_detached = logits.detach().cpu()
            labels_cpu = labels.detach().cpu().tolist()
            probs_cpu = probs.detach().cpu().tolist()
            metrics = {
                "step": step,
                "loss": float(loss.item()),
                "logit_min": float(logits_detached.min().item()),
                "logit_max": float(logits_detached.max().item()),
                "logit_mean": float(logits_detached.mean().item()),
                "roc_auc": float(roc_auc_score(labels_cpu, probs_cpu)),
                "pr_auc": float(average_precision_score(labels_cpu, probs_cpu)),
            }
            history.append(metrics)
            print(
                f"One-batch step {step}/{steps} | "
                f"loss={metrics['loss']:.6f} | "
                f"logit_min={metrics['logit_min']:.6f} | "
                f"logit_max={metrics['logit_max']:.6f} | "
                f"logit_mean={metrics['logit_mean']:.6f} | "
                f"roc_auc={metrics['roc_auc']:.6f} | "
                f"pr_auc={metrics['pr_auc']:.6f}"
            )

    return history


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    train_path = Path(args.train_path)
    valid_path = Path(args.valid_path)
    feature_config_path = Path(args.feature_config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(train_path)
    valid_df = pd.read_parquet(valid_path)
    feature_config = load_feature_config(feature_config_path)

    dense_cols = feature_config["dense_features"]
    sparse_cols = feature_config["sparse_features"]
    sparse_vocab_sizes = [
        feature_config["sparse_vocab_sizes"][col] for col in sparse_cols
    ]

    train_dataset = CriteoDeepFMDataset(train_df, dense_cols, sparse_cols)
    valid_dataset = CriteoDeepFMDataset(valid_df, dense_cols, sparse_cols)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepFM(
        dense_feature_count=len(dense_cols),
        sparse_vocab_sizes=sparse_vocab_sizes,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        global_bias_init=compute_prior_bias(train_df),
        learnable_global_bias=args.learn_global_bias,
        use_fm=not args.disable_fm,
        use_deep=not args.disable_deep,
        fm_embedding_init_std=args.fm_embedding_init_std,
        fm_scale=args.fm_scale,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.BCEWithLogitsLoss()

    if args.one_batch_overfit:
        one_batch_history = run_one_batch_overfit(
            model=model,
            train_loader=train_loader,
            device=device,
            optimizer=optimizer,
            loss_fn=loss_fn,
            steps=args.one_batch_steps,
        )
        debug_payload = {
            "mode": "one_batch_overfit",
            "history": one_batch_history,
            "training_config": {
                "batch_size": args.batch_size,
                "one_batch_steps": args.one_batch_steps,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "embedding_dim": args.embedding_dim,
                "dropout": args.dropout,
                "learn_global_bias": args.learn_global_bias,
                "use_fm": not args.disable_fm,
                "use_deep": not args.disable_deep,
                "fm_embedding_init_std": args.fm_embedding_init_std,
                "fm_scale": args.fm_scale,
                "device": str(device),
            },
        }
        debug_metrics_path = output_dir / "one_batch_debug.json"
        with debug_metrics_path.open("w", encoding="utf-8") as f:
            json.dump(debug_payload, f, ensure_ascii=False, indent=2)
        torch.save(model.state_dict(), output_dir / "deepfm_one_batch.pt")
        print(f"One-batch debug saved to: {debug_metrics_path}")
        return

    history: list[dict] = []
    best_metric_value: float | None = None
    best_epoch_record: dict | None = None
    best_model_path = output_dir / "deepfm_model_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0

        for dense_x, sparse_x, labels in train_loader:
            dense_x = dense_x.to(device)
            sparse_x = sparse_x.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(dense_x, sparse_x)
            loss = loss_fn(logits, labels)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=args.grad_clip
                )
            optimizer.step()

            total_train_loss += loss.item() * labels.size(0)

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        valid_metrics, valid_probs, valid_labels = evaluate_model(
            model, valid_loader, device, loss_fn, debug_logits=args.debug_logits
        )
        train_metrics, _, _ = evaluate_model(
            model, train_loader, device, loss_fn, debug_logits=args.debug_logits
        )

        epoch_record = {
            "epoch": epoch,
            "train_bce_loss": float(avg_train_loss),
            "train_metrics": train_metrics,
            "valid_metrics": valid_metrics,
        }
        history.append(epoch_record)

        current_metric = valid_metrics[args.save_best_by]
        if args.save_best_by == "log_loss":
            is_better = best_metric_value is None or current_metric < best_metric_value
        else:
            is_better = best_metric_value is None or current_metric > best_metric_value

        if is_better:
            best_metric_value = current_metric
            best_epoch_record = epoch_record
            torch.save(model.state_dict(), best_model_path)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={avg_train_loss:.6f} | "
            f"valid_roc_auc={valid_metrics['roc_auc']:.6f} | "
            f"valid_pr_auc={valid_metrics['pr_auc']:.6f} | "
            f"valid_log_loss={valid_metrics['log_loss']:.6f}"
        )

    metrics = {
        "final_train": history[-1]["train_metrics"],
        "final_valid": history[-1]["valid_metrics"],
        "training_config": {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "embedding_dim": args.embedding_dim,
            "dropout": args.dropout,
            "learn_global_bias": args.learn_global_bias,
            "use_fm": not args.disable_fm,
            "use_deep": not args.disable_deep,
            "fm_embedding_init_std": args.fm_embedding_init_std,
            "fm_scale": args.fm_scale,
            "grad_clip": args.grad_clip,
            "save_best_by": args.save_best_by,
            "device": str(device),
        },
        "history": history,
        "best_epoch": best_epoch_record,
    }

    model_path = output_dir / "deepfm_model.pt"
    metrics_path = output_dir / "metrics.json"
    valid_pred_path = output_dir / "valid_predictions.parquet"

    torch.save(model.state_dict(), model_path)

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    valid_predictions = pd.DataFrame(
        {"label": valid_labels, "prediction": valid_probs}
    )
    valid_predictions.to_parquet(valid_pred_path, index=False)

    print("DeepFM training finished.")
    print(f"Model saved to: {model_path}")
    print(f"Best model saved to: {best_model_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Validation predictions saved to: {valid_pred_path}")


if __name__ == "__main__":
    main()
