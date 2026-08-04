from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sasrec_dqn_model import DEFAULT_SASREC_CHECKPOINT_PATH, SASRecDQN


DEFAULT_OFFLINE_BUFFER_DIR = Path(
    r"D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_dqn_offline_rl\artifacts\offline_buffer"
)
DEFAULT_OUTPUT_DIR = Path(
    r"D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_dqn_offline_rl\artifacts"
)


class OfflineReplayDataset(Dataset):
    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        self.states = torch.from_numpy(states).long()
        self.actions = torch.from_numpy(actions).long()
        self.rewards = torch.from_numpy(rewards).float()
        self.next_states = torch.from_numpy(next_states).long()
        self.dones = torch.from_numpy(dones.astype(np.float32)).float()

    def __len__(self) -> int:
        return int(self.actions.shape[0])

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.states[index],
            self.actions[index],
            self.rewards[index],
            self.next_states[index],
            self.dones[index],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SASRec-DQN with offline Double DQN and CQL regularization."
    )
    parser.add_argument(
        "--offline-buffer-dir",
        type=str,
        default=str(DEFAULT_OFFLINE_BUFFER_DIR),
        help="Directory containing offline replay buffer artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for checkpoints and metrics.",
    )
    parser.add_argument(
        "--sasrec-checkpoint-path",
        type=str,
        default=str(DEFAULT_SASREC_CHECKPOINT_PATH),
        help="Path to the pretrained SASRec checkpoint used to initialize the encoder.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of offline training epochs.",
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
        "--q-head-learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate for the DQN head.",
    )
    parser.add_argument(
        "--encoder-learning-rate",
        type=float,
        default=1e-6,
        help="Learning rate for the SASRec encoder. Set 0 to freeze the encoder.",
    )
    parser.add_argument(
        "--encoder-warmup-epochs",
        type=int,
        default=0,
        help="Freeze the pretrained SASRec encoder for the first N epochs, then unfreeze it.",
    )
    parser.add_argument(
        "--q-head-init-std",
        type=float,
        default=0.01,
        help="Std used to reinitialize the Q-head weights around zero. Set 0 to keep PyTorch default init.",
    )
    parser.add_argument(
        "--q-head-init-mean",
        type=float,
        default=0.0,
        help="Mean used when reinitializing the Q-head weights.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Optimizer weight decay.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        help="Discount factor.",
    )
    parser.add_argument(
        "--cql-alpha",
        type=float,
        default=1.0,
        help="Initial weight of the CQL conservative penalty.",
    )
    parser.add_argument(
        "--ce-regularization-weight",
        type=float,
        default=0.1,
        help="Weight of the supervised regularization term added on top of the RL loss.",
    )
    parser.add_argument(
        "--supervised-regularizer",
        type=str,
        default="ce",
        choices=["ce", "bpr", "none"],
        help="Supervised regularizer used alongside RL loss.",
    )
    parser.add_argument(
        "--bpr-negative-count",
        type=int,
        default=1,
        help="Number of sampled negatives per positive when supervised_regularizer=bpr.",
    )
    parser.add_argument(
        "--target-tau",
        type=float,
        default=0.005,
        help="Polyak soft-update coefficient for the target network.",
    )
    parser.add_argument(
        "--adaptive-cql-alpha",
        dest="adaptive_cql_alpha",
        action="store_true",
        help="Adapt CQL alpha online so the weighted CQL penalty stays close to the TD scale.",
    )
    parser.add_argument(
        "--disable-adaptive-cql-alpha",
        dest="adaptive_cql_alpha",
        action="store_false",
        help="Disable online CQL alpha adaptation.",
    )
    parser.set_defaults(adaptive_cql_alpha=True)
    parser.add_argument(
        "--target-cql-ratio",
        type=float,
        default=1.1,
        help="Target ratio for weighted CQL penalty divided by TD loss.",
    )
    parser.add_argument(
        "--alpha-adaptation-rate",
        type=float,
        default=0.05,
        help="Step size used when adapting CQL alpha.",
    )
    parser.add_argument(
        "--min-cql-alpha",
        type=float,
        default=0.05,
        help="Lower bound for adaptive CQL alpha.",
    )
    parser.add_argument(
        "--max-cql-alpha",
        type=float,
        default=5.0,
        help="Upper bound for adaptive CQL alpha.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Print training diagnostics every N optimization steps.",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm. Set 0 to disable clipping.",
    )
    parser.add_argument(
        "--q-warning-threshold",
        type=float,
        default=100.0,
        help="Print a warning when batch max Q-value exceeds this threshold. Set 0 to disable warnings.",
    )
    parser.add_argument(
        "--q-stop-threshold",
        type=float,
        default=0.0,
        help="Stop training early when batch max Q-value exceeds this threshold. Set 0 to disable hard stopping.",
    )
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="valid_ndcg_at_10",
        choices=[
            "valid_total_loss",
            "valid_td_loss",
            "valid_cql_penalty",
            "valid_hr_at_10",
            "valid_ndcg_at_10",
        ],
        help="Validation metric used to pick the best checkpoint.",
    )
    parser.add_argument(
        "--ranking-topk",
        type=int,
        default=10,
        help="Top-k used for validation ranking metrics.",
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


def load_offline_buffer_metadata(offline_buffer_dir: Path) -> dict:
    metadata_path = offline_buffer_dir / "offline_buffer_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Offline buffer metadata not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as fin:
        return json.load(fin)


def load_replay_dataset(offline_buffer_dir: Path, split: str) -> OfflineReplayDataset:
    split_path = offline_buffer_dir / f"{split}_replay_buffer.npz"
    if not split_path.exists():
        raise FileNotFoundError(f"Replay buffer split not found: {split_path}")

    data = np.load(split_path)
    return OfflineReplayDataset(
        states=data["states"],
        actions=data["actions"],
        rewards=data["rewards"],
        next_states=data["next_states"],
        dones=data["dones"],
    )


def build_model(metadata: dict, args: argparse.Namespace, device: torch.device) -> SASRecDQN:
    model = SASRecDQN(
        num_items=int(metadata["kept_item_count"]),
        max_seq_len=int(metadata["max_seq_len"]),
        embedding_dim=128,
        num_heads=2,
        num_blocks=3,
        dropout=0.2,
    ).to(device)
    model.load_sasrec_encoder_weights(args.sasrec_checkpoint_path, map_location=device)
    if args.q_head_init_std > 0.0:
        model.reset_q_head_parameters(
            init_std=args.q_head_init_std,
            init_mean=args.q_head_init_mean,
        )
    if args.encoder_learning_rate <= 0.0:
        model.freeze_encoder()
    else:
        model.unfreeze_encoder()
    return model


def build_optimizer(model: SASRecDQN, args: argparse.Namespace) -> torch.optim.Optimizer:
    param_groups = model.build_optimizer_param_groups(
        q_head_learning_rate=args.q_head_learning_rate,
        weight_decay=args.weight_decay,
        encoder_learning_rate=args.encoder_learning_rate,
    )
    return torch.optim.Adam(param_groups)


def set_encoder_optimization_state(
    model: SASRecDQN,
    optimizer: torch.optim.Optimizer,
    should_train_encoder: bool,
    encoder_learning_rate: float,
) -> None:
    if should_train_encoder:
        model.unfreeze_encoder()
    else:
        model.freeze_encoder()

    for param_group in optimizer.param_groups:
        if param_group.get("group_name") == "encoder":
            param_group["lr"] = encoder_learning_rate if should_train_encoder else 0.0


def soft_update_target_network(
    online_model: SASRecDQN,
    target_model: SASRecDQN,
    tau: float,
) -> None:
    with torch.no_grad():
        for target_param, online_param in zip(
            target_model.parameters(),
            online_model.parameters(),
        ):
            target_param.mul_(1.0 - tau).add_(online_param, alpha=tau)


def adapt_cql_alpha(
    current_alpha: float,
    td_loss: torch.Tensor,
    cql_penalty: torch.Tensor,
    target_ratio: float,
    adaptation_rate: float,
    min_alpha: float,
    max_alpha: float,
) -> float:
    td_scale = max(float(td_loss.detach().item()), 1e-8)
    cql_scale = max(float(cql_penalty.detach().item()), 1e-8)
    effective_ratio = (current_alpha * cql_scale) / td_scale
    updated_alpha = current_alpha * np.exp(adaptation_rate * (target_ratio - effective_ratio))
    return float(np.clip(updated_alpha, min_alpha, max_alpha))


def sample_negative_items(
    positive_actions: torch.Tensor,
    num_items: int,
    negative_count: int,
) -> torch.Tensor:
    negative_items = torch.randint(
        low=1,
        high=num_items + 1,
        size=(positive_actions.size(0), negative_count),
        device=positive_actions.device,
    )
    positive_actions = positive_actions.unsqueeze(1)
    collision_mask = negative_items.eq(positive_actions)

    while collision_mask.any():
        negative_items[collision_mask] = torch.randint(
            low=1,
            high=num_items + 1,
            size=(int(collision_mask.sum().item()),),
            device=positive_actions.device,
        )
        collision_mask = negative_items.eq(positive_actions)

    return negative_items


def compute_supervised_regularization_loss(
    q_values: torch.Tensor,
    actions: torch.Tensor,
    regularizer: str,
    num_items: int,
    bpr_negative_count: int,
) -> dict[str, torch.Tensor]:
    zero = q_values.new_zeros(())
    if regularizer == "none":
        return {
            "supervised_loss": zero,
            "ce_loss": zero,
            "bpr_loss": zero,
        }

    if regularizer == "ce":
        ce_loss = F.cross_entropy(q_values, actions.long() - 1)
        return {
            "supervised_loss": ce_loss,
            "ce_loss": ce_loss,
            "bpr_loss": zero,
        }

    if regularizer == "bpr":
        positive_indices = (actions.long() - 1).unsqueeze(1)
        positive_q = q_values.gather(1, positive_indices)
        negative_items = sample_negative_items(
            positive_actions=actions.long(),
            num_items=num_items,
            negative_count=bpr_negative_count,
        )
        negative_q = q_values.gather(1, negative_items - 1)
        bpr_loss = -F.logsigmoid(positive_q - negative_q).mean()
        return {
            "supervised_loss": bpr_loss,
            "ce_loss": zero,
            "bpr_loss": bpr_loss,
        }

    raise ValueError(f"Unsupported supervised regularizer: {regularizer}")


def compute_batch_losses(
    online_model: SASRecDQN,
    target_model: SASRecDQN,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    gamma: float,
    cql_alpha: float,
    supervised_regularizer: str,
    ce_regularization_weight: float,
    bpr_negative_count: int,
    num_items: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    states, actions, rewards, next_states, dones = batch
    states = states.to(device)
    actions = actions.to(device)
    rewards = rewards.to(device)
    next_states = next_states.to(device)
    dones = dones.to(device)

    q_values = online_model.get_q_values(states)
    current_q = q_values.gather(1, (actions.long() - 1).unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        online_next_q_values = online_model.get_q_values(next_states)
        best_next_action_indices = online_next_q_values.argmax(dim=1, keepdim=True)
        target_next_q_values = target_model.get_q_values(next_states)
        next_q = target_next_q_values.gather(dim=1, index=best_next_action_indices).squeeze(1)
        target_q = rewards + gamma * (1.0 - dones) * next_q

    td_loss = F.smooth_l1_loss(current_q, target_q)
    cql_logsumexp = torch.logsumexp(q_values, dim=1)
    cql_penalty = (cql_logsumexp - current_q).mean()
    rl_loss = td_loss + cql_alpha * cql_penalty
    supervised_losses = compute_supervised_regularization_loss(
        q_values=q_values,
        actions=actions,
        regularizer=supervised_regularizer,
        num_items=num_items,
        bpr_negative_count=bpr_negative_count,
    )
    total_loss = rl_loss + ce_regularization_weight * supervised_losses["supervised_loss"]
    ratio_denominator = td_loss.detach().abs().clamp_min(1e-8)
    raw_cql_to_td_ratio = cql_penalty.detach() / ratio_denominator
    effective_cql_to_td_ratio = (cql_alpha * cql_penalty.detach()) / ratio_denominator

    return {
        "total_loss": total_loss,
        "rl_loss": rl_loss,
        "td_loss": td_loss,
        "cql_penalty": cql_penalty,
        "supervised_loss": supervised_losses["supervised_loss"],
        "ce_loss": supervised_losses["ce_loss"],
        "bpr_loss": supervised_losses["bpr_loss"],
        "raw_cql_to_td_ratio": raw_cql_to_td_ratio,
        "effective_cql_to_td_ratio": effective_cql_to_td_ratio,
        "mean_q_value": q_values.mean(),
        "max_q_value": q_values.max(),
        "mean_current_q": current_q.mean(),
        "mean_target_q": target_q.mean(),
    }


def mask_seen_items_for_ranking(
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

    target_indices = target_actions.long() - 1
    batch_indices = torch.arange(states.size(0), device=states.device)
    history_mask[batch_indices, target_indices] = False
    return masked_q_values.masked_fill(history_mask, float("-inf"))


def compute_hit_and_ndcg_at_k(
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


def run_validation(
    online_model: SASRecDQN,
    target_model: SASRecDQN,
    dataloader: DataLoader,
    gamma: float,
    cql_alpha: float,
    supervised_regularizer: str,
    ce_regularization_weight: float,
    bpr_negative_count: int,
    device: torch.device,
    ranking_topk: int,
    num_items: int,
) -> dict[str, float]:
    total_examples = 0
    metric_sums = {
        "valid_total_loss": 0.0,
        "valid_rl_loss": 0.0,
        "valid_td_loss": 0.0,
        "valid_cql_penalty": 0.0,
        "valid_supervised_loss": 0.0,
        "valid_ce_loss": 0.0,
        "valid_bpr_loss": 0.0,
        "valid_hr_at_10": 0.0,
        "valid_ndcg_at_10": 0.0,
        "valid_raw_cql_to_td_ratio": 0.0,
        "valid_effective_cql_to_td_ratio": 0.0,
        "valid_mean_q_value": 0.0,
        "valid_max_q_value": 0.0,
        "valid_mean_current_q": 0.0,
        "valid_mean_target_q": 0.0,
    }

    online_model.eval()
    target_model.eval()
    with torch.no_grad():
        for batch in dataloader:
            losses = compute_batch_losses(
                online_model=online_model,
                target_model=target_model,
                batch=batch,
                gamma=gamma,
                cql_alpha=cql_alpha,
                supervised_regularizer=supervised_regularizer,
                ce_regularization_weight=ce_regularization_weight,
                bpr_negative_count=bpr_negative_count,
                num_items=num_items,
                device=device,
            )
            states, actions, _, _, _ = batch
            states = states.to(device)
            actions = actions.to(device)
            q_values = online_model.get_q_values(states)
            masked_q_values = mask_seen_items_for_ranking(q_values, states, actions)
            topk_indices = torch.topk(
                masked_q_values,
                k=min(ranking_topk, num_items),
                dim=1,
            ).indices
            target_indices = actions.long() - 1
            hr_batch, ndcg_batch = compute_hit_and_ndcg_at_k(topk_indices, target_indices)

            batch_size = batch[0].size(0)
            total_examples += batch_size
            metric_sums["valid_total_loss"] += float(losses["total_loss"].item()) * batch_size
            metric_sums["valid_rl_loss"] += float(losses["rl_loss"].item()) * batch_size
            metric_sums["valid_td_loss"] += float(losses["td_loss"].item()) * batch_size
            metric_sums["valid_cql_penalty"] += float(losses["cql_penalty"].item()) * batch_size
            metric_sums["valid_supervised_loss"] += float(losses["supervised_loss"].item()) * batch_size
            metric_sums["valid_ce_loss"] += float(losses["ce_loss"].item()) * batch_size
            metric_sums["valid_bpr_loss"] += float(losses["bpr_loss"].item()) * batch_size
            metric_sums["valid_hr_at_10"] += hr_batch * batch_size
            metric_sums["valid_ndcg_at_10"] += ndcg_batch * batch_size
            metric_sums["valid_raw_cql_to_td_ratio"] += float(losses["raw_cql_to_td_ratio"].item()) * batch_size
            metric_sums["valid_effective_cql_to_td_ratio"] += float(
                losses["effective_cql_to_td_ratio"].item()
            ) * batch_size
            metric_sums["valid_mean_q_value"] += float(losses["mean_q_value"].item()) * batch_size
            metric_sums["valid_max_q_value"] += float(losses["max_q_value"].item()) * batch_size
            metric_sums["valid_mean_current_q"] += float(losses["mean_current_q"].item()) * batch_size
            metric_sums["valid_mean_target_q"] += float(losses["mean_target_q"].item()) * batch_size

    return {
        key: value / max(total_examples, 1)
        for key, value in metric_sums.items()
    }


def should_improve(metric_name: str, current_value: float, best_value: float) -> bool:
    if metric_name in {"valid_total_loss", "valid_td_loss", "valid_cql_penalty"}:
        return current_value < best_value
    if metric_name in {"valid_hr_at_10", "valid_ndcg_at_10"}:
        return current_value > best_value
    raise ValueError(f"Unsupported selection metric: {metric_name}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    offline_buffer_dir = Path(args.offline_buffer_dir)
    output_dir = Path(args.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    metrics_dir = output_dir / "metrics"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_offline_buffer_metadata(offline_buffer_dir)
    train_dataset = load_replay_dataset(offline_buffer_dir, split="train")
    valid_dataset = load_replay_dataset(offline_buffer_dir, split="valid")

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
    online_model = build_model(metadata, args, device)
    target_model = copy.deepcopy(online_model).to(device)
    target_model.eval()
    for parameter in target_model.parameters():
        parameter.requires_grad = False

    optimizer = build_optimizer(online_model, args)
    current_cql_alpha = float(args.cql_alpha)

    encoder_currently_trainable = args.encoder_learning_rate > 0.0
    if args.encoder_learning_rate > 0.0 and args.encoder_warmup_epochs > 0:
        encoder_currently_trainable = False
        set_encoder_optimization_state(
            model=online_model,
            optimizer=optimizer,
            should_train_encoder=False,
            encoder_learning_rate=args.encoder_learning_rate,
        )

    if args.selection_metric in {"valid_hr_at_10", "valid_ndcg_at_10"}:
        best_metric_value = float("-inf")
    else:
        best_metric_value = float("inf")
    best_epoch = 0
    global_step = 0
    epoch_metrics: list[dict[str, float]] = []
    stopped_early = False
    stop_reason = ""

    best_checkpoint_path = checkpoints_dir / "sasrec_dqn_best.pt"
    final_checkpoint_path = checkpoints_dir / "sasrec_dqn_final.pt"

    for epoch_idx in range(1, args.epochs + 1):
        should_train_encoder_this_epoch = (
            args.encoder_learning_rate > 0.0
            and epoch_idx > args.encoder_warmup_epochs
        )
        if should_train_encoder_this_epoch != encoder_currently_trainable:
            set_encoder_optimization_state(
                model=online_model,
                optimizer=optimizer,
                should_train_encoder=should_train_encoder_this_epoch,
                encoder_learning_rate=args.encoder_learning_rate,
            )
            encoder_currently_trainable = should_train_encoder_this_epoch
            if should_train_encoder_this_epoch:
                print(
                    f"Epoch {epoch_idx}: encoder warmup finished, unfreezing encoder "
                    f"with learning rate {args.encoder_learning_rate:.6g}."
                )
            else:
                print(f"Epoch {epoch_idx}: encoder kept frozen for warmup.")

        online_model.train()
        epoch_example_count = 0
        epoch_sums = {
            "train_total_loss": 0.0,
            "train_rl_loss": 0.0,
            "train_td_loss": 0.0,
            "train_cql_penalty": 0.0,
            "train_supervised_loss": 0.0,
            "train_ce_loss": 0.0,
            "train_bpr_loss": 0.0,
            "train_raw_cql_to_td_ratio": 0.0,
            "train_effective_cql_to_td_ratio": 0.0,
            "train_mean_q_value": 0.0,
            "train_max_q_value": 0.0,
            "train_mean_current_q": 0.0,
            "train_mean_target_q": 0.0,
            "train_grad_norm": 0.0,
            "train_cql_alpha": 0.0,
            "train_encoder_learning_rate": 0.0,
        }
        should_break_training = False

        for batch in train_dataloader:
            losses = compute_batch_losses(
                online_model=online_model,
                target_model=target_model,
                batch=batch,
                gamma=args.gamma,
                cql_alpha=current_cql_alpha,
                supervised_regularizer=args.supervised_regularizer,
                ce_regularization_weight=args.ce_regularization_weight,
                bpr_negative_count=args.bpr_negative_count,
                num_items=int(metadata["kept_item_count"]),
                device=device,
            )

            finite_values = {
                key: torch.isfinite(value).all().item()
                for key, value in losses.items()
            }
            if not all(finite_values.values()):
                stopped_early = True
                stop_reason = (
                    f"Non-finite values detected at epoch {epoch_idx}, "
                    f"step {global_step + 1}: {finite_values}"
                )
                print(f"Stopping early. {stop_reason}")
                should_break_training = True
                break

            optimizer.zero_grad(set_to_none=True)
            losses["total_loss"].backward()
            if args.grad_clip_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    online_model.parameters(),
                    max_norm=args.grad_clip_norm,
                )
            else:
                grad_norm = torch.zeros((), device=device)
            optimizer.step()

            batch_size = batch[0].size(0)
            epoch_example_count += batch_size
            epoch_sums["train_total_loss"] += float(losses["total_loss"].item()) * batch_size
            epoch_sums["train_rl_loss"] += float(losses["rl_loss"].item()) * batch_size
            epoch_sums["train_td_loss"] += float(losses["td_loss"].item()) * batch_size
            epoch_sums["train_cql_penalty"] += float(losses["cql_penalty"].item()) * batch_size
            epoch_sums["train_supervised_loss"] += float(losses["supervised_loss"].item()) * batch_size
            epoch_sums["train_ce_loss"] += float(losses["ce_loss"].item()) * batch_size
            epoch_sums["train_bpr_loss"] += float(losses["bpr_loss"].item()) * batch_size
            epoch_sums["train_raw_cql_to_td_ratio"] += float(losses["raw_cql_to_td_ratio"].item()) * batch_size
            epoch_sums["train_effective_cql_to_td_ratio"] += float(
                losses["effective_cql_to_td_ratio"].item()
            ) * batch_size
            epoch_sums["train_mean_q_value"] += float(losses["mean_q_value"].item()) * batch_size
            epoch_sums["train_max_q_value"] += float(losses["max_q_value"].item()) * batch_size
            epoch_sums["train_mean_current_q"] += float(losses["mean_current_q"].item()) * batch_size
            epoch_sums["train_mean_target_q"] += float(losses["mean_target_q"].item()) * batch_size
            epoch_sums["train_grad_norm"] += float(grad_norm.item()) * batch_size
            epoch_sums["train_cql_alpha"] += current_cql_alpha * batch_size
            epoch_sums["train_encoder_learning_rate"] += (
                args.encoder_learning_rate if encoder_currently_trainable else 0.0
            ) * batch_size

            global_step += 1
            soft_update_target_network(
                online_model=online_model,
                target_model=target_model,
                tau=args.target_tau,
            )

            if args.adaptive_cql_alpha:
                current_cql_alpha = adapt_cql_alpha(
                    current_alpha=current_cql_alpha,
                    td_loss=losses["td_loss"],
                    cql_penalty=losses["cql_penalty"],
                    target_ratio=args.target_cql_ratio,
                    adaptation_rate=args.alpha_adaptation_rate,
                    min_alpha=args.min_cql_alpha,
                    max_alpha=args.max_cql_alpha,
                )

            current_max_q_value = float(losses["max_q_value"].item())
            if args.q_warning_threshold > 0 and current_max_q_value > args.q_warning_threshold:
                print(
                    f"Q warning at step {global_step}: "
                    f"max_q={current_max_q_value:.6f} exceeded "
                    f"warning threshold {args.q_warning_threshold:.6f}"
                )

            if args.q_stop_threshold > 0 and current_max_q_value > args.q_stop_threshold:
                stopped_early = True
                stop_reason = (
                    f"Batch max Q-value {current_max_q_value:.6f} exceeded "
                    f"stop threshold {args.q_stop_threshold:.6f} at "
                    f"epoch {epoch_idx}, step {global_step}"
                )
                print(f"Stopping early. {stop_reason}")
                should_break_training = True
                break

            if global_step % args.log_interval == 0:
                print(
                    f"Step {global_step} - "
                    f"total_loss={losses['total_loss'].item():.6f} - "
                    f"rl_loss={losses['rl_loss'].item():.6f} - "
                    f"td_loss={losses['td_loss'].item():.6f} - "
                    f"cql_penalty={losses['cql_penalty'].item():.6f} - "
                    f"supervised_loss={losses['supervised_loss'].item():.6f} - "
                    f"ce_loss={losses['ce_loss'].item():.6f} - "
                    f"bpr_loss={losses['bpr_loss'].item():.6f} - "
                    f"effective_cql_to_td_ratio={losses['effective_cql_to_td_ratio'].item():.6f} - "
                    f"cql_alpha={current_cql_alpha:.6f} - "
                    f"mean_q={losses['mean_q_value'].item():.6f} - "
                    f"max_q={losses['max_q_value'].item():.6f} - "
                    f"grad_norm={float(grad_norm.item()):.6f}"
                )

        if should_break_training:
            break

        train_metrics = {
            key: value / max(epoch_example_count, 1)
            for key, value in epoch_sums.items()
        }
        valid_metrics = run_validation(
            online_model=online_model,
            target_model=target_model,
            dataloader=valid_dataloader,
            gamma=args.gamma,
            cql_alpha=current_cql_alpha,
            supervised_regularizer=args.supervised_regularizer,
            ce_regularization_weight=args.ce_regularization_weight,
            bpr_negative_count=args.bpr_negative_count,
            device=device,
            ranking_topk=args.ranking_topk,
            num_items=int(metadata["kept_item_count"]),
        )

        if should_break_training:
            break

        selection_metric_value = float(valid_metrics[args.selection_metric])
        is_best_epoch = should_improve(
            args.selection_metric,
            selection_metric_value,
            best_metric_value,
        )
        if is_best_epoch:
            best_metric_value = selection_metric_value
            best_epoch = epoch_idx
            torch.save(online_model.state_dict(), best_checkpoint_path)

        epoch_summary = {
            "epoch": epoch_idx,
            **{key: float(value) for key, value in train_metrics.items()},
            **{key: float(value) for key, value in valid_metrics.items()},
            "selection_metric": args.selection_metric,
            "selection_metric_value": selection_metric_value,
            "is_best_epoch": is_best_epoch,
        }
        epoch_metrics.append(epoch_summary)

        print(
            f"Epoch {epoch_idx}/{args.epochs} - "
            f"train_total_loss={epoch_summary['train_total_loss']:.6f} - "
            f"train_rl_loss={epoch_summary['train_rl_loss']:.6f} - "
            f"train_td_loss={epoch_summary['train_td_loss']:.6f} - "
            f"train_cql_penalty={epoch_summary['train_cql_penalty']:.6f} - "
            f"train_supervised_loss={epoch_summary['train_supervised_loss']:.6f} - "
            f"train_ce_loss={epoch_summary['train_ce_loss']:.6f} - "
            f"train_bpr_loss={epoch_summary['train_bpr_loss']:.6f} - "
            f"train_effective_cql_to_td_ratio={epoch_summary['train_effective_cql_to_td_ratio']:.6f} - "
            f"train_cql_alpha={epoch_summary['train_cql_alpha']:.6f} - "
            f"train_encoder_lr={epoch_summary['train_encoder_learning_rate']:.6f} - "
            f"train_grad_norm={epoch_summary['train_grad_norm']:.6f} - "
            f"valid_total_loss={epoch_summary['valid_total_loss']:.6f} - "
            f"valid_rl_loss={epoch_summary['valid_rl_loss']:.6f} - "
            f"valid_td_loss={epoch_summary['valid_td_loss']:.6f} - "
            f"valid_cql_penalty={epoch_summary['valid_cql_penalty']:.6f} - "
            f"valid_supervised_loss={epoch_summary['valid_supervised_loss']:.6f} - "
            f"valid_ce_loss={epoch_summary['valid_ce_loss']:.6f} - "
            f"valid_bpr_loss={epoch_summary['valid_bpr_loss']:.6f} - "
            f"valid_hr@{args.ranking_topk}={epoch_summary['valid_hr_at_10']:.6f} - "
            f"valid_ndcg@{args.ranking_topk}={epoch_summary['valid_ndcg_at_10']:.6f} - "
            f"valid_effective_cql_to_td_ratio={epoch_summary['valid_effective_cql_to_td_ratio']:.6f} - "
            f"valid_mean_q={epoch_summary['valid_mean_q_value']:.6f} - "
            f"valid_max_q={epoch_summary['valid_max_q_value']:.6f}"
        )

    torch.save(online_model.state_dict(), final_checkpoint_path)

    training_summary = {
        "offline_buffer_dir": str(offline_buffer_dir),
        "device": str(device),
        "seed": args.seed,
        "epochs_completed": len(epoch_metrics),
        "epochs_requested": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "q_head_learning_rate": args.q_head_learning_rate,
        "encoder_learning_rate": args.encoder_learning_rate,
        "encoder_warmup_epochs": args.encoder_warmup_epochs,
        "q_head_init_std": args.q_head_init_std,
        "q_head_init_mean": args.q_head_init_mean,
        "weight_decay": args.weight_decay,
        "gamma": args.gamma,
        "td_loss_type": "smooth_l1_huber",
        "cql_alpha_initial": args.cql_alpha,
        "cql_alpha_final": current_cql_alpha,
        "supervised_regularizer": args.supervised_regularizer,
        "ce_regularization_weight": args.ce_regularization_weight,
        "bpr_negative_count": args.bpr_negative_count,
        "adaptive_cql_alpha": args.adaptive_cql_alpha,
        "target_cql_ratio": args.target_cql_ratio,
        "alpha_adaptation_rate": args.alpha_adaptation_rate,
        "min_cql_alpha": args.min_cql_alpha,
        "max_cql_alpha": args.max_cql_alpha,
        "target_tau": args.target_tau,
        "log_interval": args.log_interval,
        "grad_clip_norm": args.grad_clip_norm,
        "q_warning_threshold": args.q_warning_threshold,
        "q_stop_threshold": args.q_stop_threshold,
        "selection_metric": args.selection_metric,
        "ranking_topk": args.ranking_topk,
        "num_items": int(metadata["kept_item_count"]),
        "max_seq_len": int(metadata["max_seq_len"]),
        "train_transition_count": len(train_dataset),
        "valid_transition_count": len(valid_dataset),
        "encoder_frozen": args.encoder_learning_rate <= 0.0,
        "encoder_finally_trainable": encoder_currently_trainable,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "sasrec_checkpoint_path": str(Path(args.sasrec_checkpoint_path)),
        "best_epoch": best_epoch,
        "best_selection_metric_value": best_metric_value,
        "best_checkpoint_path": str(best_checkpoint_path),
        "final_checkpoint_path": str(final_checkpoint_path),
        "epoch_metrics": epoch_metrics,
    }

    metrics_path = metrics_dir / "training_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fout:
        json.dump(training_summary, fout, ensure_ascii=False, indent=2)

    print("Offline CQL-DQN training finished.")
    print(
        f"Best epoch: {best_epoch} - "
        f"{args.selection_metric}={best_metric_value:.6f}"
    )
    print(f"Best checkpoint saved to: {best_checkpoint_path}")
    print(f"Final checkpoint saved to: {final_checkpoint_path}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
