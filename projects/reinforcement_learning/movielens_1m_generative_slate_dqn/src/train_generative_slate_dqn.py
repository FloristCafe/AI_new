from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

CURRENT_PROJECT_DIR = Path(
    r"D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_generative_slate_dqn"
)
DEFAULT_TRAIN_SEQUENCE_PATH = Path(
    r"D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_baseline\artifacts\preprocessed\train_sequence_supervision.npz"
)
DEFAULT_VALID_SEQUENCE_PATH = Path(
    r"D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_baseline\artifacts\preprocessed\valid_sequences.npz"
)
DEFAULT_METADATA_PATH = Path(
    r"D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_baseline\artifacts\preprocessed\metadata.json"
)
DEFAULT_OUTPUT_DIR = CURRENT_PROJECT_DIR / "artifacts"
RECSIM_ENV_DIR = Path(
    r"D:\Python\Artificial Intelligence\projects\recommendation\recommender_mdp_gymnasium"
)

if str(RECSIM_ENV_DIR) not in sys.path:
    sys.path.append(str(RECSIM_ENV_DIR))

from micro_recsim_env import SlateRecSimEnv
from slate_dqn_model import DEFAULT_SASREC_CHECKPOINT_PATH, SlateDQN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a generative Slate-DQN agent online with SlateRecSimEnv."
    )
    parser.add_argument(
        "--train-sequence-path",
        type=str,
        default=str(DEFAULT_TRAIN_SEQUENCE_PATH),
        help="Path to the training sequence pool used to initialize online episodes.",
    )
    parser.add_argument(
        "--valid-sequence-path",
        type=str,
        default=str(DEFAULT_VALID_SEQUENCE_PATH),
        help="Path to the validation sequence pool used for online evaluation.",
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
        help="Directory used to save checkpoints and training metrics.",
    )
    parser.add_argument(
        "--sasrec-checkpoint-path",
        type=str,
        default=str(DEFAULT_SASREC_CHECKPOINT_PATH),
        help="Pretrained SASRec checkpoint used as the frozen user encoder.",
    )
    parser.add_argument(
        "--slate-size",
        type=int,
        default=5,
        help="Number of items generated for each slate.",
    )
    parser.add_argument(
        "--total-episodes",
        type=int,
        default=1000,
        help="Number of online episodes to run.",
    )
    parser.add_argument(
        "--max-slates-per-episode",
        type=int,
        default=10,
        help="Maximum number of slates generated before an episode terminates.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Optimizer learning rate.",
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
        default=1.0,
        help="Discount factor. Keep 1.0 for the shared slate return update.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Mini-batch size sampled from the replay buffer.",
    )
    parser.add_argument(
        "--replay-buffer-capacity",
        type=int,
        default=50000,
        help="Maximum number of micro-transitions kept in the replay buffer.",
    )
    parser.add_argument(
        "--warmup-transitions",
        type=int,
        default=512,
        help="Number of collected micro-transitions required before learning starts.",
    )
    parser.add_argument(
        "--updates-per-slate",
        type=int,
        default=1,
        help="Number of optimization steps after each generated slate.",
    )
    parser.add_argument(
        "--target-tau",
        type=float,
        default=0.01,
        help="Polyak averaging coefficient for the target network.",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm. Set 0 to disable.",
    )
    parser.add_argument(
        "--epsilon-start",
        type=float,
        default=0.2,
        help="Initial epsilon used for slate generation exploration.",
    )
    parser.add_argument(
        "--epsilon-end",
        type=float,
        default=0.02,
        help="Final epsilon after decay.",
    )
    parser.add_argument(
        "--epsilon-decay-episodes",
        type=int,
        default=800,
        help="Number of episodes used to linearly decay epsilon.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=100,
        help="Run online evaluation every N training episodes.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=100,
        help="Number of validation episodes used during online evaluation.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=20,
        help="Print training progress every N episodes.",
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


def load_metadata(metadata_path: Path) -> dict:
    with metadata_path.open("r", encoding="utf-8") as fin:
        return json.load(fin)


def load_sequence_pool(npz_path: Path) -> dict[str, np.ndarray]:
    data = np.load(npz_path)
    return {key: data[key] for key in data.files}


def build_model(metadata: dict, args: argparse.Namespace, device: torch.device) -> SlateDQN:
    model = SlateDQN(
        num_items=int(metadata["kept_item_count"]),
        max_seq_len=int(metadata["max_seq_len"]),
        slate_size=args.slate_size,
        embedding_dim=128,
        num_heads=2,
        num_blocks=3,
        dropout=0.2,
        context_dropout=0.1,
    ).to(device)
    model.load_sasrec_encoder_weights(args.sasrec_checkpoint_path, map_location=device)
    model.freeze_user_encoder()
    return model


def build_optimizer(
    model: SlateDQN,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    param_groups = model.build_optimizer_param_groups(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    return torch.optim.Adam(param_groups)


def soft_update_target_network(
    online_model: SlateDQN,
    target_model: SlateDQN,
    tau: float,
) -> None:
    with torch.no_grad():
        for target_param, online_param in zip(
            target_model.parameters(),
            online_model.parameters(),
        ):
            target_param.mul_(1.0 - tau).add_(online_param, alpha=tau)


def epsilon_by_episode(
    episode_index: int,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay_episodes: int,
) -> float:
    if epsilon_decay_episodes <= 0:
        return epsilon_end
    decay_progress = min(max(episode_index - 1, 0) / epsilon_decay_episodes, 1.0)
    return float(epsilon_start + (epsilon_end - epsilon_start) * decay_progress)


def sample_sequence_start(
    sequence_pool: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> tuple[np.ndarray, int | None]:
    sample_index = int(rng.integers(0, len(sequence_pool["input_ids"])))
    input_ids = sequence_pool["input_ids"][sample_index].astype(np.int64)
    user_ids = sequence_pool.get("user_ids")
    user_id = int(user_ids[sample_index]) if user_ids is not None else None
    return input_ids.copy(), user_id


def build_env(metadata: dict, args: argparse.Namespace) -> SlateRecSimEnv:
    return SlateRecSimEnv(
        slate_size=args.slate_size,
        num_items=int(metadata["kept_item_count"]),
        max_seq_len=int(metadata["max_seq_len"]),
        max_episode_steps=args.max_slates_per_episode,
    )


def mask_selected_items_on_device(
    q_values: torch.Tensor,
    prefix_tensor: torch.Tensor,
) -> torch.Tensor:
    masked_q_values = q_values.clone()
    selected_item_ids = prefix_tensor[prefix_tensor > 0]
    if selected_item_ids.numel() > 0:
        masked_q_values[selected_item_ids.long() - 1] = float("-inf")
    return masked_q_values


def choose_action_from_q_values_on_device(
    q_values: torch.Tensor,
    prefix_tensor: torch.Tensor,
    epsilon: float,
    rng: np.random.Generator,
    num_items: int,
) -> int:
    del num_items
    masked_q_values = mask_selected_items_on_device(q_values, prefix_tensor)
    available_action_indices = torch.nonzero(
        torch.isfinite(masked_q_values),
        as_tuple=False,
    ).squeeze(-1)
    if available_action_indices.numel() == 0:
        raise RuntimeError("No available items left to complete the slate.")

    if rng.random() < epsilon:
        sampled_offset = int(rng.integers(0, int(available_action_indices.numel())))
        return int(available_action_indices[sampled_offset].item() + 1)
    return int(torch.argmax(masked_q_values).item() + 1)


def generate_slate(
    model: SlateDQN,
    state_input_ids: np.ndarray,
    slate_size: int,
    epsilon: float,
    rng: np.random.Generator,
    device: torch.device,
    num_items: int,
) -> tuple[np.ndarray, list[dict[str, np.ndarray | int]]]:
    state_tensor = torch.from_numpy(state_input_ids).long().unsqueeze(0).to(device)
    prefix_ids = np.zeros(slate_size, dtype=np.int64)
    prefix_tensor = torch.zeros((1, slate_size), dtype=torch.long, device=device)
    micro_transitions: list[dict[str, np.ndarray | int]] = []

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for position_index in range(slate_size):
                q_values = model.get_q_values(state_tensor, prefix_tensor).squeeze(0)
                action_id = choose_action_from_q_values_on_device(
                    q_values=q_values,
                    prefix_tensor=prefix_tensor.squeeze(0),
                    epsilon=epsilon,
                    rng=rng,
                    num_items=num_items,
                )
                micro_transitions.append(
                    {
                        "state_input_ids": state_input_ids.copy(),
                        "prefix_ids": prefix_ids.copy(),
                        "action_id": int(action_id),
                        "position_index": position_index,
                    }
                )
                prefix_ids[position_index] = action_id
                prefix_tensor[0, position_index] = action_id
    finally:
        if was_training:
            model.train()
        else:
            model.eval()

    return prefix_ids.copy(), micro_transitions


def append_slate_transitions(
    replay_buffer: deque,
    micro_transitions: list[dict[str, np.ndarray | int]],
    reward: float,
    next_input_ids: np.ndarray,
    slate_size: int,
) -> None:
    terminal_prefix = np.zeros(slate_size, dtype=np.int64)
    for micro_transition in micro_transitions:
        replay_buffer.append(
            {
                "state_input_ids": micro_transition["state_input_ids"],
                "prefix_ids": micro_transition["prefix_ids"],
                "action_id": int(micro_transition["action_id"]),
                "reward": float(reward),
                "next_input_ids": next_input_ids.copy(),
                "next_prefix_ids": terminal_prefix.copy(),
                "done": 1.0,
                "position_index": int(micro_transition["position_index"]),
            }
        )


def sample_replay_batch(
    replay_buffer: deque,
    batch_size: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    batch_indices = rng.choice(len(replay_buffer), size=batch_size, replace=False)
    batch_samples = [replay_buffer[int(batch_index)] for batch_index in batch_indices]
    return {
        "state_input_ids": np.stack(
            [sample["state_input_ids"] for sample in batch_samples]
        ).astype(np.int64),
        "prefix_ids": np.stack(
            [sample["prefix_ids"] for sample in batch_samples]
        ).astype(np.int64),
        "action_ids": np.asarray(
            [sample["action_id"] for sample in batch_samples],
            dtype=np.int64,
        ),
        "rewards": np.asarray(
            [sample["reward"] for sample in batch_samples],
            dtype=np.float32,
        ),
        "next_input_ids": np.stack(
            [sample["next_input_ids"] for sample in batch_samples]
        ).astype(np.int64),
        "next_prefix_ids": np.stack(
            [sample["next_prefix_ids"] for sample in batch_samples]
        ).astype(np.int64),
        "dones": np.asarray(
            [sample["done"] for sample in batch_samples],
            dtype=np.float32,
        ),
    }


def compute_dqn_loss(
    online_model: SlateDQN,
    target_model: SlateDQN,
    batch: dict[str, np.ndarray],
    gamma: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    state_input_ids = torch.from_numpy(batch["state_input_ids"]).long().to(device)
    prefix_ids = torch.from_numpy(batch["prefix_ids"]).long().to(device)
    action_ids = torch.from_numpy(batch["action_ids"]).long().to(device)
    rewards = torch.from_numpy(batch["rewards"]).float().to(device)
    next_input_ids = torch.from_numpy(batch["next_input_ids"]).long().to(device)
    next_prefix_ids = torch.from_numpy(batch["next_prefix_ids"]).long().to(device)
    dones = torch.from_numpy(batch["dones"]).float().to(device)

    current_q = online_model.gather_q_values(
        input_ids=state_input_ids,
        prefix_ids=prefix_ids,
        action_ids=action_ids,
    )
    with torch.no_grad():
        next_q_values = target_model.get_q_values(next_input_ids, next_prefix_ids)
        next_q = next_q_values.max(dim=1).values
        target_q = rewards + gamma * (1.0 - dones) * next_q

    td_loss = F.smooth_l1_loss(current_q, target_q)
    return {
        "loss": td_loss,
        "mean_current_q": current_q.mean(),
        "mean_target_q": target_q.mean(),
        "mean_reward": rewards.mean(),
    }


def run_online_evaluation(
    model: SlateDQN,
    env: SlateRecSimEnv,
    valid_sequence_pool: dict[str, np.ndarray],
    eval_episodes: int,
    slate_size: int,
    max_slates_per_episode: int,
    device: torch.device,
    rng: np.random.Generator,
    num_items: int,
) -> dict[str, float]:
    total_episode_reward = 0.0
    total_slate_count = 0
    total_click_count = 0.0

    for eval_episode_index in range(eval_episodes):
        input_ids, user_id = sample_sequence_start(valid_sequence_pool, rng)
        observation, _ = env.reset(
            seed=int(rng.integers(0, 1_000_000_000)),
            options={
                "input_ids": input_ids,
                "user_id": user_id,
            },
        )
        episode_reward = 0.0
        slate_count = 0
        terminated = False
        truncated = False

        while not terminated and not truncated and slate_count < max_slates_per_episode:
            slate_action, _ = generate_slate(
                model=model,
                state_input_ids=observation["input_ids"],
                slate_size=slate_size,
                epsilon=0.0,
                rng=rng,
                device=device,
                num_items=num_items,
            )
            observation, reward, terminated, truncated, info = env.step(slate_action)
            episode_reward += float(reward)
            total_click_count += float(info["total_clicks"])
            slate_count += 1

        total_episode_reward += episode_reward
        total_slate_count += slate_count

    safe_episode_count = max(eval_episodes, 1)
    safe_slate_count = max(total_slate_count, 1)
    return {
        "eval_mean_episode_reward": total_episode_reward / safe_episode_count,
        "eval_mean_slate_reward": total_episode_reward / safe_slate_count,
        "eval_mean_clicks_per_slate": total_click_count / safe_slate_count,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    rng = np.random.default_rng(args.seed)

    train_sequence_path = Path(args.train_sequence_path)
    valid_sequence_path = Path(args.valid_sequence_path)
    metadata_path = Path(args.metadata_path)
    output_dir = Path(args.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    metrics_dir = output_dir / "metrics"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(metadata_path)
    train_sequence_pool = load_sequence_pool(train_sequence_path)
    valid_sequence_pool = load_sequence_pool(valid_sequence_path)

    online_model = build_model(metadata, args, device)
    target_model = copy.deepcopy(online_model).to(device)
    target_model.eval()
    for parameter in target_model.parameters():
        parameter.requires_grad = False

    optimizer = build_optimizer(
        model=online_model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_env = build_env(metadata, args)
    eval_env = build_env(metadata, args)

    replay_buffer: deque = deque(maxlen=args.replay_buffer_capacity)
    best_eval_mean_slate_reward = float("-inf")
    best_episode = 0
    episode_metrics: list[dict[str, float]] = []

    best_checkpoint_path = checkpoints_dir / "slate_dqn_best.pt"
    final_checkpoint_path = checkpoints_dir / "slate_dqn_final.pt"

    for episode_index in range(1, args.total_episodes + 1):
        epsilon = epsilon_by_episode(
            episode_index=episode_index,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay_episodes=args.epsilon_decay_episodes,
        )
        start_input_ids, user_id = sample_sequence_start(train_sequence_pool, rng)
        observation, _ = train_env.reset(
            seed=int(rng.integers(0, 1_000_000_000)),
            options={
                "input_ids": start_input_ids,
                "user_id": user_id,
            },
        )

        episode_reward = 0.0
        slate_count = 0
        total_clicks = 0.0
        optimization_steps = 0
        last_loss = float("nan")
        last_mean_current_q = float("nan")
        last_mean_target_q = float("nan")
        terminated = False
        truncated = False

        while not terminated and not truncated and slate_count < args.max_slates_per_episode:
            state_input_ids = observation["input_ids"].copy()
            slate_action, micro_transitions = generate_slate(
                model=online_model,
                state_input_ids=state_input_ids,
                slate_size=args.slate_size,
                epsilon=epsilon,
                rng=rng,
                device=device,
                num_items=int(metadata["kept_item_count"]),
            )
            next_observation, reward, terminated, truncated, info = train_env.step(
                slate_action
            )
            append_slate_transitions(
                replay_buffer=replay_buffer,
                micro_transitions=micro_transitions,
                reward=float(reward),
                next_input_ids=next_observation["input_ids"],
                slate_size=args.slate_size,
            )

            episode_reward += float(reward)
            total_clicks += float(info["total_clicks"])
            slate_count += 1
            observation = next_observation

            if len(replay_buffer) >= max(args.warmup_transitions, args.batch_size):
                for _ in range(args.updates_per_slate):
                    online_model.train()
                    batch = sample_replay_batch(
                        replay_buffer=replay_buffer,
                        batch_size=args.batch_size,
                        rng=rng,
                    )
                    loss_dict = compute_dqn_loss(
                        online_model=online_model,
                        target_model=target_model,
                        batch=batch,
                        gamma=args.gamma,
                        device=device,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss_dict["loss"].backward()
                    if args.grad_clip_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(
                            online_model.parameters(),
                            max_norm=args.grad_clip_norm,
                        )
                    optimizer.step()
                    soft_update_target_network(
                        online_model=online_model,
                        target_model=target_model,
                        tau=args.target_tau,
                    )

                    optimization_steps += 1
                    last_loss = float(loss_dict["loss"].item())
                    last_mean_current_q = float(loss_dict["mean_current_q"].item())
                    last_mean_target_q = float(loss_dict["mean_target_q"].item())

        episode_summary = {
            "episode": episode_index,
            "epsilon": float(epsilon),
            "episode_reward": float(episode_reward),
            "slate_count": float(slate_count),
            "mean_slate_reward": float(episode_reward / max(slate_count, 1)),
            "mean_clicks_per_slate": float(total_clicks / max(slate_count, 1)),
            "replay_buffer_size": float(len(replay_buffer)),
            "optimization_steps": float(optimization_steps),
            "last_loss": float(last_loss),
            "last_mean_current_q": float(last_mean_current_q),
            "last_mean_target_q": float(last_mean_target_q),
        }

        if episode_index % args.eval_interval == 0 or episode_index == args.total_episodes:
            eval_metrics = run_online_evaluation(
                model=online_model,
                env=eval_env,
                valid_sequence_pool=valid_sequence_pool,
                eval_episodes=args.eval_episodes,
                slate_size=args.slate_size,
                max_slates_per_episode=args.max_slates_per_episode,
                device=device,
                rng=rng,
                num_items=int(metadata["kept_item_count"]),
            )
            episode_summary.update(eval_metrics)

            current_eval_reward = float(eval_metrics["eval_mean_slate_reward"])
            if current_eval_reward > best_eval_mean_slate_reward:
                best_eval_mean_slate_reward = current_eval_reward
                best_episode = episode_index
                torch.save(online_model.state_dict(), best_checkpoint_path)

        episode_metrics.append(episode_summary)

        if episode_index % args.log_interval == 0 or episode_index == 1:
            recent_metrics = episode_metrics[-args.log_interval :]
            recent_mean_slate_reward = float(
                np.mean([metric["mean_slate_reward"] for metric in recent_metrics])
            )
            print(
                f"Episode {episode_index}/{args.total_episodes} - "
                f"epsilon={epsilon:.4f} - "
                f"mean_slate_reward={episode_summary['mean_slate_reward']:.4f} - "
                f"recent_mean_slate_reward={recent_mean_slate_reward:.4f} - "
                f"replay_buffer={len(replay_buffer)} - "
                f"last_loss={last_loss:.6f}"
            )

    torch.save(online_model.state_dict(), final_checkpoint_path)

    training_summary = {
        "train_sequence_path": str(train_sequence_path),
        "valid_sequence_path": str(valid_sequence_path),
        "metadata_path": str(metadata_path),
        "output_dir": str(output_dir),
        "sasrec_checkpoint_path": str(Path(args.sasrec_checkpoint_path)),
        "device": str(device),
        "seed": args.seed,
        "num_items": int(metadata["kept_item_count"]),
        "max_seq_len": int(metadata["max_seq_len"]),
        "slate_size": args.slate_size,
        "total_episodes": args.total_episodes,
        "max_slates_per_episode": args.max_slates_per_episode,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gamma": args.gamma,
        "batch_size": args.batch_size,
        "replay_buffer_capacity": args.replay_buffer_capacity,
        "warmup_transitions": args.warmup_transitions,
        "updates_per_slate": args.updates_per_slate,
        "target_tau": args.target_tau,
        "grad_clip_norm": args.grad_clip_norm,
        "epsilon_start": args.epsilon_start,
        "epsilon_end": args.epsilon_end,
        "epsilon_decay_episodes": args.epsilon_decay_episodes,
        "eval_interval": args.eval_interval,
        "eval_episodes": args.eval_episodes,
        "best_episode": best_episode,
        "best_eval_mean_slate_reward": best_eval_mean_slate_reward,
        "best_checkpoint_path": str(best_checkpoint_path),
        "final_checkpoint_path": str(final_checkpoint_path),
        "episode_metrics": episode_metrics,
    }

    metrics_path = metrics_dir / "training_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fout:
        json.dump(training_summary, fout, ensure_ascii=False, indent=2)

    print("Generative Slate-DQN training finished.")
    print(
        f"Best episode: {best_episode} - "
        f"eval_mean_slate_reward={best_eval_mean_slate_reward:.6f}"
    )
    print(f"Best checkpoint saved to: {best_checkpoint_path}")
    print(f"Final checkpoint saved to: {final_checkpoint_path}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
