from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class DQNBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity: int, observation_dim: int):
        self.capacity = capacity
        self.observation_dim = observation_dim
        self.observations = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.position = 0

    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        self.observations[self.position] = observation
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_observations[self.position] = next_observation
        self.dones[self.position] = float(done)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
        device: torch.device,
    ) -> DQNBatch:
        indices = rng.integers(self.size, size=batch_size)
        return DQNBatch(
            observations=torch.as_tensor(
                self.observations[indices],
                dtype=torch.float32,
                device=device,
            ),
            actions=torch.as_tensor(
                self.actions[indices],
                dtype=torch.int64,
                device=device,
            ),
            rewards=torch.as_tensor(
                self.rewards[indices],
                dtype=torch.float32,
                device=device,
            ),
            next_observations=torch.as_tensor(
                self.next_observations[indices],
                dtype=torch.float32,
                device=device,
            ),
            dones=torch.as_tensor(
                self.dones[indices],
                dtype=torch.float32,
                device=device,
            ),
        )

    def __len__(self) -> int:
        return self.size


class QNetwork(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        activation: str = "relu",
    ):
        super().__init__()

        if activation == "relu":
            activation_layer = nn.ReLU
        elif activation == "silu":
            activation_layer = nn.SiLU
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.net = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            activation_layer(),
            nn.Linear(hidden_dim, hidden_dim),
            activation_layer(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


class DQNAgent:
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int,
        activation: str,
        learning_rate: float,
        gamma: float,
        batch_size: int,
        target_sync_interval: int,
        replay_buffer_capacity: int,
        device: torch.device,
        seed: int,
    ):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_sync_interval = target_sync_interval
        self.device = device
        self.rng = np.random.default_rng(seed)

        self.q_network = QNetwork(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            activation=activation,
        ).to(device)
        self.target_network = QNetwork(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            activation=activation,
        ).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=learning_rate,
        )
        self.loss_fn = nn.SmoothL1Loss()
        self.replay_buffer = ReplayBuffer(
            capacity=replay_buffer_capacity,
            observation_dim=observation_dim,
        )
        self.update_count = 0

    def select_action(self, observation: np.ndarray, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.action_dim))
        return self.greedy_action(observation)

    def greedy_action(self, observation: np.ndarray) -> int:
        with torch.no_grad():
            observation_tensor = torch.as_tensor(
                observation,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            q_values = self.q_network(observation_tensor).squeeze(0)
            return int(torch.argmax(q_values).item())

    def store_transition(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        self.replay_buffer.add(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            done=done,
        )

    def can_update(self, min_buffer_size: int) -> bool:
        return len(self.replay_buffer) >= max(min_buffer_size, self.batch_size)

    def update(self) -> dict[str, float]:
        batch = self.replay_buffer.sample(
            batch_size=self.batch_size,
            rng=self.rng,
            device=self.device,
        )

        current_q_values = self.q_network(batch.observations)
        current_action_values = current_q_values.gather(
            1,
            batch.actions.unsqueeze(1),
        ).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_network(batch.next_observations)
            best_next_values = next_q_values.max(dim=1).values
            targets = batch.rewards + self.gamma * (1.0 - batch.dones) * best_next_values

        loss = self.loss_fn(current_action_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.target_sync_interval == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return {
            "loss": float(loss.item()),
            "mean_q_value": float(current_action_values.mean().item()),
            "mean_target": float(targets.mean().item()),
        }

    def state_dict(self) -> dict:
        return {
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_count": self.update_count,
        }
