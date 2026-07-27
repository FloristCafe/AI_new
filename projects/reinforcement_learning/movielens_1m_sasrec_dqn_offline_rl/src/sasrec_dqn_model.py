from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


DEFAULT_SASREC_CHECKPOINT_PATH = Path(
    r"D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_baseline\artifacts\experiments\seq50_dim128_blocks3_drop02_ce\checkpoints\sasrec_best.pt"
)

# Some PyTorch builds produce NaNs in eval-mode MultiheadAttention fastpath
# for left-padded sequence batches on CPU. Disable the fastpath globally so
# target-network and validation passes remain numerically stable.
if hasattr(torch.backends, "mha"):
    torch.backends.mha.set_fastpath_enabled(False)


class SASRecEncoder(nn.Module):
    def __init__(
        self,
        num_items: int,
        max_seq_len: int = 50,
        embedding_dim: int = 128,
        num_heads: int = 2,
        num_blocks: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim

        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        self.input_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_blocks,
            enable_nested_tensor=False,
        )
        self.output_layer_norm = nn.LayerNorm(embedding_dim)

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds configured max_seq_len {self.max_seq_len}."
            )

        positions = (
            torch.arange(seq_len, device=input_ids.device)
            .unsqueeze(0)
            .expand(batch_size, seq_len)
        )
        padding_mask = input_ids.eq(0)

        hidden_states = self.item_embedding(input_ids) + self.position_embedding(positions)
        hidden_states = self.input_dropout(hidden_states)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        hidden_states = self.encoder(
            hidden_states,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )
        hidden_states = self.output_layer_norm(hidden_states)
        return hidden_states

    def get_last_hidden_state(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.encode(input_ids)
        non_padding_counts = input_ids.ne(0).sum(dim=1).clamp(min=1)
        last_positions = non_padding_counts - 1
        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
        return hidden_states[batch_indices, last_positions]


class SASRecDQN(nn.Module):
    def __init__(
        self,
        num_items: int,
        max_seq_len: int = 50,
        embedding_dim: int = 128,
        num_heads: int = 2,
        num_blocks: int = 3,
        dropout: float = 0.2,
        encoder_learning_rate_scale: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.encoder_learning_rate_scale = encoder_learning_rate_scale

        self.encoder = SASRecEncoder(
            num_items=num_items,
            max_seq_len=max_seq_len,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            dropout=dropout,
        )
        self.q_head = nn.Linear(embedding_dim, num_items)

    def get_state_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.encoder.get_last_hidden_state(input_ids)

    def get_q_values(self, input_ids: torch.Tensor) -> torch.Tensor:
        state_embedding = self.get_state_embedding(input_ids)
        return self.q_head(state_embedding)

    def gather_q_values(
        self,
        input_ids: torch.Tensor,
        action_ids: torch.Tensor,
    ) -> torch.Tensor:
        q_values = self.get_q_values(input_ids)
        if action_ids.dim() == 1:
            action_ids = action_ids.unsqueeze(-1)
        action_indices = action_ids.long() - 1
        return q_values.gather(dim=1, index=action_indices).squeeze(-1)

    def freeze_encoder(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = True

    def load_sasrec_encoder_weights(
        self,
        checkpoint_path: Path | str = DEFAULT_SASREC_CHECKPOINT_PATH,
        strict: bool = True,
        map_location: str | torch.device = "cpu",
    ) -> None:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"SASRec checkpoint not found: {checkpoint_path}")

        state_dict = torch.load(checkpoint_path, map_location=map_location)
        remapped_state_dict: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("encoder."):
                remapped_key = key.replace("encoder.", "encoder.encoder.", 1)
            elif key == "item_embedding.weight":
                remapped_key = "encoder.item_embedding.weight"
            elif key == "position_embedding.weight":
                remapped_key = "encoder.position_embedding.weight"
            elif key.startswith("output_layer_norm."):
                remapped_key = key.replace("output_layer_norm.", "encoder.output_layer_norm.", 1)
            else:
                continue
            remapped_state_dict[remapped_key] = value

        try:
            missing_keys, unexpected_keys = self.load_state_dict(
                remapped_state_dict,
                strict=False,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to load SASRec encoder weights due to a shape mismatch. "
                "Most likely the checkpoint architecture does not match the "
                "current SASRec-DQN encoder configuration."
            ) from exc
        if strict:
            filtered_missing = [
                key for key in missing_keys if not key.startswith("q_head.")
            ]
            if filtered_missing or unexpected_keys:
                raise RuntimeError(
                    "Failed to strictly load SASRec encoder weights. "
                    f"Missing keys: {filtered_missing}; unexpected keys: {unexpected_keys}"
                )

    def build_optimizer_param_groups(
        self,
        q_head_learning_rate: float,
        weight_decay: float = 0.0,
        encoder_learning_rate: float | None = None,
    ) -> list[dict[str, object]]:
        if encoder_learning_rate is None:
            encoder_learning_rate = q_head_learning_rate * self.encoder_learning_rate_scale

        parameter_groups: list[dict[str, object]] = [
            {
                "params": list(self.q_head.parameters()),
                "lr": q_head_learning_rate,
                "weight_decay": weight_decay,
            }
        ]

        encoder_parameters = [
            parameter for parameter in self.encoder.parameters() if parameter.requires_grad
        ]
        if encoder_parameters:
            parameter_groups.append(
                {
                    "params": encoder_parameters,
                    "lr": encoder_learning_rate,
                    "weight_decay": weight_decay,
                }
            )
        return parameter_groups

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.get_q_values(input_ids)
