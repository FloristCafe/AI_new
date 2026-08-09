from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


DEFAULT_SASREC_CHECKPOINT_PATH = Path(
    r"D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_baseline\artifacts\experiments\seq50_dim128_blocks3_drop02_ce\checkpoints\sasrec_best.pt"
)

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
        return self.output_layer_norm(hidden_states)

    def get_last_hidden_state(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.encode(input_ids)
        non_padding_counts = input_ids.ne(0).sum(dim=1).clamp(min=1)
        last_positions = non_padding_counts - 1
        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
        return hidden_states[batch_indices, last_positions]


class SlateContextEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )

    def forward(
        self,
        prefix_embeddings: torch.Tensor,
        prefix_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = prefix_ids.size(0)
        prefix_lengths = prefix_ids.ne(0).sum(dim=1)
        context_vector = prefix_embeddings.new_zeros((batch_size, self.gru.hidden_size))
        if int(prefix_lengths.max().item()) == 0:
            return context_vector

        encoded_prefix, _ = self.gru(prefix_embeddings)
        valid_batch_mask = prefix_lengths > 0
        valid_batch_indices = torch.nonzero(valid_batch_mask, as_tuple=False).squeeze(-1)
        last_positions = prefix_lengths[valid_batch_mask] - 1
        context_vector[valid_batch_indices] = encoded_prefix[
            valid_batch_indices,
            last_positions,
        ]
        return context_vector


class SlateDQN(nn.Module):
    def __init__(
        self,
        num_items: int,
        max_seq_len: int = 50,
        slate_size: int = 5,
        embedding_dim: int = 128,
        num_heads: int = 2,
        num_blocks: int = 3,
        dropout: float = 0.2,
        context_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.slate_size = slate_size
        self.embedding_dim = embedding_dim

        self.user_encoder = SASRecEncoder(
            num_items=num_items,
            max_seq_len=max_seq_len,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            dropout=dropout,
        )
        self.slate_context_encoder = SlateContextEncoder(
            embedding_dim=embedding_dim,
            hidden_dim=embedding_dim,
            num_layers=1,
            dropout=context_dropout,
        )
        self.q_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, num_items),
        )

    def encode_user_state(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.user_encoder.get_last_hidden_state(input_ids)

    def encode_slate_prefix(self, prefix_ids: torch.Tensor) -> torch.Tensor:
        prefix_embeddings = self.user_encoder.item_embedding(prefix_ids)
        return self.slate_context_encoder(prefix_embeddings, prefix_ids)

    def get_q_values(
        self,
        input_ids: torch.Tensor,
        prefix_ids: torch.Tensor,
    ) -> torch.Tensor:
        user_state = self.encode_user_state(input_ids)
        slate_context = self.encode_slate_prefix(prefix_ids)
        fused_state = torch.cat([user_state, slate_context], dim=-1)
        return self.q_head(fused_state)

    def gather_q_values(
        self,
        input_ids: torch.Tensor,
        prefix_ids: torch.Tensor,
        action_ids: torch.Tensor,
    ) -> torch.Tensor:
        q_values = self.get_q_values(input_ids, prefix_ids)
        if action_ids.dim() == 1:
            action_ids = action_ids.unsqueeze(-1)
        action_indices = action_ids.long() - 1
        return q_values.gather(dim=1, index=action_indices).squeeze(-1)

    def freeze_user_encoder(self) -> None:
        for parameter in self.user_encoder.parameters():
            parameter.requires_grad = False

    def unfreeze_user_encoder(self) -> None:
        for parameter in self.user_encoder.parameters():
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
                remapped_key = key.replace("encoder.", "user_encoder.encoder.", 1)
            elif key == "item_embedding.weight":
                remapped_key = "user_encoder.item_embedding.weight"
            elif key == "position_embedding.weight":
                remapped_key = "user_encoder.position_embedding.weight"
            elif key.startswith("output_layer_norm."):
                remapped_key = key.replace(
                    "output_layer_norm.",
                    "user_encoder.output_layer_norm.",
                    1,
                )
            else:
                continue
            remapped_state_dict[remapped_key] = value

        missing_keys, unexpected_keys = self.load_state_dict(
            remapped_state_dict,
            strict=False,
        )
        if strict:
            filtered_missing = [
                key
                for key in missing_keys
                if not key.startswith("q_head.")
                and not key.startswith("slate_context_encoder.")
            ]
            if filtered_missing or unexpected_keys:
                raise RuntimeError(
                    "Failed to strictly load SASRec encoder weights. "
                    f"Missing keys: {filtered_missing}; unexpected keys: {unexpected_keys}"
                )

    def build_optimizer_param_groups(
        self,
        learning_rate: float,
        weight_decay: float = 0.0,
    ) -> list[dict[str, object]]:
        trainable_parameters = [
            parameter for parameter in self.parameters() if parameter.requires_grad
        ]
        return [
            {
                "group_name": "slate_dqn",
                "params": trainable_parameters,
                "lr": learning_rate,
                "weight_decay": weight_decay,
            }
        ]

    def forward(
        self,
        input_ids: torch.Tensor,
        prefix_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.get_q_values(input_ids, prefix_ids)
