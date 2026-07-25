from __future__ import annotations

import torch
from torch import nn


class SASRec(nn.Module):
    def __init__(
        self,
        num_items: int,
        max_seq_len: int = 50,
        embedding_dim: int = 64,
        num_heads: int = 2,
        num_blocks: int = 2,
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
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)
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

    def score_all_items(self, hidden_states: torch.Tensor) -> torch.Tensor:
        item_weights = self.item_embedding.weight[1:]
        return hidden_states @ item_weights.transpose(0, 1)

    def score_candidates(
        self,
        hidden_states: torch.Tensor,
        candidate_item_ids: torch.Tensor,
    ) -> torch.Tensor:
        candidate_embeddings = self.item_embedding(candidate_item_ids)
        if hidden_states.dim() == candidate_embeddings.dim():
            return torch.sum(hidden_states * candidate_embeddings, dim=-1)
        if hidden_states.dim() + 1 == candidate_embeddings.dim():
            return torch.sum(hidden_states.unsqueeze(-2) * candidate_embeddings, dim=-1)
        raise ValueError(
            "Hidden states and candidate item ids have incompatible shapes for scoring."
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        last_hidden_state = self.get_last_hidden_state(input_ids)
        return self.score_all_items(last_hidden_state)
