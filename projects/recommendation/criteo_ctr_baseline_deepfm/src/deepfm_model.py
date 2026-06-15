import torch
from torch import nn


class DeepFM(nn.Module):
    def __init__(
        self,
        dense_feature_count: int,
        sparse_vocab_sizes: list[int],
        embedding_dim: int = 8,
        mlp_hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if mlp_hidden_dims is None:
            mlp_hidden_dims = [64, 32]

        self.dense_feature_count = dense_feature_count
        self.sparse_feature_count = len(sparse_vocab_sizes)
        self.embedding_dim = embedding_dim

        # First-order linear term for dense features.
        self.linear_dense = nn.Linear(dense_feature_count, 1)

        # First-order linear term for sparse ids.
        self.linear_sparse_embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size, 1) for vocab_size in sparse_vocab_sizes]
        )

        # Shared embeddings used by both FM and deep parts.
        self.fm_embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size, embedding_dim) for vocab_size in sparse_vocab_sizes]
        )

        deep_input_dim = dense_feature_count + self.sparse_feature_count * embedding_dim
        mlp_layers: list[nn.Module] = []
        prev_dim = deep_input_dim

        for hidden_dim in mlp_hidden_dims:
            mlp_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        self.deep_mlp = nn.Sequential(*mlp_layers)
        self.deep_output = nn.Linear(prev_dim, 1)

    def forward(self, dense_x: torch.Tensor, sparse_x: torch.Tensor) -> torch.Tensor:
        # Linear part: dense contribution + sparse first-order weights.
        linear_dense_logit = self.linear_dense(dense_x)

        sparse_linear_terms = []
        sparse_embeddings = []

        for feature_index in range(self.sparse_feature_count):
            ids = sparse_x[:, feature_index]
            sparse_linear_terms.append(
                self.linear_sparse_embeddings[feature_index](ids)
            )
            sparse_embeddings.append(self.fm_embeddings[feature_index](ids))

        linear_sparse_logit = torch.stack(sparse_linear_terms, dim=1).sum(dim=1)

        # FM second-order interaction term.
        stacked_embeddings = torch.stack(sparse_embeddings, dim=1)
        summed_embeddings = stacked_embeddings.sum(dim=1)
        square_of_sum = summed_embeddings.pow(2)
        sum_of_square = (stacked_embeddings.pow(2)).sum(dim=1)
        fm_logit = 0.5 * (square_of_sum - sum_of_square).sum(dim=1, keepdim=True)

        # Deep component learns higher-order nonlinear patterns from dense values
        # and concatenated sparse embeddings.
        deep_input = torch.cat(
            [dense_x, stacked_embeddings.reshape(dense_x.size(0), -1)], dim=1
        )
        deep_hidden = self.deep_mlp(deep_input)
        deep_logit = self.deep_output(deep_hidden)

        logits = linear_dense_logit + linear_sparse_logit + fm_logit + deep_logit
        return logits.squeeze(1)
