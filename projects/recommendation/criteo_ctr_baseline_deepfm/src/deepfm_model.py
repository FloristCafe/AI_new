import torch
from torch import nn


class DeepFM(nn.Module):
    def __init__(
        self,
        dense_feature_count: int,
        dense_bucket_vocab_sizes: list[int],
        sparse_vocab_sizes: list[int],
        embedding_dim: int = 8,
        mlp_hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        global_bias_init: float = 0.0,
        learnable_global_bias: bool = False,
        use_fm: bool = True,
        use_deep: bool = True,
        fm_embedding_init_std: float = 0.01,
        fm_scale: float = 0.1,
    ) -> None:
        super().__init__()

        if mlp_hidden_dims is None:
            mlp_hidden_dims = [64, 32]

        self.dense_feature_count = dense_feature_count
        self.dense_bucket_feature_count = len(dense_bucket_vocab_sizes)
        self.sparse_feature_count = len(sparse_vocab_sizes)
        self.embedding_dim = embedding_dim
        self.use_fm = use_fm
        self.use_deep = use_deep
        self.fm_scale = fm_scale

        # Keep dense floats in the linear branch so the model can still learn
        # a direct first-order response from original continuous values.
        self.linear_dense = nn.Linear(dense_feature_count, 1)

        # First-order linear term for sparse ids.
        self.linear_sparse_embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size, 1) for vocab_size in sparse_vocab_sizes]
        )

        # Shared sparse embeddings used by both FM and deep parts.
        self.fm_embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size, embedding_dim) for vocab_size in sparse_vocab_sizes]
        )

        # Dense bucket ids are embedded before entering the deep branch so the
        # dense and sparse signals live in the same representation space.
        self.dense_bucket_embeddings = nn.ModuleList(
            [
                nn.Embedding(vocab_size, embedding_dim)
                for vocab_size in dense_bucket_vocab_sizes
            ]
        )

        deep_input_dim = (
            self.dense_bucket_feature_count + self.sparse_feature_count
        ) * embedding_dim
        mlp_layers: list[nn.Module] = []
        prev_dim = deep_input_dim

        for hidden_dim in mlp_hidden_dims:
            mlp_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        self.deep_mlp = nn.Sequential(*mlp_layers)
        self.deep_output = nn.Linear(prev_dim, 1)

        global_bias_tensor = torch.tensor([global_bias_init], dtype=torch.float32)
        if learnable_global_bias:
            self.global_bias = nn.Parameter(global_bias_tensor)
        else:
            self.register_buffer("global_bias", global_bias_tensor)

        self._init_fm_embeddings(fm_embedding_init_std)
        self._init_dense_bucket_embeddings(fm_embedding_init_std)

    def _init_fm_embeddings(self, init_std: float) -> None:
        for embedding in self.fm_embeddings:
            nn.init.normal_(embedding.weight, mean=0.0, std=init_std)

    def _init_dense_bucket_embeddings(self, init_std: float) -> None:
        for embedding in self.dense_bucket_embeddings:
            nn.init.normal_(embedding.weight, mean=0.0, std=init_std)

    def forward(
        self,
        dense_x: torch.Tensor,
        dense_bucket_x: torch.Tensor,
        sparse_x: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
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
        linear_logit = linear_dense_logit + linear_sparse_logit

        stacked_sparse_embeddings = torch.stack(sparse_embeddings, dim=1)
        if self.use_fm:
            summed_embeddings = stacked_sparse_embeddings.sum(dim=1)
            square_of_sum = summed_embeddings.pow(2)
            sum_of_square = (stacked_sparse_embeddings.pow(2)).sum(dim=1)
            fm_logit = (
                0.5 * (square_of_sum - sum_of_square).sum(dim=1, keepdim=True)
            ) * self.fm_scale
        else:
            fm_logit = torch.zeros_like(linear_dense_logit)

        if self.use_deep:
            dense_bucket_embeddings = []
            for feature_index in range(self.dense_bucket_feature_count):
                ids = dense_bucket_x[:, feature_index]
                dense_bucket_embeddings.append(
                    self.dense_bucket_embeddings[feature_index](ids)
                )

            stacked_dense_bucket_embeddings = torch.stack(
                dense_bucket_embeddings, dim=1
            )
            deep_input = torch.cat(
                [
                    stacked_dense_bucket_embeddings.reshape(dense_x.size(0), -1),
                    stacked_sparse_embeddings.reshape(dense_x.size(0), -1),
                ],
                dim=1,
            )
            deep_hidden = self.deep_mlp(deep_input)
            deep_logit = self.deep_output(deep_hidden)
        else:
            deep_logit = torch.zeros_like(linear_dense_logit)

        logits = (
            linear_logit
            + fm_logit
            + deep_logit
            + self.global_bias.view(1, 1)
        )
        final_logits = logits.squeeze(1)

        if return_components:
            components = {
                "linear_dense_logit": linear_dense_logit.squeeze(1),
                "linear_sparse_logit": linear_sparse_logit.squeeze(1),
                "linear_logit": linear_logit.squeeze(1),
                "fm_logit": fm_logit.squeeze(1),
                "deep_logit": deep_logit.squeeze(1),
                "global_bias": self.global_bias.expand_as(final_logits),
            }
            return final_logits, components

        return final_logits
