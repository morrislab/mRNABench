"""Pooling over the sequence dimension for fine-tuning and analysis."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def no_pool(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Return token embeddings unpooled, with the sequence axis at -2.

    Fills the agg_fn slot of EmbeddingModel.embed() to recover per-token
    embeddings rather than a pooled vector, normalizing the wrapper
    layouts (including Borzoi/Enformer's channels-first) so the hidden
    axis is always last.

    Embed one sequence per call: wrappers stack their per-sequence
    results, which requires equal token counts across the batch.

    Args:
        x: Token embeddings with the sequence axis at dim.
        dim: Sequence axis that pooling would have reduced.

    Returns:
        Token embeddings with the sequence axis at -2.
    """
    return torch.movedim(x, dim, -2)


class DotProductPooler(nn.Module):
    """Softmax-weighted token average using a single learned query.

    Scores each token by its dot product with one learned query and
    returns the weighted average of the token embeddings themselves.
    Holds hidden_dim parameters, with no key, value or output
    projection, so it can be compared against mean pooling without a
    capacity confound: the pooler is smaller than the head it feeds.

    Reduces to mean pooling when the query is zero, so any gain over
    mean pooling comes from learned token weighting alone.

    Fills the agg_fn slot of EmbeddingModel.embed() like torch.mean.
    """

    def __init__(self, hidden_dim: int):
        """Initialize DotProductPooler.

        Args:
            hidden_dim: Backbone embedding dimension.
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.query = nn.Parameter(torch.empty(hidden_dim))
        nn.init.normal_(self.query, std=hidden_dim ** -0.5)

    def forward(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Pool token embeddings over the sequence axis.

        Args:
            x: Token embeddings with the sequence axis at dim.
            dim: Sequence axis to pool over.

        Returns:
            Pooled embeddings with the sequence axis removed.

        Raises:
            ValueError: If the hidden axis does not match hidden_dim.
        """
        x = torch.movedim(x, dim, -2)

        if x.shape[-1] != self.hidden_dim:
            raise ValueError(
                "Expected hidden dim {} at axis -1 after pooling axis {}, "
                "got shape {}.".format(self.hidden_dim, dim, tuple(x.shape))
            )

        x = x.to(self.query.dtype)

        scores = (x @ self.query) / math.sqrt(self.hidden_dim)
        weights = F.softmax(scores, dim=-1)

        return (weights.unsqueeze(-1) * x).sum(dim=-2)


class AttentionPooler(nn.Module):
    """Multi-head attention pooling with a single learnable query.

    Substitutes for torch.mean as the agg_fn threaded into
    EmbeddingModel.embed(). Model wrappers call agg_fn(x, dim=k) with k
    the sequence axis, so forward() matches that signature and needs no
    changes in any model wrapper.

    Instead of weighting every token equally, a learned query attends
    over the tokens and emits a convex combination of their values,
    letting the head see position-specific signal (e.g. UTR motifs) that
    mean pooling averages away.

    Only usable for fine-tuning: the parameters are trained jointly with
    the LoRA adapters and head, so it has no meaning for linear probing
    on frozen precomputed embeddings.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        """Initialize AttentionPooler.

        Args:
            hidden_dim: Backbone embedding dimension.
            num_heads: Number of attention heads. Must divide hidden_dim.
            dropout: Dropout probability on the attention weights.

        Raises:
            ValueError: If num_heads does not divide hidden_dim.
        """
        super().__init__()

        if hidden_dim % num_heads != 0:
            divisors = [
                d for d in range(1, min(num_heads, hidden_dim) + 1)
                if hidden_dim % d == 0
            ]
            raise ValueError(
                "num_heads ({}) must divide hidden_dim ({}). Valid head "
                "counts at or below {}: {}".format(
                    num_heads, hidden_dim, num_heads, divisors
                )
            )

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.query = nn.Parameter(torch.empty(1, 1, hidden_dim))
        nn.init.normal_(self.query, std=hidden_dim ** -0.5)

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Pool token embeddings over the sequence axis.

        Args:
            x: Token embeddings with sequence axis at dim and hidden axis
                as the only remaining trailing axis.
            dim: Sequence axis to pool over.

        Returns:
            Pooled embeddings with the sequence axis removed.

        Raises:
            ValueError: If the hidden axis does not match hidden_dim.
        """
        # Sequence axis to -2 leaves hidden at -1 for every wrapper
        # convention, including Borzoi/Enformer's (batch, hidden, len).
        x = torch.movedim(x, dim, -2)

        if x.shape[-1] != self.hidden_dim:
            raise ValueError(
                "Expected hidden dim {} at axis -1 after pooling axis {}, "
                "got shape {}.".format(self.hidden_dim, dim, tuple(x.shape))
            )

        leading_shape = x.shape[:-2]
        x = x.reshape(-1, x.shape[-2], self.hidden_dim)
        x = x.to(self.query.dtype)

        query = self.query.expand(x.shape[0], -1, -1)
        pooled, _ = self.attention(query, x, x, need_weights=False)
        pooled = self.norm(pooled.squeeze(-2))

        return pooled.reshape(*leading_shape, self.hidden_dim)
