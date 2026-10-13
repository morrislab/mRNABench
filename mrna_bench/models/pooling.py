from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class Pooling:
    """Aggregates token states into the representation that is persisted.

    A pooling is identified by its class name and fields, which are stored
    with the embeddings it produces. Subclasses implement __call__.
    """

    def __call__(
        self,
        hidden_states: torch.Tensor,
        dim: int = 0,
    ) -> torch.Tensor:
        """Pool token states.

        Args:
            hidden_states: Token representations to pool.
            dim: Dimension containing the tokens.

        Returns:
            Pooled representations in FP32.
        """
        raise NotImplementedError

    @property
    def config(self) -> dict:
        """Class name and fields identifying this pooling."""
        return {"name": type(self).__name__, **asdict(self)}


@dataclass(frozen=True)
class MeanPool(Pooling):
    """Mean of the token states."""

    def __call__(
        self,
        hidden_states: torch.Tensor,
        dim: int = 0,
    ) -> torch.Tensor:
        """Mean-pool token states in FP32."""
        # there can be perf differences if using a FA2
        # model with FP16 hidden states, so we cast to FP32
        return hidden_states.float().mean(dim=dim)


@dataclass(frozen=True)
class NoPool(Pooling):
    """Token states left unpooled."""

    def __call__(
        self,
        hidden_states: torch.Tensor,
        dim: int = 0,
    ) -> torch.Tensor:
        """Return token states unpooled, in FP32."""
        _ = dim
        return hidden_states.float()


@dataclass(frozen=True)
class PositionPool(Pooling):
    """Mean of the token states in sliding windows.

    Assumes one token per nucleotide. Window ``k`` covers positions
    ``[k * stride, k * stride + window_size)``; the last window may be
    shorter.
    """

    window_size: int
    stride: int

    def __call__(
        self,
        hidden_states: torch.Tensor,
        dim: int = 0,
    ) -> torch.Tensor:
        """Map (seq_len, H) token states to (n_windows, H)."""
        _ = dim
        seq_len = hidden_states.shape[0]
        n_windows = 1 + max(
            0, -(-(seq_len - self.window_size) // self.stride)
        )
        return torch.stack([
            hidden_states[k * self.stride:k * self.stride + self.window_size]
            .float().mean(0)
            for k in range(n_windows)
        ])
