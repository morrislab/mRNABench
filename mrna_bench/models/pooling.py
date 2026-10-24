from dataclasses import asdict, dataclass
from typing import ClassVar

import numpy as np
import torch


@dataclass(frozen=True)
class Pooling:
    """Aggregates token states into the representation that is persisted.

    A pooling is identified by its class name and fields, which are stored
    with the embeddings it produces. Subclasses implement __call__ and list
    the feature views of their output in views, the first being the whole
    output.
    """

    views: ClassVar[tuple[str, ...]] = ()

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

    @classmethod
    def select_view(cls, embeddings: np.ndarray, view: str) -> np.ndarray:
        """Select a feature view of embeddings this pooling produced.

        Args:
            embeddings: Pooled embeddings, shape (N, D).
            view: One of views.

        Returns:
            Embedding columns of the view.
        """
        if view not in cls.views:
            raise ValueError(
                f"View {view!r} is not available for {cls.__name__}."
            )
        return embeddings


@dataclass(frozen=True)
class MeanPool(Pooling):
    """Mean of the token states."""

    views = ("mean",)

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
    """Token states left unpooled.

    Also recorded for models that return their own fixed features and ignore
    the pooling, such as the naive baselines.
    """

    views = ("none",)

    def __call__(
        self,
        hidden_states: torch.Tensor,
        dim: int = 0,
    ) -> torch.Tensor:
        """Return token states unpooled, in FP32."""
        _ = dim
        return hidden_states.float()


@dataclass(frozen=True)
class MeanMaxPool(Pooling):
    """Mean and max of the token states, concatenated as [mean, max].

    The mean half is identical to MeanPool, so one embedding run serves the
    mean, max and mean+max feature views.
    """

    views = ("mean+max", "mean", "max")

    def __call__(
        self,
        hidden_states: torch.Tensor,
        dim: int = 0,
    ) -> torch.Tensor:
        """Mean- and max-pool token states in FP32."""
        mean = hidden_states.float().mean(dim=dim)
        if hidden_states.shape[dim] == 0:
            return torch.cat([mean, torch.full_like(mean, float("nan"))], -1)
        # max is a selection, so taking it before the cast is exact
        return torch.cat([mean, hidden_states.amax(dim=dim).float()], -1)

    @classmethod
    def select_view(cls, embeddings: np.ndarray, view: str) -> np.ndarray:
        """Select the mean half, the max half, or both."""
        embeddings = super().select_view(embeddings, view)
        half = embeddings.shape[1] // 2
        if view == "mean":
            return embeddings[:, :half]
        if view == "max":
            return embeddings[:, half:]
        return embeddings


def get_pooling_class(config: dict) -> type[Pooling] | None:
    """Get the pooling class named in a stored pooling config."""
    classes = {cls.__name__: cls for cls in Pooling.__subclasses__()}
    return classes.get(config.get("name", ""))
