import numpy as np
import torch

from mrna_bench.models import mean_pool


def embed_one(
    model,
    sequence: str,
    cds: np.ndarray | None = None,
    splice: np.ndarray | None = None,
    agg_fn=mean_pool,
) -> torch.Tensor:
    """Call the public batch API for one sequence."""
    return torch.stack(model.embed(
        [sequence],
        [cds] if cds is not None else None,
        [splice] if splice is not None else None,
        agg_fn,
    ))


def assert_pooled_batch_matches_single(
    model,
    sequences: list[str],
    cds: list[np.ndarray] | None = None,
    splice: list[np.ndarray] | None = None,
) -> None:
    """Check pooled batch outputs against one-at-a-time inference."""
    batch_output = model.embed(sequences, cds, splice)
    for i, sequence in enumerate(sequences):
        single = model.embed(
            [sequence],
            [cds[i]] if cds is not None else None,
            [splice[i]] if splice is not None else None,
        )[0]
        torch.testing.assert_close(
            batch_output[i].cpu(),
            single.cpu(),
            atol=1e-5,
            rtol=1e-5,
        )


def assert_raw_batch_matches_single(
    model,
    sequences: list[str],
    batch_output: list[torch.Tensor],
    cds: list[np.ndarray] | None = None,
    splice: list[np.ndarray] | None = None,
    atol: float = 1e-4,
    min_cosine: float | None = None,
) -> None:
    """Check unpooled batch outputs against one-at-a-time inference."""
    def identity(x, **kwargs):
        return x
    for i, sequence in enumerate(sequences):
        single = model.embed(
            [sequence],
            [cds[i]] if cds is not None else None,
            [splice[i]] if splice is not None else None,
            identity,
        )[0]
        if min_cosine is not None:
            cosine = torch.nn.functional.cosine_similarity(
                batch_output[i].flatten().float().cpu(),
                single.flatten().float().cpu(),
                dim=0,
            )
            assert cosine >= min_cosine
            continue
        torch.testing.assert_close(
            batch_output[i].cpu(),
            single.cpu(),
            atol=atol,
            rtol=1e-5,
        )
