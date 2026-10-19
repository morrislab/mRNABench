import pytest

import torch

from mrna_bench.fine_tune.poolers import no_pool


@pytest.mark.parametrize("shape,dim,expected", [
    ((8, 4), 0, (8, 4)),          # chunked wrappers: (len, hidden)
    ((1, 8, 4), 1, (1, 8, 4)),    # helix_mrna/evo2: (batch, len, hidden)
    ((1, 4, 8), 2, (1, 8, 4)),    # borzoi/enformer: (batch, hidden, len)
])
def test_normalizes_sequence_axis(shape, dim, expected):
    """Check the sequence axis lands at -2 for every wrapper layout."""
    assert no_pool(torch.randn(*shape), dim=dim).shape == expected


def test_preserves_token_values():
    """Check no_pool reorders axes without altering embeddings."""
    x = torch.randn(1, 4, 8)

    assert torch.equal(no_pool(x, dim=2), x.transpose(1, 2))


def test_mean_over_no_pool_matches_torch_mean():
    """Check pooling no_pool output reproduces the mean it replaces."""
    x = torch.randn(1, 4, 8)

    assert torch.allclose(
        no_pool(x, dim=2).mean(dim=-2), torch.mean(x, dim=2), atol=1e-6
    )
