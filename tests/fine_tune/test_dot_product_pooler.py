import pytest

import torch

from mrna_bench.fine_tune.poolers import DotProductPooler

HIDDEN_DIM = 16


@pytest.fixture
def pooler():
    """Dot product pooler over HIDDEN_DIM embeddings."""
    torch.manual_seed(0)
    return DotProductPooler(HIDDEN_DIM)


def test_holds_only_a_query_vector(pooler):
    """Check the pooler costs exactly hidden_dim parameters."""
    assert sum(p.numel() for p in pooler.parameters()) == HIDDEN_DIM


@pytest.mark.parametrize("shape,dim", [
    ((8, HIDDEN_DIM), 0),
    ((4, 8, HIDDEN_DIM), 1),
    ((4, HIDDEN_DIM, 8), 2),
])
def test_output_shape_matches_mean(pooler, shape, dim):
    """Check pooled shape matches torch.mean for every wrapper layout."""
    x = torch.randn(*shape)

    assert pooler(x, dim=dim).shape == torch.mean(x, dim=dim).shape


def test_zero_query_reduces_to_mean(pooler):
    """Check a zero query recovers mean pooling exactly."""
    with torch.no_grad():
        pooler.query.zero_()

    x = torch.randn(4, 8, HIDDEN_DIM)

    assert torch.allclose(pooler(x, dim=1), x.mean(dim=1), atol=1e-6)


def test_weights_form_a_convex_combination(pooler):
    """Check output stays within the convex hull of the tokens."""
    x = torch.randn(32, HIDDEN_DIM)

    pooled = pooler(x, dim=0)

    assert (pooled <= x.max(dim=0).values + 1e-5).all()
    assert (pooled >= x.min(dim=0).values - 1e-5).all()


def test_attends_non_uniformly(pooler):
    """Check a trained query weights tokens unequally."""
    x = torch.randn(8, HIDDEN_DIM)

    assert not torch.allclose(pooler(x, dim=0), torch.mean(x, dim=0))


def test_rejects_hidden_dim_mismatch(pooler):
    """Check a wrong hidden axis is rejected rather than silently pooled."""
    with pytest.raises(ValueError, match="Expected hidden dim"):
        pooler(torch.randn(8, HIDDEN_DIM + 1), dim=0)


def test_casts_input_to_parameter_dtype(pooler):
    """Check a half precision backbone output pools to the head's dtype."""
    x = torch.randn(8, HIDDEN_DIM, dtype=torch.bfloat16)

    assert pooler(x, dim=0).dtype == torch.float32


def test_gradients_reach_the_query(pooler):
    """Check pooled output backpropagates into the query."""
    pooler(torch.randn(8, HIDDEN_DIM), dim=0).sum().backward()

    assert pooler.query.grad is not None
    assert not torch.allclose(
        pooler.query.grad, torch.zeros_like(pooler.query)
    )


def test_upweights_tokens_aligned_with_the_query(pooler):
    """Check the token matching the query dominates the pooled output."""
    with torch.no_grad():
        pooler.query.copy_(torch.eye(HIDDEN_DIM)[0] * 50)

    x = torch.zeros(64, HIDDEN_DIM)
    x[:, 0] = -1.0
    x[7] = torch.ones(HIDDEN_DIM)

    pooled = pooler(x, dim=0)

    assert torch.allclose(pooled, x[7], atol=1e-3)
