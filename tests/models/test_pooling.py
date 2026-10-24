import numpy as np
import pytest
import torch

from mrna_bench.models.pooling import (
    MeanMaxPool,
    MeanPool,
    NoPool,
    get_pooling_class,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_mean_max_pool_halves(dtype):
    """The first half matches MeanPool exactly and the max half is the max."""
    hidden = torch.randn(7, 5).to(dtype)

    pooled = MeanMaxPool()(hidden)

    assert pooled.shape == (10,) and pooled.dtype == torch.float32
    assert torch.equal(pooled[:5], MeanPool()(hidden))
    assert torch.equal(pooled[5:], hidden.float().max(0).values)


def test_mean_max_pool_empty_is_nan():
    """No tokens to pool gives NaN of the pooled width, as MeanPool does."""
    pooled = MeanMaxPool()(torch.zeros(0, 4))

    assert pooled.shape == (8,) and torch.isnan(pooled).all()


def test_select_view():
    """Each pooling selects its own feature views."""
    emb = np.arange(12).reshape(2, 6)

    views = ["mean", "max", "mean+max"]
    mean, max_, both = (MeanMaxPool.select_view(emb, v) for v in views)
    np.testing.assert_array_equal(mean, emb[:, :3])
    np.testing.assert_array_equal(max_, emb[:, 3:])
    np.testing.assert_array_equal(both, emb)
    np.testing.assert_array_equal(MeanPool.select_view(emb, "mean"), emb)
    np.testing.assert_array_equal(NoPool.select_view(emb, "none"), emb)


def test_get_pooling_class():
    """Stored pooling configs map back to their pooling class."""
    for pooling in [MeanPool(), MeanMaxPool(), NoPool()]:
        assert get_pooling_class(pooling.config) is type(pooling)
    assert get_pooling_class({"name": "Unknown"}) is None


@pytest.mark.parametrize("pooling,view", [
    (MeanPool, "max"),
    (NoPool, "mean"),
    (MeanMaxPool, "none"),
])
def test_select_view_rejects_missing_view(pooling, view):
    """A view the pooling did not produce is an error."""
    with pytest.raises(ValueError, match="not available"):
        pooling.select_view(np.zeros((2, 4)), view)
