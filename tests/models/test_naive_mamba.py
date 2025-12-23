import pytest
import numpy as np

pytest.importorskip("torch")
pytest.importorskip("mamba_ssm")
import torch
from mrna_bench.models.naive_mamba import NaiveMamba


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def model(device) -> NaiveMamba:
    """Get NaiveMamba model."""
    return NaiveMamba("naive-mamba", device)


def test_naive_mamba_sixtrack_forward(model):
    """Test NaiveMamba sixtrack forward pass."""
    sequence = "ATGATG"
    cds = np.array([1, 0, 0, 1, 0, 0])
    splice = np.array([0, 0, 0, 0, 0, 0])

    out = model.embed_sequence_sixtrack(sequence, cds, splice)
    # Output dim is d_model = 64
    assert out.shape == (1, 64)


def test_naive_mamba_is_sixtrack(model):
    """Test NaiveMamba is_sixtrack flag is True."""
    assert model.is_sixtrack is True


def test_naive_mamba_fourtrack_not_supported(model):
    """Test NaiveMamba fourtrack raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        model.embed_sequence("ATGATG")


def test_naive_mamba_agg_fn_not_supported(model):
    """Test that custom agg_fn raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        model.embed_sequence_sixtrack(
            "ATGATG",
            np.array([0] * 6),
            np.array([0] * 6),
            agg_fn=torch.sum
        )


def test_naive_mamba_longer_sequence(model):
    """Test NaiveMamba with longer sequence."""
    sequence = "ATGATGATGATGATGATG"
    cds = np.array([1, 0, 0] * 6)
    splice = np.array([0] * 18)

    out = model.embed_sequence_sixtrack(sequence, cds, splice)
    assert out.shape == (1, 64)
