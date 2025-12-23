import pytest

pytest.importorskip("torch")
import torch
from mrna_bench.models.dnabert_s import DNABERTS


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def model(device) -> DNABERTS:
    """Get DNABERT-S model."""
    return DNABERTS("dnabert-s", device)


def test_dnaberts_forward(model):
    """Test DNABERT-S forward pass."""
    out = model.embed_sequence("ATGATG")
    assert out.shape == (1, 768)


def test_dnaberts_sixtrack_not_supported(model):
    """Test DNABERT-S sixtrack raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        model.embed_sequence_sixtrack("ATGATG", None, None, None)


def test_dnaberts_invalid_version(device):
    """Test DNABERT-S raises ValueError for invalid version."""
    with pytest.raises(ValueError):
        DNABERTS("invalid-version", device)
