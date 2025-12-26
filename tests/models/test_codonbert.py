import pytest

pytest.importorskip("torch")
import torch
from mrna_bench.models.codonbert import CodonBERT


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def model(device) -> CodonBERT:
    """Get CodonBERT model."""
    return CodonBERT("codonbert", device)


def test_codonbert_forward(model):
    """Test CodonBERT forward pass."""
    out = model.embed_sequence("ATGATG")
    assert out.shape == (1, 768)


def test_codonbert_sixtrack_not_supported(model):
    """Test CodonBERT sixtrack raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        model.embed_sequence_sixtrack("ATGATG", None, None, None)


def test_codonbert_invalid_version(device):
    """Test CodonBERT raises ValueError for invalid version."""
    with pytest.raises(ValueError):
        CodonBERT("invalid-version", device)
