import pytest

pytest.importorskip("torch")

import torch
from mrna_bench.models.dnabert import DNABERT2


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def dnabert2(device) -> DNABERT2:
    """Get DNABERT2 model."""
    return DNABERT2("dnabert2", device)


def test_dnabert2_forward(dnabert2):
    """Test DNABERT2 forward pass."""
    assert dnabert2.is_sixtrack is False

    out = dnabert2.embed_sequence("ATGATG")
    assert out.shape == (1, 768)


def test_dnabert2_errors_overlap(dnabert2):
    """Test DNABERT2 throws error when using overlap."""
    with pytest.raises(ValueError):
        dnabert2.embed_sequence("ATGATG", overlap=1)
