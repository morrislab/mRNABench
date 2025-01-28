import pytest

pytest.importorskip("torch")

import torch
from mrna_bench.models.dnabert import DNABERT2


@pytest.fixture
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def test_dnabert2_forward(device):
    """Test DNABERT2 initialization and forward pass."""
    model = DNABERT2("dnabert2", device)

    out = model.embed_sequence("ATGATG")
    assert out.shape == (1, 768)
