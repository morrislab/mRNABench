import pytest

pytest.importorskip("torch")

import torch
from mrna_bench.models.aido import AIDORNA


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def aidomodel(device) -> AIDORNA:
    """Get AIDORNA model."""
    return AIDORNA("aido_rna_650m", device)


def test_aido_forward(aidomodel):
    """Test AIDORNA initialization and forward pass."""
    assert aidomodel.is_sixtrack is False

    out = aidomodel.embed_sequence("ATGATG")
    assert out.shape == (1, 1280)


def test_aido_forward_long(aidomodel):
    """Test AIDORNA forward pass."""
    long_sequence = "ATGC" * 257  # 1028 nucleotides
    out = aidomodel.embed_sequence(long_sequence)
    assert out.shape == (1, 1280)
