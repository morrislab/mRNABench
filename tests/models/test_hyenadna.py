import pytest

pytest.importorskip("torch")

import torch
from mrna_bench.models.hyenadna import HyenaDNA


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def hyenadna(device) -> HyenaDNA:
    """Get HyenaDNA model."""
    return HyenaDNA("hyenadna-small-32k-seqlen-hf", device)


def test_hyena_forward(hyenadna):
    """Test HyenaDNA forward pass."""
    assert hyenadna.is_sixtrack is False
    assert hyenadna.model.training is False

    out = hyenadna.embed_sequence("ATGATG")
    assert out.shape == (1, 256)


def test_hyena_errors_overlap(hyenadna):
    """Test HyenaDNA throws error if overlap is not zero."""
    with pytest.raises(ValueError):
        hyenadna.embed_sequence("ATGATG", overlap=1)
