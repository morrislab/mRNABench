import pytest

pytest.importorskip("torch")

import torch
from mrna_bench.models.nucleotide_transformer_v3 import NucleotideTransformerV3


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def ntmodelv3(device) -> NucleotideTransformerV3:
    """Get NucleotideTransformerV3 model."""
    return NucleotideTransformerV3("v3_8m_pre", device)


def test_ntv3_forward(ntmodelv3):
    """Test NucleotideTransformerV3 initialization and forward pass."""
    assert ntmodelv3.is_sixtrack is False

    out = ntmodelv3.embed_sequence("ATGATG")
    assert out.shape == (1, 256)


def test_ntv3_forward_posttrained(device):
    """Test NucleotideTransformerV3 post-trained model forward pass."""
    model = NucleotideTransformerV3("v3_100m_post", device)
    model.set_species("human")

    out = model.embed_sequence("ATGATG")
    assert out.shape == (1, 768)


def test_ntv3_forward_non_128(ntmodelv3):
    """Test NucleotideTransformerV3 forward pass with non-multiple of 128 length."""
    long_sequence = "ATGC" * 33  # 132 nucleotides
    out = ntmodelv3.embed_sequence(long_sequence)
    assert out.shape == (1, 256)
