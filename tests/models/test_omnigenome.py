from unittest.mock import patch

import pytest
import torch

from mrna_bench.models.omnigenome import OmniGenome


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def omnigenome_52m(device) -> OmniGenome:
    """Get OmniGenome 52M model."""
    return OmniGenome("omnigenome-52m", device)


@pytest.fixture(scope="module")
def omnigenome_186m(device) -> OmniGenome:
    """Get OmniGenome 186M model."""
    return OmniGenome("omnigenome-186m", device)


def test_omnigenome_52m_forward(omnigenome_52m):
    """Test OmniGenome 52M forward pass."""
    assert omnigenome_52m.is_sixtrack is False

    text = "ACUUGGCCA"
    output = omnigenome_52m.embed_sequence(text)
    assert output.shape[0] == 1
    assert output.shape[1] == 480


def test_omnigenome_186m_forward(omnigenome_186m):
    """Test OmniGenome 186M forward pass."""
    assert omnigenome_186m.is_sixtrack is False

    text = "ACUUGGCCA"
    output = omnigenome_186m.embed_sequence(text)
    assert output.shape[0] == 1
    assert output.shape[1] == 720


def test_omnigenome_forward_conversion(omnigenome_52m):
    """Test OmniGenome forward pass converts T->U."""
    text = "ACTTGGCCA"

    with patch.object(
        omnigenome_52m,
        "chunk_sequence",
        side_effect=omnigenome_52m.chunk_sequence
    ) as mock:
        omnigenome_52m.embed_sequence(text)
        mock.assert_called_once_with(
            "ACUUGGCCA",
            omnigenome_52m.max_length - 2
        )
