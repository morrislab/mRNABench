from collections import namedtuple

import pytest
from unittest.mock import patch

import torch

from mrna_bench.models.rnaernie import RNAErnie


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def rnaernie(device) -> RNAErnie:
    """Get RNAErnie model."""
    return RNAErnie("rnaernie", device)


def test_rnaernie_forward(rnaernie):
    """Test RNAErnie forward pass."""
    assert rnaernie.is_sixtrack is False

    text = "ACTTGGCCA"
    output = rnaernie.embed_sequence(text)
    assert output.shape == (1, 768)


def test_rnaernie_forward_conversion(rnaernie):
    """Test RNAErnie forward pass converts T->U."""
    text = "ACTTGGCCA"

    with patch.object(
        rnaernie,
        "chunk_sequence",
        side_effect=rnaernie.chunk_sequence
    ) as mock:
        rnaernie.embed_sequence(text)
        mock.assert_called_once_with("ACUUGGCCA", rnaernie.max_length - 2)


def test_rnaernie_forward_chunked(rnaernie):
    """Test RNAErnie forward pass for chunked inputs."""
    input_length = rnaernie.max_length + 100
    text = "A" * input_length

    # Assume only two chunks. Adding constant of 4 accounts for sep/cls for
    # both first and second chunk.
    ground_truth_vals = torch.mean(torch.cat([
        torch.arange(rnaernie.max_length).float(),
        torch.arange((input_length - rnaernie.max_length) + 4).float()
    ]))

    def side_effect(input_ids, attention_mask):
        if mock_forward.call_count == 1:
            assert input_ids.shape[1] == rnaernie.max_length
        else:
            assert input_ids.shape[1] == input_length - rnaernie.max_length + 4

        pos = torch.arange(input_ids.shape[1]).unsqueeze(0).unsqueeze(-1)
        pos = pos.float().repeat(1, 1, 768)

        MockOut = namedtuple("MockOut", ["last_hidden_state"])
        return MockOut(pos)

    with patch.object(
        rnaernie,
        "model",
        side_effect=side_effect
    ) as mock_forward:
        output = rnaernie.embed_sequence(text)
        assert torch.allclose(output, ground_truth_vals)
        assert mock_forward.call_count == 2
