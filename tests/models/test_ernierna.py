from collections import namedtuple

import pytest
from unittest.mock import patch

import torch

from mrna_bench.models.ernierna import ERNIERNA


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def ernierna(device) -> ERNIERNA:
    """Get ERNIE-RNA model."""
    return ERNIERNA("ernierna", device)


def test_ernierna_forward(ernierna):
    """Test ERNIE-RNA forward pass."""
    assert ernierna.is_sixtrack is False

    text = "ACTTGGCCA"
    output = ernierna.embed_sequence(text)
    assert output.shape == (1, 768)


def test_ernierna_forward_conversion(ernierna):
    """Test ERNIE-RNA forward pass converts T->U."""
    text = "ACTTGGCCA"

    with patch.object(
        ernierna,
        "chunk_sequence",
        side_effect=ernierna.chunk_sequence
    ) as mock:
        ernierna.embed_sequence(text)
        mock.assert_called_once_with("ACUUGGCCA", ernierna.max_length - 2, 0)


def test_ernierna_forward_chunked(ernierna):
    """Test ERNIE-RNA forward pass for chunked inputs."""
    input_length = ernierna.max_length + 100
    text = "A" * input_length

    # Assume only two chunks. Adding constant of 4 accounts for sep/cls for
    # both first and second chunk.
    ground_truth_vals = torch.mean(torch.cat([
        torch.arange(ernierna.max_length).float(),
        torch.arange((input_length - ernierna.max_length) + 4).float()
    ]))

    def side_effect(input_ids, attention_mask):
        if mock_forward.call_count == 1:
            assert input_ids.shape[1] == ernierna.max_length
        else:
            assert input_ids.shape[1] == input_length - ernierna.max_length + 4

        pos = torch.arange(input_ids.shape[1]).unsqueeze(0).unsqueeze(-1)
        pos = pos.float().repeat(1, 1, 768)

        MockOut = namedtuple("MockOut", ["last_hidden_state"])
        return MockOut(pos)

    with patch.object(
        ernierna,
        "model",
        side_effect=side_effect
    ) as mock_forward:
        output = ernierna.embed_sequence(text)
        assert torch.allclose(output, ground_truth_vals)
        assert mock_forward.call_count == 2
