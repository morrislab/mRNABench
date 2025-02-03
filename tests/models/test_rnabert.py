from collections import namedtuple

import pytest
from unittest.mock import patch

import torch

from mrna_bench.models.rnabert import RNABERT


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def rnabert(device) -> RNABERT:
    """Get RNABERT model."""
    return RNABERT("rnabert", device)


def test_rnabert_forward(rnabert):
    """Test RNABERT forward pass."""
    assert rnabert.is_sixtrack is False

    text = "ACTTGGCCA"
    output = rnabert.embed_sequence(text)
    assert output.shape == (1, 120)


def test_rnabert_forward_conversion(rnabert):
    """Test RNABERT forward pass converts T->U."""
    text = "ACTTGGCCA"

    with patch.object(
        rnabert,
        "chunk_sequence",
        side_effect=rnabert.chunk_sequence
    ) as mock:
        rnabert.embed_sequence(text, overlap=0)
        mock.assert_called_once_with("ACUUGGCCA", rnabert.max_length - 2, 0)


def test_rnabert_forward_chunked(rnabert):
    """Test RNABERT forward pass for chunked inputs."""
    input_length = rnabert.max_length + 100
    text = "A" * input_length

    # Assume only two chunks. Adding constant of 4 accounts for sep/cls for
    # both first and second chunk.
    ground_truth_vals = torch.mean(torch.cat([
        torch.arange(rnabert.max_length).float(),
        torch.arange((input_length - rnabert.max_length) + 4).float()
    ]))

    def side_effect(input_ids, attention_mask):
        if mock_forward.call_count == 1:
            assert input_ids.shape[1] == rnabert.max_length
        else:
            assert input_ids.shape[1] == input_length - rnabert.max_length + 4

        pos = torch.arange(input_ids.shape[1]).unsqueeze(0).unsqueeze(-1)
        pos = pos.float().repeat(1, 1, 120)

        MockOut = namedtuple("MockOut", ["last_hidden_state"])
        return MockOut(pos)

    with patch.object(
        rnabert,
        "model",
        side_effect=side_effect
    ) as mock_forward:
        output = rnabert.embed_sequence(text, overlap=0)
        assert torch.allclose(output, ground_truth_vals)
        assert mock_forward.call_count == 2


def test_rnabert_forward_chunked_overlap(rnabert):
    """Test RNABERT forward pass for chunked inputs with overlap."""
    overlap = 100
    spillover = 200
    input_seq = "A" * (rnabert.max_length - 2) + "G" * spillover

    c1_ids = [1] + [6] * (rnabert.max_length - 2) + [2]
    c2_ids = [1] + [6] * overlap + [8] * spillover + [2]

    def side_effect(input_ids, attention_mask):
        if mock_forward.call_count == 1:
            assert input_ids[0].tolist() == c1_ids
        else:
            assert input_ids[0].tolist() == c2_ids

        pos = torch.arange(input_ids.shape[1]).unsqueeze(0).unsqueeze(-1)
        pos = pos.float().repeat(1, 1, 120)

        MockOut = namedtuple("MockOut", ["last_hidden_state"])
        return MockOut(pos)

    with patch.object(
        rnabert,
        "model",
        side_effect=side_effect
    ) as mock_forward:
        rnabert.embed_sequence(input_seq, overlap=overlap)
        assert mock_forward.call_count == 2
