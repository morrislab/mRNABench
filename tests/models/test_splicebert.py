from collections import namedtuple

import pytest
from unittest.mock import patch

pytest.importorskip("torch")

import torch
from mrna_bench.models.splicebert import SpliceBERT


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def splicebert_510(device) -> SpliceBERT:
    """Get SpliceBERT model."""
    return SpliceBERT("SpliceBERT.510nt", device)


@pytest.fixture(scope="module")
def splicebert_1024(device) -> SpliceBERT:
    """Get SpliceBERT model."""
    return SpliceBERT("SpliceBERT.1024nt", device)


def test_splicebert_forward(splicebert_510):
    """Test SpliceBERT forward pass."""
    assert splicebert_510.max_length == 510
    assert splicebert_510.is_sixtrack is False

    out = splicebert_510.embed_sequence("ATGATG")
    assert out.shape == (1, 512)


def test_splicebert_forward_chunked_1024(splicebert_1024):
    """Test SpliceBERT forward pass with chunking for 1024nt model."""
    spillover = 10
    input_seq = "A" * (splicebert_1024.max_length + spillover)

    ground_truth_vals = torch.mean(torch.cat([
        torch.arange(splicebert_1024.max_length + 2).float(),
        torch.arange(spillover + 2).float()
    ])).repeat(1, 512)

    def side_effect(input_ids):
        pos = torch.arange(input_ids.shape[1]).unsqueeze(0).unsqueeze(-1)
        pos = pos.float().repeat(1, 1, 512)

        MockOut = namedtuple("MockOut", ["last_hidden_state"])
        mock_out = MockOut(last_hidden_state=pos)
        return mock_out

    with patch.object(
        splicebert_1024.model,
        "forward",
        side_effect=side_effect
    ):
        output = splicebert_1024.embed_sequence(
            input_seq,
            agg_fn=torch.mean
        ).cpu()

        assert torch.allclose(output, ground_truth_vals)


def test_splicebert_forward_chunked_510(splicebert_510):
    """Test SpliceBERT forward pass with chunking for 510nt model."""
    spillover = 10
    input_seq = "A" * spillover + "G" * splicebert_510.max_length

    ground_truth_vals = torch.mean(torch.cat([
        torch.arange(splicebert_510.max_length + 2).float(),
        torch.arange(splicebert_510.max_length + 2).float()
    ])).repeat(1, 512)

    def side_effect(input_ids):
        assert input_ids.shape[1] == 512

        pos = torch.arange(input_ids.shape[1]).unsqueeze(0).unsqueeze(-1)
        pos = pos.float().repeat(1, 1, 512)

        if mock_method.call_count == 1:
            assert torch.all(input_ids[1: 1 + spillover] == 6)
            assert torch.all(input_ids[1 + spillover:-1] == 8)
        else:
            assert torch.all(input_ids[1:-1] == 8)

        MockOut = namedtuple("MockOut", ["last_hidden_state"])
        mock_out = MockOut(last_hidden_state=pos)
        return mock_out

    with patch.object(splicebert_510.model, "forward") as mock_method:
        mock_method.side_effect = side_effect

        output = splicebert_510.embed_sequence(
            input_seq,
            agg_fn=torch.mean
        ).cpu()

        assert torch.allclose(output, ground_truth_vals)
