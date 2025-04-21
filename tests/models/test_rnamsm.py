from collections import namedtuple

import pytest
from unittest.mock import patch

import torch

from mrna_bench.models.rnamsm import RNAMSM


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def rnamsm(device) -> RNAMSM:
    """Get RNA-MSM model."""
    return RNAMSM("rnamsm", device)


def test_rnamsm_forward(rnamsm):
    """Test RNA-MSM forward pass."""
    assert rnamsm.is_sixtrack is False

    text = "ACTTGGCCA"
    output = rnamsm.embed_sequence(text)
    assert output.shape == (1, 768)


def test_rnamsm_forward_conversion(rnamsm):
    """Test RNA-MSM forward pass converts T->U."""
    text = "ACTTGGCCA"

    with patch.object(
        rnamsm,
        "chunk_sequence",
        side_effect=rnamsm.chunk_sequence
    ) as mock:
        rnamsm.embed_sequence(text)
        mock.assert_called_once_with("ACUUGGCCA", rnamsm.max_length - 2)


def test_rnamsm_forward_chunked(rnamsm):
    """Test RNA-MSM forward pass for chunked inputs."""
    input_length = rnamsm.max_length + 100
    text = "A" * input_length

    # Assume only two chunks. Adding constant of 4 accounts for sep/cls for
    # both first and second chunk.
    ground_truth_vals = torch.mean(torch.cat([
        torch.arange(rnamsm.max_length).float(),
        torch.arange((input_length - rnamsm.max_length) + 4).float()
    ]))

    def side_effect(input_ids, attention_mask):
        if mock_forward.call_count == 1:
            assert input_ids.shape[1] == rnamsm.max_length
        else:
            assert input_ids.shape[1] == input_length - rnamsm.max_length + 4

        pos = torch.arange(input_ids.shape[1]).unsqueeze(0).unsqueeze(-1)
        pos = pos.float().repeat(1, 1, 768)

        MockOut = namedtuple("MockOut", ["last_hidden_state"])
        return MockOut(pos)

    with patch("multimolecule.RnaMsmModel.forward") as mock_forward:
        mock_forward.side_effect = side_effect

        output = rnamsm.embed_sequence(text, agg_fn=torch.mean).cpu()

        assert torch.allclose(output, ground_truth_vals)
