from collections import namedtuple

import pytest
from unittest.mock import patch

import numpy as np
pytest.importorskip("torch")
import torch

from mrna_bench.models.utrlm import UTRLM


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def utrlm(device) -> UTRLM:
    """Get UTR-LM model."""
    return UTRLM("utrlm-te_el", device)


@pytest.fixture(scope="module")
def utrlm_5utr(device) -> UTRLM:
    """Get UTR-LM model for 5'utr only."""
    return UTRLM("utrlm-te_el-utronly", device)


def test_utrlm_forward(utrlm):
    """Test UTR-LM initialization and forward pass."""
    text = "ACUUUGGCCA"
    output = utrlm.embed_sequence(text, agg_fn=torch.mean).cpu()
    assert output.shape == (1, 128)


def test_utrlm_forward_conversion(utrlm):
    """Test UTR-LM forward pass converts T->U."""
    text = "ACTTTGGCCA"

    with patch.object(
        utrlm,
        "chunk_sequence",
        side_effect=utrlm.chunk_sequence
    ) as mock:
        utrlm.embed_sequence(text)
        mock.assert_called_once_with("ACUUUGGCCA", utrlm.max_length - 2)


def test_utrlm_forward_chunked(utrlm):
    """Test UTR-LM forward pass for chunked inputs."""
    input_length = 1200
    text = "A" * input_length

    # Assume only two chunks. Adding constant of 4 accounts for sep/cls for
    # both first and second chunk.
    ground_truth_vals = torch.mean(torch.cat([
        torch.arange(utrlm.max_length).float(),
        torch.arange((input_length - utrlm.max_length) + 4).float()
    ])).repeat(1, 128)

    def side_effect(input_ids, attention_mask):
        pos = torch.arange(input_ids.shape[1]).unsqueeze(0).unsqueeze(-1)
        pos = pos.float().repeat(1, 1, 128)

        MockOut = namedtuple("MockOut", ["last_hidden_state"])
        return MockOut(pos)

    with patch("multimolecule.UtrLmModel.forward") as mock_forward:
        mock_forward.side_effect = side_effect

        output = utrlm.embed_sequence(text, agg_fn=torch.mean).cpu()

        assert torch.allclose(output, ground_truth_vals)


def test_utrlm_forward_5utr(utrlm_5utr):
    """Test embedding only the 5'utr using UTR-LM."""
    text = "AAAGGG"
    cds = np.array([0, 0, 0, 1, 1, 1])

    ground_truth_vals = torch.mean(torch.arange(5).float()).repeat(1, 128)

    def side_effect(input_ids, attention_mask):
        pos = torch.arange(input_ids.shape[1]).unsqueeze(0).unsqueeze(-1)
        pos = pos.float().repeat(1, 1, 128)

        MockOut = namedtuple("MockOut", ["last_hidden_state"])
        return MockOut(pos)

    with patch("multimolecule.UtrLmModel.forward") as mock_forward:
        mock_forward.side_effect = side_effect

        output = utrlm_5utr.embed_sequence_sixtrack(
            text,
            cds,
            cds,
            agg_fn=torch.mean
        ).cpu()

        assert torch.allclose(output, ground_truth_vals)


def test_utrlm_forward_5utr_missing(utrlm_5utr):
    """Test embedding only the 5'utr using UTR-LM when no cds exists.

    Behaviour should be to use whole sequence.
    """
    text = "AAAGGG"
    cds = np.array([0, 0, 0, 0, 0, 0])

    ground_truth_vals = torch.mean(torch.arange(8).float()).repeat(1, 128)

    def side_effect(input_ids, attention_mask):
        pos = torch.arange(input_ids.shape[1]).unsqueeze(0).unsqueeze(-1)
        pos = pos.float().repeat(1, 1, 128)

        MockOut = namedtuple("MockOut", ["last_hidden_state"])
        return MockOut(pos)

    with patch("multimolecule.UtrLmModel.forward") as mock_forward:
        mock_forward.side_effect = side_effect

        output = utrlm_5utr.embed_sequence_sixtrack(
            text,
            cds,
            cds,
            agg_fn=torch.mean
        ).cpu()

        assert torch.allclose(output, ground_truth_vals)


def test_utrlm_forward_5utr_sixtrack(utrlm_5utr):
    """Test UTR-LM properly sets sixtrack flag for UTR only embedding."""
    assert utrlm_5utr.is_sixtrack is True
