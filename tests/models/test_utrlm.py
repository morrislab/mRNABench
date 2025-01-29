from collections import namedtuple

import pytest
from unittest.mock import patch

import numpy as np
pytest.importorskip("torch")
import torch

from mrna_bench.models.utrlm import UTRLM


@pytest.fixture
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def test_utrlm_forward(device):
    """Test UTR-LM initialization and forward pass."""
    text = "ACUUUGGCCA"
    model = UTRLM("utrlm-te_el", device)

    output = model.embed_sequence(text, agg_fn=torch.mean).cpu()
    assert output.shape == (1, 128)


def test_utrlm_forward_chunked(device):
    """Test UTR-LM forward pass for chunked inputs."""
    input_length = 1200
    text = "A" * input_length
    model = UTRLM("utrlm-te_el", device)

    # Assume only two chunks. Adding constant of 4 accounts for sep/cls for
    # both first and second chunk.
    ground_truth_vals = torch.mean(torch.cat([
        torch.arange(model.MAX_LENGTH).float(),
        torch.arange((input_length - model.MAX_LENGTH) + 4).float()
    ])).repeat(1, 128)

    def side_effect(input_ids, attention_mask):
        pos = torch.arange(input_ids.shape[1]).unsqueeze(0).unsqueeze(-1)
        pos = pos.float().repeat(1, 1, 128)

        MockOut = namedtuple("MockOut", ["last_hidden_state"])
        return MockOut(pos)

    with patch("multimolecule.UtrLmModel.forward") as mock_forward:
        mock_forward.side_effect = side_effect

        output = model.embed_sequence(
            text,
            overlap=0,
            agg_fn=torch.mean
        ).cpu()

        assert torch.allclose(output, ground_truth_vals)


def test_utrlm_forward_chunked_overlap(device):
    """Test UTR-LM forward pass for chunked inputs with overlap."""
    model = UTRLM("utrlm-te_el", device)

    overlap = model.MAX_LENGTH // 8
    input_seq = "A" * (model.MAX_LENGTH - 2) + "G" * (model.MAX_LENGTH // 4)

    c1_ids = [1] + [6] * (model.MAX_LENGTH - 2) + [2]
    c2_ids = [1] + [6] * overlap + [8] * (model.MAX_LENGTH // 4) + [2]

    ground_truth_vals = torch.mean(torch.cat([
        torch.arange(model.MAX_LENGTH).float(),
        torch.arange((len(input_seq) - model.MAX_LENGTH) + 4 + overlap).float()
    ])).repeat(1, 128)

    def side_effect(input_ids, attention_mask):
        # Evaluate overlap
        input_ids = input_ids.float().cpu()[0]

        is_chunk_1 = torch.equal(input_ids, torch.Tensor(c1_ids))
        is_chunk_2 = torch.equal(input_ids, torch.Tensor(c2_ids))
        assert is_chunk_1 or is_chunk_2

        pos = torch.arange(len(input_ids)).unsqueeze(0).unsqueeze(-1)
        pos = pos.float().repeat(1, 1, 128)

        MockOut = namedtuple("MockOut", ["last_hidden_state"])
        return MockOut(pos)

    with patch("multimolecule.UtrLmModel.forward") as mock_forward:
        mock_forward.side_effect = side_effect

        output = model.embed_sequence(
            input_seq,
            overlap=overlap,
            agg_fn=torch.mean
        ).cpu()

        assert torch.allclose(output, ground_truth_vals)


def test_utrlm_forward_5utr(device):
    """Test embedding only the 5'utr using UTR-LM."""
    text = "AAAGGG"
    cds = np.array([0, 0, 0, 1, 1, 1])

    model = UTRLM("utrlm-te_el", device)

    ground_truth_vals = torch.mean(torch.arange(5).float()).repeat(1, 128)

    def side_effect(input_ids, attention_mask):
        pos = torch.arange(input_ids.shape[1]).unsqueeze(0).unsqueeze(-1)
        pos = pos.float().repeat(1, 1, 128)

        MockOut = namedtuple("MockOut", ["last_hidden_state"])
        return MockOut(pos)

    with patch("multimolecule.UtrLmModel.forward") as mock_forward:
        mock_forward.side_effect = side_effect

        output = model.embed_sequence_sixtrack(
            text,
            cds,
            cds,
            overlap=0,
            agg_fn=torch.mean
        ).cpu()

        assert torch.allclose(output, ground_truth_vals)


def test_utrlm_forward_5utr_missing(device):
    """Test embedding only the 5'utr using UTR-LM when no cds exists.

    Behaviour should be to use whole sequence.
    """
    text = "AAAGGG"
    cds = np.array([0, 0, 0, 0, 0, 0])

    model = UTRLM("utrlm-te_el-utronly", device)

    ground_truth_vals = torch.mean(torch.arange(8).float()).repeat(1, 128)

    def side_effect(input_ids, attention_mask):
        pos = torch.arange(input_ids.shape[1]).unsqueeze(0).unsqueeze(-1)
        pos = pos.float().repeat(1, 1, 128)

        MockOut = namedtuple("MockOut", ["last_hidden_state"])
        return MockOut(pos)

    with patch("multimolecule.UtrLmModel.forward") as mock_forward:
        mock_forward.side_effect = side_effect

        output = model.embed_sequence_sixtrack(
            text,
            cds,
            cds,
            overlap=0,
            agg_fn=torch.mean
        ).cpu()

        assert torch.allclose(output, ground_truth_vals)


def test_utrlm_forward_5utr_sixtrack(device):
    """Test UTR-LM properly sets sixtrack flag for UTR only embedding."""
    model = UTRLM("utrlm-te_el-utronly", device)

    assert model.is_sixtrack is True
