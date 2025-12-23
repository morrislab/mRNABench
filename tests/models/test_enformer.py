from collections import namedtuple
from unittest.mock import patch

import pytest
import torch

from mrna_bench.models.enformer import Enformer


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def enformer(device) -> Enformer:
    """Get Enformer model."""
    return Enformer("enformer-official-rough", device)


def test_enformer_forward(enformer):
    """Test Enformer forward pass."""
    assert enformer.is_sixtrack is False

    text = "ACGT" * 25000
    output = enformer.embed_sequence(text)
    assert output.shape[0] == 1
    assert output.shape[1] == 3072


def test_enformer_padding_logic(enformer):
    """Test that padding and bin extraction work correctly."""
    bin_size = enformer.bin_size
    max_length = enformer.max_length

    def side_effect(batch, return_embeddings, target_length):
        seq_len = batch.shape[1]
        num_bins = seq_len // bin_size
        pos = torch.arange(num_bins).unsqueeze(0).unsqueeze(-1)
        emb = pos.expand(1, -1, 3072).float()
        return None, emb

    with patch.object(enformer, "model", side_effect=side_effect):
        short_seq = "ACGT" * 10000
        seq_len = len(short_seq)

        output = enformer.embed_sequence(short_seq, agg_fn=lambda x, dim: x)

        padding_left = (max_length - seq_len) // 2
        expected_start_bin = padding_left // bin_size
        expected_end_bin = (padding_left + seq_len + (bin_size - 1)) // bin_size
        expected_num_bins = expected_end_bin - expected_start_bin

        assert output.shape[2] == expected_num_bins

        first_bin_val = output[0, 0, 0].item()
        assert first_bin_val == expected_start_bin
