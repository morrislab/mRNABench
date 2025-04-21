import pytest
from unittest.mock import patch

pytest.importorskip("torch")

import torch
from mrna_bench.models.nucleotide_transformer import NucleotideTransformer


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def ntmodel(device) -> NucleotideTransformer:
    """Get NucleotideTransformer model."""
    return NucleotideTransformer("v2-50m-multi-species", device)


def test_nt_forward(ntmodel):
    """Test NucleotideTransformer initialization and forward pass."""
    assert ntmodel.is_sixtrack is False

    out = ntmodel.embed_sequence("ATGATG")
    assert out.shape == (1, 512)


def test_nt_forward_chunked(ntmodel):
    """Test NucleotideTransformer forward pass with chunking."""
    tokenizer = ntmodel.tokenizer

    spillover = 10
    # NOTE: ATGATG encodes to token 461
    input_seq = "ATGATG" * (tokenizer.model_max_length + spillover)

    ground_truth_vals = torch.mean(torch.cat([
        torch.arange(ntmodel.max_length).float(),
        torch.arange(spillover + 1).float()
    ])).repeat(1, 512)

    def side_effect(
        input_ids,
        attention_mask,
        encoder_attention_mask,
        output_hidden_states
    ):
        pos = torch.arange(input_ids.shape[1]).unsqueeze(0).unsqueeze(-1)
        pos = pos.float().repeat(1, 1, 512)

        mock_out = {"hidden_states": [pos]}
        return mock_out

    with patch.object(ntmodel.model, "forward", side_effect=side_effect):
        output = ntmodel.embed_sequence(input_seq, agg_fn=torch.mean).cpu()

        assert torch.allclose(output, ground_truth_vals)
