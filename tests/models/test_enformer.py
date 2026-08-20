import pytest

import math
from unittest.mock import patch

import torch

from tests.model_utils import assert_raw_batch_matches_single, embed_one

from mrna_bench.models.enformer import Enformer


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def enformer(device) -> Enformer:
    """Get Enformer model."""
    return Enformer("enformer-official-rough", device, "eager")


def test_enformer_forward(enformer):
    """Test Enformer forward pass."""
    text = "ACGT" * 25000
    output = embed_one(enformer, text)
    assert output.shape[0] == 1
    assert output.shape[1] == 3072


def test_enformer_embed_batch(enformer):
    """Test batch embed matches individual embeddings."""
    enformer.set_inference_mode()
    sequences = [
        "ACGT" * 25000,
        "ACGT" * 30000,
    ]

    with patch.object(
        enformer.model,
        "forward",
        wraps=enformer.model.forward,
    ) as forward:
        batch_output = torch.stack(enformer.embed(sequences)).cpu()
    assert forward.call_args.args[0].shape[0] == 2
    assert batch_output.shape == (2, 3072)

    for i, seq in enumerate(sequences):
        single_output = embed_one(enformer, seq).cpu()
        assert torch.allclose(
            batch_output[i:i + 1],
            single_output,
            atol=1e-5
        ), "Mismatch at sequence {} (len {})".format(i, len(seq))


@torch.no_grad()
def test_enformer_predict_tracks(enformer):
    """Expose native tracks aligned to the original sequence bins."""
    with patch.object(
        enformer.model,
        "forward",
        wraps=enformer.model.forward,
    ) as forward:
        output = enformer.predict_tracks(["ACGT" * 250])[0]
    batch = forward.call_args.args[0]
    first_base = int((batch[0].sum(dim=-1) > 0).nonzero()[0])

    assert output.bin_size == enformer.bin_size
    assert output.start == 0
    assert first_base % enformer.bin_size == 0
    assert output.values["human"].shape[1] == 5313


@torch.no_grad()
def test_enformer_embed_ragged_agg(enformer):
    """Test embed with identity agg_fn returns per-bin embeddings (ragged)."""
    seqs = ["ACGT" * 250, "ACGT" * 1000]  # 1000 and 4000 bp -> different bin counts
    out = enformer.embed(seqs, agg_fn=lambda x, **kwargs: x)
    # Unpooled bins retain padding-context differences that pooling averages.
    assert_raw_batch_matches_single(enformer, seqs, out, atol=1e-3)
    assert out[0].dim() == 2  # (num_bins, hidden_dim)
    assert out[1].dim() == 2
    assert out[0].shape[0] != out[1].shape[0]  # ragged: different bin counts
    assert out[0].shape[1] == out[1].shape[1]  # same hidden dim


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

        output = embed_one(enformer, short_seq, agg_fn=lambda x: x)

        padding_left = (max_length - seq_len) // 2 // bin_size * bin_size
        expected_start_bin = padding_left // bin_size
        expected_end_bin = math.ceil((padding_left + seq_len) / bin_size)
        expected_num_bins = expected_end_bin - expected_start_bin

        assert output.shape[1] == expected_num_bins

        first_bin_val = output[0, 0, 0].item()
        assert first_bin_val == expected_start_bin


def test_enformer_extract_structure(enformer):
    """extract() returns (dict, dict) with matching keys; hidden states are 2D."""
    seq = "ACGT" * 1000
    h, s = enformer.extract([seq], layers=[0])
    assert isinstance(h, dict) and isinstance(s, dict)
    assert set(h.keys()) == set(s.keys())
    layer = next(iter(h))
    assert h[layer][0][0].dim() == 2
    assert h[layer][0][0].device.type == "cpu"


def test_enformer_extract_layer_selection(enformer):
    """Requesting layers=[0] returns exactly 1 layer."""
    seq = "ACGT" * 1000
    h, _ = enformer.extract([seq], layers=[0])
    assert len(h) == 1


def test_enformer_extract_scores_none(enformer):
    """CNN/stem layers return None scores (layers=[0] resolves to stem)."""
    seq = "ACGT" * 1000
    _, s = enformer.extract([seq], layers=[0], return_attentions=True)
    assert all(v is None for v in s.values())


def test_enformer_extract_transformer_attention(enformer):
    """Transformer layers return (H, T, T) attention weights with rows summing to 1."""
    seq = "ACGT" * 1000
    _, s = enformer.extract([seq], layers=["transformer.0"], return_attentions=True)
    assert s["transformer.0"] is not None
    w = s["transformer.0"][0][0]  # seq[0], chunk[0] -> (H, T, T)
    assert w.dim() == 3
    assert w.shape[1] == w.shape[2]  # T x T square
    assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-6)


def test_enformer_extract_cnn_scores_none_with_attentions(enformer):
    """CNN layers return None scores even when return_attentions=True."""
    seq = "ACGT" * 1000
    _, s = enformer.extract([seq], layers=["conv_tower.0"], return_attentions=True)
    assert s["conv_tower.0"] is None
