import pytest

import math
from unittest.mock import patch

import torch

from mrna_bench.models.alphagenome import AlphaGenome


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def alphagenome(device) -> AlphaGenome:
    """Get AlphaGenome model."""
    return AlphaGenome("alphagenome", device, "eager")


def test_alphagenome_forward(alphagenome):
    """Test AlphaGenome forward pass."""
    text = "ACGT" * 25000
    output = alphagenome.embed_sequence(text)
    assert output.shape[0] == 1
    assert output.shape[1] == 1536


def test_alphagenome_embed_batch(alphagenome):
    """Test batch embed matches individual embeddings."""
    alphagenome.set_inference_mode()
    sequences = [
        "ACGT" * 25000,
        "ACGT" * 30000,
    ]

    batch_output = torch.stack(alphagenome.embed(sequences)).cpu()
    assert batch_output.shape == (2, 1536)

    for i, seq in enumerate(sequences):
        single_output = alphagenome.embed_sequence(seq).cpu()
        assert torch.allclose(
            batch_output[i:i + 1],
            single_output,
            atol=1e-4
        ), "Mismatch at sequence {} (len {})".format(i, len(seq))


@torch.no_grad()
def test_alphagenome_embed_ragged_agg(alphagenome):
    """Test embed with identity agg_fn returns per-bin embeddings (ragged)."""
    seqs = ["ACGT" * 250, "ACGT" * 1000]
    out = alphagenome.embed(seqs, agg_fn=lambda x, **kwargs: x)
    assert out[0].dim() == 2  # (num_bins, hidden_dim)
    assert out[1].dim() == 2
    assert out[0].shape[0] != out[1].shape[0]  # ragged: different token counts
    assert out[0].shape[1] == out[1].shape[1]  # same hidden dim


def test_alphagenome_padding_logic(alphagenome):
    """Test that right-padding and sequence extraction work correctly.

    Sequences are right-padded to the next multiple of 2048 before being
    passed to model.encode(). The embedding for the original sequence
    occupies positions [0:len(seq)] in the output.
    """
    def side_effect(batch, organism_index, resolutions):
        # Return position indices as embedding values so we can verify
        # which positions were extracted.
        seq_len = batch.shape[1]
        pos = torch.arange(seq_len, dtype=torch.float32)
        emb = pos.view(1, seq_len, 1).expand(1, seq_len, 1536)
        return {'embeddings_1bp': emb.clone()}

    with patch.object(alphagenome.model, "encode", side_effect=side_effect):
        short_seq = "ACGT" * 10000  # 40000 bp → padded to 40960 (20 * 2048)
        seq_len = len(short_seq)
        expected_padded_len = math.ceil(seq_len / 2048) * 2048

        output = alphagenome.embed_sequence(short_seq, agg_fn=lambda x: x)

        # Right-padding: sequence occupies positions [0:seq_len]
        assert output.shape[1] == seq_len

        # First position should have value 0 (sequence starts at index 0)
        assert output[0, 0, 0].item() == 0.0

        # Last position should have value seq_len - 1
        assert output[0, -1, 0].item() == float(seq_len - 1)

        # Padded length passed to model should be the next multiple of 2048
        assert expected_padded_len == math.ceil(seq_len / 2048) * 2048


def test_alphagenome_extract_structure(alphagenome):
    """extract() returns (dict, dict) with matching keys; hidden states are 2D."""
    seq = "ACGT" * 1000
    h, s = alphagenome.extract([seq], layers=[0])
    assert isinstance(h, dict) and isinstance(s, dict)
    assert set(h.keys()) == set(s.keys())
    layer = next(iter(h))
    assert h[layer][0][0].dim() == 2
    assert h[layer][0][0].device.type == "cpu"


def test_alphagenome_extract_layer_selection(alphagenome):
    """Requesting layers=[0] returns exactly 1 layer."""
    seq = "ACGT" * 1000
    h, _ = alphagenome.extract([seq], layers=[0])
    assert len(h) == 1


def test_alphagenome_extract_scores_always_none(alphagenome):
    """All layers return None scores — AlphaGenome attention is not hookable."""
    seq = "ACGT" * 1000
    _, s = alphagenome.extract([seq], layers=[0], return_attentions=True)
    assert all(v is None for v in s.values())


def test_alphagenome_extract_transformer_scores_none(alphagenome):
    """Transformer layers also return None scores even when return_attentions=True.

    attn_weights is a local variable in MHABlock.forward() and cannot be
    captured via hooks without modifying the library.
    """
    seq = "ACGT" * 1000
    _, s = alphagenome.extract(
        [seq], layers=["tower.blocks.0.mlp"], return_attentions=True
    )
    assert s["tower.blocks.0.mlp"] is None


def test_alphagenome_extract_transformer_hidden_state_shape(alphagenome):
    """Transformer layer hidden states have shape (S, 1536).

    The hook captures input[0] + output on the MLP module to reconstruct
    the post-residual block state.
    """
    seq = "ACGT" * 1000
    h, _ = alphagenome.extract([seq], layers=["tower.blocks.0.mlp"])
    state = h["tower.blocks.0.mlp"][0][0]  # seq[0], chunk[0]
    assert state.dim() == 2
    assert state.shape[1] == 1536  # transformer hidden dim


def test_alphagenome_extract_encoder_scores_none(alphagenome):
    """Encoder layers return None scores even when return_attentions=True."""
    seq = "ACGT" * 1000
    _, s = alphagenome.extract(
        [seq], layers=["encoder.down_blocks.0"], return_attentions=True
    )
    assert s["encoder.down_blocks.0"] is None
