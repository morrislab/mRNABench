import pytest

from unittest.mock import patch

pytest.importorskip("torch")

import torch

from tests.model_utils import (
    assert_pooled_batch_matches_single,
    assert_raw_batch_matches_single,
    embed_one,
)
from mrna_bench.models.gb_rna import GBRNA


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def gbmodel(device) -> GBRNA:
    """Get GBRNA model."""
    model = GBRNA("GB.RNA-650M", device, "eager")
    model.set_inference_mode()
    return model


def test_gb_forward(gbmodel):
    """Test GBRNA initialization and forward pass."""

    out = embed_one(gbmodel, "ATGATG")
    assert out.shape == (1, 1280)


def test_gb_dna_rna_equivalent(gbmodel):
    """GB.RNA normalizes RNA notation to the upstream T-token convention."""
    import torch
    dna = embed_one(gbmodel, "ATGCATGCATGC").cpu()
    rna = embed_one(gbmodel, "AUGCAUGCAUGC").cpu()
    assert torch.equal(dna, rna)


def test_gb_forward_long(gbmodel):
    """Test GBRNA forward pass."""
    long_sequence = "ATGC" * 257  # 1028 nucleotides
    out = embed_one(gbmodel, long_sequence)
    assert out.shape == (1, 1280)


@torch.no_grad()
def test_gb_embed_batch_ragged(gbmodel):
    """Test ragged batches match individual embeddings."""
    sequences = [
        "ACTG" * 5,
        "ACTG" * 50,
        "ACTG" * 200,
        "ACTG" * 10,
    ]

    assert_pooled_batch_matches_single(gbmodel, sequences)


def test_gb_excludes_special_tokens(gbmodel):
    """Verify pooling mask excludes CLS and SEP special tokens."""
    with patch.object(gbmodel, "model") as mock_model:
        mock_model.return_value.last_hidden_state = torch.ones(
            1, 8, 1280, device=gbmodel.device
        )

        _, mask = gbmodel._forward_chunks(["ATGATG"])

        # Mask should be 0 at CLS (pos 0) and SEP (last real token pos)
        assert mask[0, 0].item() == 0  # CLS
        assert mask[0, -1].item() == 0  # SEP (last padding pos, but seq fills all)
        # Nucleotide positions (1 to seq_len-1) should be 1
        assert mask[0, 1:-1].sum().item() == 6


@torch.no_grad()
def test_gb_embed_ragged_agg(gbmodel):
    """Test embed with identity agg_fn returns per-token embeddings (ragged)."""
    seqs = ["ATGATG", "GCGCGCGCGCGC"]
    out = gbmodel.embed(seqs, agg_fn=lambda x, **kwargs: x)
    assert_raw_batch_matches_single(gbmodel, seqs, out)
    assert out[0].dim() == 2  # (num_tokens, hidden_dim)
    assert out[1].dim() == 2
    assert out[0].shape[0] != out[1].shape[0]  # ragged: different token counts
    assert out[0].shape[1] == out[1].shape[1]  # same hidden dim


def test_gb_gradient_flow(gbmodel):
    """Test that gradients can flow through the model."""
    gbmodel.set_train_mode()

    out = gbmodel.embed(["ATGATG"])
    assert out[0].requires_grad, "Output should require gradients"

    loss = torch.stack(out).sum()
    loss.backward()

    has_grad = False
    for param in gbmodel.model.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break

    assert has_grad, "No gradients flowed to model parameters"
    gbmodel.set_inference_mode()


def test_gb_extract_structure(gbmodel):
    """extract() returns (dict, dict) with matching keys; hidden states are 2D."""
    h, s = gbmodel.extract(["ATGATG"], layers=[0])
    assert isinstance(h, dict) and isinstance(s, dict)
    assert set(h.keys()) == set(s.keys())
    layer = next(iter(h))
    assert h[layer][0][0].dim() == 2
    assert h[layer][0][0].device.type == "cpu"


def test_gb_extract_layer_selection(gbmodel):
    """Requesting layers=[0] returns exactly 1 layer."""
    h, _ = gbmodel.extract(["ATGATG"], layers=[0])
    assert len(h) == 1


def test_gb_extract_attention_weights(gbmodel):
    """return_attentions=True yields (H, T, T) tensors with rows summing to 1."""
    h, s = gbmodel.extract(["ATGATG"], layers=[0], return_attentions=True)
    layer = next(iter(s))
    attn = s[layer]
    assert attn is not None
    w = attn[0][0]
    assert w.dim() == 3
    assert w.shape[1] == w.shape[2]
    assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-6)
