import pytest

from unittest.mock import patch

pytest.importorskip("torch")

import torch

from tests.model_utils import (
    assert_pooled_batch_matches_single,
    assert_raw_batch_matches_single,
    embed_one,
)
from mrna_bench.models.aido_dna import AIDODNA


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def aidomodel(device) -> AIDODNA:
    """Get AIDODNA model."""
    model = AIDODNA("AIDO.DNA-300M", device, "eager")
    model.set_inference_mode()
    return model


def test_aido_forward(aidomodel):
    """Test AIDODNA initialization and forward pass."""

    out = embed_one(aidomodel, "ATGATG")
    assert out.shape == (1, 1024)


def test_aido_dna_rna_equivalent(aidomodel):
    """AIDO.DNA is DNA-trained; RNA (U) and DNA (T) inputs must match.

    The wrapper converts U->T so RNA sequences aren't fed the distinct,
    untrained 'U' token.
    """
    import torch
    dna = embed_one(aidomodel, "ATGCATGCATGC").cpu()
    rna = embed_one(aidomodel, "AUGCAUGCAUGC").cpu()
    assert torch.equal(dna, rna)


def test_aido_forward_long(aidomodel):
    """Test AIDODNA forward pass."""
    long_sequence = "ATGC" * 257  # 1028 nucleotides
    out = embed_one(aidomodel, long_sequence)
    assert out.shape == (1, 1024)


@torch.no_grad()
def test_aido_embed_batch_ragged(aidomodel):
    """Test ragged batches match individual embeddings."""
    sequences = [
        "ACTG" * 5,
        "ACTG" * 50,
        "ACTG" * 200,
        "ACTG" * 10,
    ]

    assert_pooled_batch_matches_single(aidomodel, sequences)


def test_aido_excludes_special_tokens(aidomodel):
    """Verify pooling mask excludes CLS and SEP special tokens."""
    with patch.object(aidomodel, "model") as mock_model:
        mock_model.return_value.last_hidden_state = torch.ones(
            1, 8, 1024, device=aidomodel.device
        )

        _, mask = aidomodel._forward_chunks(["ATGATG"])

        # Mask should be 0 at CLS (pos 0) and SEP (last real token pos)
        assert mask[0, 0].item() == 0  # CLS
        assert mask[0, -1].item() == 0  # SEP (last padding pos, but seq fills all)
        # Nucleotide positions (1 to seq_len-1) should be 1
        assert mask[0, 1:-1].sum().item() == 6


@torch.no_grad()
def test_aido_embed_ragged_agg(aidomodel):
    """Test embed with identity agg_fn returns per-token embeddings (ragged)."""
    seqs = ["ATGATG", "GCGCGCGCGCGC"]
    out = aidomodel.embed(seqs, agg_fn=lambda x, **kwargs: x)
    assert_raw_batch_matches_single(aidomodel, seqs, out)
    assert out[0].dim() == 2  # (num_tokens, hidden_dim)
    assert out[1].dim() == 2
    assert out[0].shape[0] != out[1].shape[0]  # ragged: different token counts
    assert out[0].shape[1] == out[1].shape[1]  # same hidden dim


def test_aido_gradient_flow(aidomodel):
    """Test that gradients can flow through the model."""
    aidomodel.set_train_mode()

    out = aidomodel.embed(["ATGATG"])
    assert out[0].requires_grad, "Output should require gradients"

    loss = torch.stack(out).sum()
    loss.backward()

    has_grad = False
    for param in aidomodel.model.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break

    assert has_grad, "No gradients flowed to model parameters"
    aidomodel.set_inference_mode()


def test_aido_extract_structure(aidomodel):
    """extract() returns (dict, dict) with matching keys; hidden states are 2D."""
    h, s = aidomodel.extract(["ATGATG"], layers=[0])
    assert isinstance(h, dict) and isinstance(s, dict)
    assert set(h.keys()) == set(s.keys())
    layer = next(iter(h))
    assert h[layer][0][0].dim() == 2
    assert h[layer][0][0].device.type == "cpu"


def test_aido_extract_layer_selection(aidomodel):
    """Requesting layers=[0] returns exactly 1 layer."""
    h, _ = aidomodel.extract(["ATGATG"], layers=[0])
    assert len(h) == 1


def test_aido_extract_attention_weights(aidomodel):
    """return_attentions=True yields (H, T, T) tensors with rows summing to 1."""
    h, s = aidomodel.extract(["ATGATG"], layers=[0], return_attentions=True)
    layer = next(iter(s))
    attn = s[layer]
    assert attn is not None
    w = attn[0][0]
    assert w.dim() == 3
    assert w.shape[1] == w.shape[2]
    assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-6)
