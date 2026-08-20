import math

import pytest

pytest.importorskip("torch")
import torch

from tests.model_utils import (
    assert_pooled_batch_matches_single,
    assert_raw_batch_matches_single,
)
from mrna_bench.models.carbon import Carbon


HIDDEN_DIM = 1024  # Carbon-500M


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def model(device) -> Carbon:
    """Get Carbon-500M model (smallest version)."""
    model = Carbon("Carbon-500M", device, "eager")
    model.set_inference_mode()
    return model


def test_carbon_forward(model):
    """Test Carbon forward pass."""
    out = model.embed(["ATGATGATGATG"])
    assert len(out) == 1 and out[0].shape == (HIDDEN_DIM,)


def test_carbon_causal_likelihood(model):
    """Carbon exposes its native autoregressive head."""
    sequence = "ATGATGATGATG"
    assert model.supports("causal_likelihood")
    assert model.logits([sequence])[0].ndim == 2
    assert math.isfinite(model.sequence_score([sequence])[0])


def test_carbon_embed_batch(model):
    """Test pooled ragged batches match individual embeddings."""
    assert_pooled_batch_matches_single(
        model,
        ["ATGATG", "GCGCGCGCGCGCGCGCGC"],
    )


def test_carbon_excludes_dna_tags(model):
    """Verify pooling mask excludes the <dna>/</dna> delimiter tokens."""
    _, mask = model._forward_chunks(["ATGCGCATGCGC"])
    # First token is <dna>, last is </dna>; both must be excluded.
    assert mask[0, 0].item() == 0
    assert mask[0, -1].item() == 0
    # The interior 6-mer tokens are kept.
    assert mask[0, 1:-1].sum().item() > 0


@torch.no_grad()
def test_carbon_embed_ragged_agg(model):
    """Test embed with identity agg_fn returns per-token embeddings (ragged)."""
    seqs = ["ATGATG", "GCGCGCGCGCGCGCGCGC"]
    out = model.embed(seqs, agg_fn=lambda x, **kwargs: x)
    assert_raw_batch_matches_single(model, seqs, out)
    assert out[0].dim() == 2  # (num_tokens, hidden_dim)
    assert out[1].dim() == 2
    assert out[0].shape[0] != out[1].shape[0]  # ragged: different token counts
    assert out[0].shape[1] == out[1].shape[1]  # same hidden dim


def test_carbon_gradient_flow(model):
    """Test that gradients can flow through the model."""
    model.set_train_mode()

    out = model.embed(["ATGATGATGATG"])
    assert out[0].requires_grad, "Output should require gradients"

    loss = torch.stack(out).sum()
    loss.backward()

    has_grad = False
    for param in model.model.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break

    assert has_grad, "No gradients flowed to model parameters"
    model.set_inference_mode()


def test_carbon_extract_structure(model):
    """extract() returns (dict, dict) with matching keys; hidden states are 2D."""
    h, s = model.extract(["ATGATGATGATG"], layers=[0])
    assert isinstance(h, dict) and isinstance(s, dict)
    assert set(h.keys()) == set(s.keys())
    layer = next(iter(h))
    assert h[layer][0][0].dim() == 2
    assert h[layer][0][0].device.type == "cpu"


def test_carbon_extract_layer_selection(model):
    """Requesting layers=[0] returns exactly 1 layer."""
    h, _ = model.extract(["ATGATGATGATG"], layers=[0])
    assert len(h) == 1


def test_carbon_extract_attention_weights(model):
    """return_attentions=True with eager yields (H, T, T) tensors."""
    h, s = model.extract(["ATGATGATGATG"], layers=[0], return_attentions=True)
    layer = next(iter(s))
    attn = s[layer]
    assert attn is not None
    w = attn[0][0]
    assert w.dim() == 3
    assert w.shape[1] == w.shape[2]
    assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-6)
