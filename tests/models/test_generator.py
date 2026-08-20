import math

import pytest

from unittest.mock import Mock, patch

pytest.importorskip("torch")
import torch

from tests.model_utils import (
    assert_pooled_batch_matches_single,
    assert_raw_batch_matches_single,
)
from mrna_bench.models.generator import GENERator


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def model(device) -> GENERator:
    """Get GENERator model."""
    model = GENERator("v2-eukaryote-3b-base", device, "eager")
    model.set_inference_mode()
    return model


def test_generator_forward(model):
    """Test GENERator forward pass."""
    out = model.embed(["ATGATG"])
    assert len(out) == 1 and out[0].shape == (3072,)


def test_generator_likelihood_tokenization_pads_trailing_bases():
    """Likelihood scoring preserves a final partial k-mer."""
    model = GENERator.__new__(GENERator)
    model.k = 3
    model.tokenizer = Mock(return_value={"input_ids": torch.tensor([[1]])})

    model._tokenize_for_logits("ATGA")

    model.tokenizer.assert_called_once_with(
        "ATGAAA",
        return_tensors="pt",
        add_special_tokens=True,
    )


def test_generator_causal_likelihood(model):
    """GENERator exposes its native autoregressive head."""
    sequence = "ATGATGATGATG"
    assert model.supports("causal_likelihood")
    assert model.logits([sequence])[0].ndim == 2
    assert math.isfinite(model.sequence_score([sequence])[0])


def test_generator_embed_batch(model):
    """Test pooled ragged batches match individual embeddings."""
    assert_pooled_batch_matches_single(
        model,
        ["ATGATG", "GCGCGCGCGCGC"],
    )


def test_generator_excludes_special_tokens(model):
    """Verify pooling excludes BOS while retaining the sequence k-mer."""
    with patch.object(model, "model") as mock_model:
        mock_model.return_value.hidden_states = [
            torch.ones(1, 2, 3072, device=model.device)
        ]

        _, mask = model._forward_chunks(["ATGATG"])

        assert mask[0, 0].item() == 0  # BOS
        assert mask[0, 1:].sum().item() == 1


@torch.no_grad()
def test_generator_embed_ragged_agg(model):
    """Test embed with identity agg_fn returns per-token embeddings (ragged)."""
    seqs = ["ATGATG", "GCGCGCGCGCGC"]
    out = model.embed(seqs, agg_fn=lambda x, **kwargs: x)
    assert_raw_batch_matches_single(model, seqs, out)
    assert out[0].dim() == 2  # (num_tokens, hidden_dim)
    assert out[1].dim() == 2
    assert out[0].shape[0] != out[1].shape[0]  # ragged: different token counts
    assert out[0].shape[1] == out[1].shape[1]  # same hidden dim


def test_generator_gradient_flow(model):
    """Test that gradients can flow through the model."""
    model.set_train_mode()

    out = model.embed(["ATGATG"])
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
    

def test_generator_extract_structure(model):
    """extract() returns (dict, dict) with matching keys; hidden states are 2D."""
    h, s = model.extract(["ATGATG"], layers=[0])
    assert isinstance(h, dict) and isinstance(s, dict)
    assert set(h.keys()) == set(s.keys())
    layer = next(iter(h))
    assert h[layer][0][0].dim() == 2
    assert h[layer][0][0].device.type == "cpu"


def test_generator_extract_layer_selection(model):
    """Requesting layers=[0] returns exactly 1 layer."""
    h, _ = model.extract(["ATGATG"], layers=[0])
    assert len(h) == 1


def test_generator_extract_attention_weights(model):
    """return_attentions=True with eager yields (H, T, T) tensors."""
    h, s = model.extract(["ATGATG"], layers=[0], return_attentions=True)
    layer = next(iter(s))
    attn = s[layer]
    assert attn is not None
    w = attn[0][0]
    assert w.dim() == 3
    assert w.shape[1] == w.shape[2]
    assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-6)
