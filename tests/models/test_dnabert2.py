import math
import pytest

pytest.importorskip("torch")

import torch

from tests.model_utils import (
    assert_pooled_batch_matches_single,
    assert_raw_batch_matches_single,
    embed_one,
)
from mrna_bench.models.dnabert2 import DNABERT2


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def dnabert2(device) -> DNABERT2:
    """Get DNABERT2 model."""
    model = DNABERT2("DNABERT2", device, "eager")
    model.set_inference_mode()
    return model


def test_dnabert2_forward(dnabert2):
    """Test DNABERT2 forward pass."""
    out = embed_one(dnabert2, "ATGATG")
    assert out.shape == (1, 768)


def test_dnabert2_masked_marginal_llr(dnabert2):
    """DNABERT2 scores the changed BPE token span."""
    score = dnabert2.masked_marginal_llr(
        ["ATGATGATGATG"],
        ["ATGACGATGATG"],
    )[0]
    assert math.isfinite(score)


@torch.no_grad()
def test_dnabert2_embed_batch_ragged(dnabert2):
    """Test ragged batches match individual embeddings."""
    sequences = [
        "ATGATG" * 10,
        "ATGATG" * 50,
        "ATGATG" * 100,
    ]

    assert_pooled_batch_matches_single(dnabert2, sequences)


@torch.no_grad()
def test_dnabert2_excludes_special_tokens(dnabert2):
    """Test that CLS and SEP tokens are excluded from pooling."""
    text = "ATGATG" * 20

    toks = dnabert2.tokenizer([text], return_tensors="pt", padding=True)
    toks = toks.to(dnabert2.device)
    hidden_states = dnabert2.model(**toks)[0]

    # Mean over ALL tokens (including CLS/SEP)
    mean_all = hidden_states.mean(dim=1).cpu()

    # Mean excluding first and last (CLS/SEP)
    mean_no_special = hidden_states[:, 1:-1, :].mean(dim=1).cpu()

    output = embed_one(dnabert2, text).cpu()

    assert torch.equal(output, mean_no_special), \
        "Output should exclude CLS/SEP tokens"
    assert not torch.allclose(output, mean_all, atol=1e-5), \
        "Output should differ from mean including special tokens"


@torch.no_grad()
def test_dnabert2_embed_ragged_agg(dnabert2):
    """Test embed with identity agg_fn returns per-token embeddings (ragged)."""
    seqs = ["ATGATG", "GCGCGCGCGCGC"]
    out = dnabert2.embed(seqs, agg_fn=lambda x, **kwargs: x)
    assert_raw_batch_matches_single(dnabert2, seqs, out)
    assert out[0].dim() == 2  # (num_tokens, hidden_dim)
    assert out[1].dim() == 2
    assert out[0].shape[0] != out[1].shape[0]  # ragged: different token counts
    assert out[0].shape[1] == out[1].shape[1]  # same hidden dim


def test_dnabert2_gradient_flow(dnabert2):
    """Test that gradients can flow through the model."""
    dnabert2.set_train_mode()

    out = dnabert2.embed(["ATGATG"])
    assert out[0].requires_grad, "Output should require gradients"

    loss = torch.stack(out).sum()
    loss.backward()

    has_grad = False
    for param in dnabert2.model.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break

    assert has_grad, "No gradients flowed to model parameters"
    dnabert2.set_inference_mode()
    

def test_dnabert_extract_structure(dnabert2):
    """extract() returns (dict, dict) with matching keys; hidden states are 2D."""
    h, s = dnabert2.extract(["ATGATG"], layers=[0])
    assert isinstance(h, dict) and isinstance(s, dict)
    assert set(h.keys()) == set(s.keys())
    layer = next(iter(h))
    assert h[layer][0][0].dim() == 2
    assert h[layer][0][0].device.type == "cpu"


def test_dnabert_extract_layer_selection(dnabert2):
    """Requesting layers=[0] returns exactly 1 layer."""
    h, _ = dnabert2.extract(["ATGATG"], layers=[0])
    assert len(h) == 1


def test_dnabert_extract_attention_weights(dnabert2):
    """return_attentions=True yields (H, T, T) tensors with rows summing to 1."""
    h, s = dnabert2.extract(["ATGATG"], layers=[0], return_attentions=True)
    layer = next(iter(s))
    attn = s[layer]
    assert attn is not None
    w = attn[0][0]
    assert w.dim() == 3
    assert w.shape[1] == w.shape[2]
    assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-6)
