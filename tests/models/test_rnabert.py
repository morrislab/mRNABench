import pytest

pytest.importorskip("torch")

import torch

from tests.model_utils import (
    assert_pooled_batch_matches_single,
    assert_raw_batch_matches_single,
    embed_one,
)
from mrna_bench.models.rnabert import RNABERT


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def model(device) -> RNABERT:
    """Get RNABERT model."""
    model = RNABERT("RNABERT", device, "eager")
    model.set_inference_mode()
    return model


def test_rnabert_forward(model):
    """Test RNABERT forward pass."""
    out = embed_one(model, "ATGATG")
    assert out.shape == (1, 120)


@torch.no_grad()
def test_rnabert_converts_t_to_u(model):
    """Test that RNABERT converts T->U for proper tokenization."""
    dna_seq = "ATGATGATG"
    rna_seq = "AUGAUGAUG"

    dna_output = embed_one(model, dna_seq).cpu()
    rna_output = embed_one(model, rna_seq).cpu()

    assert torch.equal(dna_output, rna_output), \
        "DNA (T) and RNA (U) sequences should produce identical embeddings"


@torch.no_grad()
def test_rnabert_embed_batch_ragged(model):
    """Test ragged batches match individual embeddings."""
    sequences = [
        "ATGATG" * 10,
        "ATGATG" * 50,
        "ATGATG" * 100,
    ]

    assert_pooled_batch_matches_single(model, sequences)


@torch.no_grad()
def test_rnabert_excludes_special_tokens(model):
    """RNABERT's tokenizer adds no CLS/EOS, so all tokens are real content.

    The pooled embedding should therefore equal the mean over all (non-pad)
    token positions, not a CLS/EOS-trimmed subset.
    """
    text = "AUGAUG" * 20

    toks = model.tokenizer([text], return_tensors="pt", padding=True)
    toks = toks.to(model.device)
    hidden_states = model.model(**toks).last_hidden_state

    # No special tokens are added, so every position is real content.
    special_ids = model.tokenizer.all_special_ids
    assert not any(
        tok_id in special_ids for tok_id in toks["input_ids"][0].tolist()
    ), "RNABERT tokenizer unexpectedly added special tokens"

    mean_all = hidden_states.mean(dim=1).cpu()
    output = embed_one(model, text).cpu()

    assert torch.equal(output, mean_all), \
        "Output should pool over all real tokens (RNABERT has no CLS/EOS)"


@torch.no_grad()
def test_rnabert_embed_ragged_agg(model):
    """Test embed with identity agg_fn returns per-token embeddings (ragged)."""
    seqs = ["ATGATG", "GCGCGCGCGCGC"]
    out = model.embed(seqs, agg_fn=lambda x, **kwargs: x)
    assert_raw_batch_matches_single(model, seqs, out)
    assert out[0].dim() == 2  # (num_tokens, hidden_dim)
    assert out[1].dim() == 2
    assert out[0].shape[0] != out[1].shape[0]  # ragged: different token counts
    assert out[0].shape[1] == out[1].shape[1]  # same hidden dim


def test_rnabert_gradient_flow(model):
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


def test_rnabert_extract_structure(model):
    """extract() returns (dict, dict) with matching keys; hidden states are 2D."""
    h, s = model.extract(["ATGATG"], layers=[0])
    assert isinstance(h, dict) and isinstance(s, dict)
    assert set(h.keys()) == set(s.keys())
    layer = next(iter(h))
    assert h[layer][0][0].dim() == 2
    assert h[layer][0][0].device.type == "cpu"


def test_rnabert_extract_layer_selection(model):
    """Requesting layers=[0] returns exactly 1 layer."""
    h, _ = model.extract(["ATGATG"], layers=[0])
    assert len(h) == 1


def test_rnabert_extract_attention_weights(model):
    """return_attentions=True yields (H, T, T) tensors with rows summing to 1."""
    h, s = model.extract(["ATGATG"], layers=[0], return_attentions=True)
    layer = next(iter(s))
    attn = s[layer]
    assert attn is not None
    w = attn[0][0]
    assert w.dim() == 3
    assert w.shape[1] == w.shape[2]
    assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-6)
