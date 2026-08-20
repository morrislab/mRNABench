import math
import pytest

pytest.importorskip("torch")

import torch

from tests.model_utils import (
    assert_pooled_batch_matches_single,
    assert_raw_batch_matches_single,
    embed_one,
)
from mrna_bench.models.dnabert import DNABERT


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def dnabert(device) -> DNABERT:
    """Get original k-mer DNABERT model (6-mer)."""
    model = DNABERT("DNABERT-6mer", device, "eager")
    model.set_inference_mode()
    return model


def test_dnabert_kmer_short_names():
    """Short names preserve the original hyphenated identifiers."""
    assert DNABERT.get_model_short_name("DNABERT-3mer") == "dnabert-3mer"
    assert DNABERT.get_model_short_name("DNABERT-4mer") == "dnabert-4mer"
    assert DNABERT.get_model_short_name("DNABERT-5mer") == "dnabert-5mer"
    assert DNABERT.get_model_short_name("DNABERT-6mer") == "dnabert-6mer"


def test_dnabert_scoring_chunks_preserve_boundary_kmers():
    """Likelihood chunks cover every overlapping k-mer exactly once."""
    model = DNABERT.__new__(DNABERT)
    model.k = 3
    model.max_kmer_tokens = 4
    model.max_chunk_length = 6
    sequence = "ABCDEFGHIJ"

    chunks = model._score_chunks(sequence, None, None)
    kmers = [
        chunk[index:index + model.k]
        for chunk, _, _ in chunks
        for index in range(len(chunk) - model.k + 1)
    ]

    assert kmers == [
        sequence[index:index + model.k]
        for index in range(len(sequence) - model.k + 1)
    ]


def test_dnabert_kmer_forward(dnabert):
    """Test k-mer DNABERT forward pass."""
    out = embed_one(dnabert, "ATGATGATGATG")
    assert out.shape == (1, 768)


def test_dnabert_kmer_k_value(dnabert):
    """The 6-mer model parses k=6 from its version name."""
    assert dnabert.k == 6


def test_dnabert_masked_marginal_llr(dnabert):
    """DNABERT scores every overlapping k-mer changed by the substitution."""
    score = dnabert.masked_marginal_llr(
        ["ATGATGATGATG"],
        ["ATGACGATGATG"],
    )[0]
    assert math.isfinite(score)


@torch.no_grad()
def test_dnabert_kmer_embed_batch_ragged(dnabert):
    """Test ragged batches match individual embeddings."""
    sequences = [
        "ATGATG" * 10,
        "ATGATG" * 50,
        "ATGATG" * 100,
    ]

    assert_pooled_batch_matches_single(dnabert, sequences)


@torch.no_grad()
def test_dnabert_kmer_excludes_special_tokens(dnabert):
    """Test that CLS and SEP tokens are excluded from pooling."""
    text = "ATGATG" * 20
    kmer = dnabert._seq_to_kmers(text)

    toks = dnabert.tokenizer([kmer], return_tensors="pt", padding=True)
    toks = toks.to(dnabert.device)
    hidden_states = dnabert.model(**toks)[0]

    mean_all = hidden_states.mean(dim=1).cpu()
    mean_no_special = hidden_states[:, 1:-1, :].mean(dim=1).cpu()

    output = embed_one(dnabert, text).cpu()

    assert torch.allclose(output, mean_no_special, atol=1e-5), \
        "Output should exclude CLS/SEP tokens"
    assert not torch.allclose(output, mean_all, atol=1e-5), \
        "Output should differ from mean including special tokens"


@torch.no_grad()
def test_dnabert_kmer_embed_ragged_agg(dnabert):
    """Test embed with identity agg_fn returns per-token embeddings (ragged)."""
    seqs = ["ATGATGATG", "GCGCGCGCGCGCGCGC"]
    out = dnabert.embed(seqs, agg_fn=lambda x, **kwargs: x)
    assert_raw_batch_matches_single(dnabert, seqs, out)
    assert out[0].dim() == 2  # (num_tokens, hidden_dim)
    assert out[1].dim() == 2
    assert out[0].shape[0] != out[1].shape[0]  # ragged: different token counts
    assert out[0].shape[1] == out[1].shape[1] == 768  # same hidden dim


def test_dnabert_kmer_gradient_flow(dnabert):
    """Test that gradients can flow through the model."""
    dnabert.set_train_mode()

    out = dnabert.embed(["ATGATGATGATG"])
    assert out[0].requires_grad, "Output should require gradients"

    loss = torch.stack(out).sum()
    loss.backward()

    has_grad = False
    for param in dnabert.model.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break

    assert has_grad, "No gradients flowed to model parameters"
    dnabert.set_inference_mode()


def test_dnabert_kmer_extract_structure(dnabert):
    """extract() returns (dict, dict) with matching keys; hidden states are 2D."""
    h, s = dnabert.extract(["ATGATGATGATG"], layers=[0])
    assert isinstance(h, dict) and isinstance(s, dict)
    assert set(h.keys()) == set(s.keys())
    layer = next(iter(h))
    assert h[layer][0][0].dim() == 2
    assert h[layer][0][0].device.type == "cpu"


def test_dnabert_kmer_extract_layer_selection(dnabert):
    """Requesting layers=[0] returns exactly 1 layer."""
    h, _ = dnabert.extract(["ATGATGATGATG"], layers=[0])
    assert len(h) == 1


def test_dnabert_kmer_extract_attention_weights(dnabert):
    """return_attentions=True yields (H, T, T) tensors with rows summing to 1."""
    h, s = dnabert.extract(
        ["ATGATGATGATG"], layers=[0], return_attentions=True
    )
    layer = next(iter(s))
    attn = s[layer]
    assert attn is not None
    w = attn[0][0]
    assert w.dim() == 3
    assert w.shape[1] == w.shape[2]
    assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-6)
