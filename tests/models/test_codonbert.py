import math
import pytest
from unittest.mock import patch

import numpy as np

pytest.importorskip("torch")
import torch

from tests.model_utils import (
    assert_pooled_batch_matches_single,
    assert_raw_batch_matches_single,
    embed_one,
)
from mrna_bench.models.codonbert import CodonBERT


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def model(device) -> CodonBERT:
    """Get CodonBERT model."""
    model = CodonBERT("CodonBERT", device, "eager")
    model.set_inference_mode()
    return model


def make_cds(seq: str) -> np.ndarray:
    """Make CDS array marking every codon start (assumes pure CDS input)."""
    arr = np.zeros(len(seq), dtype=int)
    arr[::3] = 1
    return arr


def test_codonbert_requires_cds(model):
    """Test that CodonBERT raises error when CDS is not provided."""
    with pytest.raises(ValueError, match="CodonBERT requires cds"):
        model.embed(["ATGATG"])


def test_codonbert_pseudo_likelihood_uses_cds(model):
    """CodonBERT excludes UTR sequence before masked scoring."""
    sequence = "CCC" + "ATGATG" + "GGG"
    cds = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0])
    coding = "ATGATG"
    coding_cds = make_cds(coding)

    with pytest.raises(ValueError, match="requires cds"):
        model.sequence_score([sequence])
    transcript_logits = model.logits([sequence], cds=[cds])[0]
    coding_logits = model.logits([coding], cds=[coding_cds])[0]
    assert torch.allclose(transcript_logits, coding_logits)
    assert math.isfinite(model.sequence_score([sequence], cds=[cds])[0])


def test_codonbert_masked_marginal_llr(model):
    """CodonBERT masks the changed codon rather than a nucleotide token."""
    reference = "CCCATGATGGGG"
    alternate = "CCCACGATGGGG"
    cds = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0])
    score = model.masked_marginal_llr(
        [reference],
        [alternate],
        cds=[cds],
    )[0]
    assert math.isfinite(score)


def test_codonbert_forward(model):
    """Test CodonBERT forward pass."""
    seq = "ATGATG"
    out = embed_one(model, seq, cds=make_cds(seq))
    assert out.shape == (1, 768)


@torch.no_grad()
def test_codonbert_converts_t_to_u(model):
    """Test that CodonBERT converts T->U for proper tokenization."""
    dna_seq = "ATGATGATG"
    rna_seq = "AUGAUGAUG"
    cds = make_cds(dna_seq)

    dna_output = embed_one(model, dna_seq, cds=cds).cpu()
    rna_output = embed_one(model, rna_seq, cds=make_cds(rna_seq)).cpu()

    assert torch.allclose(dna_output, rna_output, atol=1e-5), \
        "DNA (T) and RNA (U) sequences should produce identical embeddings"


@torch.no_grad()
def test_codonbert_embed_batch_ragged(model):
    """Test ragged batches match individual embeddings."""
    sequences = [
        "ATG" * 10,
        "ATG" * 50,
        "ATG" * 100,
        "ATG" * 20,
    ]
    cds = [make_cds(s) for s in sequences]

    assert_pooled_batch_matches_single(
        model,
        sequences,
        cds=cds,
    )


@torch.no_grad()
def test_codonbert_excludes_special_tokens(model):
    """Test that CLS and SEP tokens are excluded from pooling."""
    text = "AUG" * 20  # 20 codons in RNA notation

    # Tokenize using the same codon-format that _forward_chunks uses
    codon_text = model._nt_to_codons(text)
    toks = model.tokenizer([codon_text], return_tensors="pt", padding=True)
    toks = toks.to(model.device)
    hidden_states = model.model(**toks).last_hidden_state

    mean_all = hidden_states.mean(dim=1).cpu()
    mean_no_special = hidden_states[:, 1:-1, :].mean(dim=1).cpu()

    cds = make_cds(text)
    output = embed_one(model, text, cds=cds).cpu()

    assert torch.allclose(output, mean_no_special, atol=1e-5), \
        "Output should exclude CLS/SEP tokens"
    assert not torch.allclose(output, mean_all, atol=1e-5), \
        "Output should differ from mean including special tokens"


@torch.no_grad()
def test_codonbert_single_codon(model):
    """Test embedding a single codon."""
    seq = "ATG"
    output = embed_one(model, seq, cds=make_cds(seq)).cpu()
    assert output.shape == (1, 768)
    assert not torch.isnan(output).any()


@torch.no_grad()
def test_codonbert_cds_slice(model):
    """Test that CDS extraction correctly slices the CDS region."""
    input_seq = "A" * 30 + "T" * 30 + "G" * 40
    cds = np.array([0] * 30 + [1, 0, 0] * 10 + [0] * 40)

    with patch.object(
        model,
        "_forward_chunks",
        wraps=model._forward_chunks
    ) as mock:
        mock.return_value = (
            torch.zeros(1, 10, 768, device=model.device),
            torch.ones(1, 10, device=model.device)
        )
        model.embed([input_seq], cds=[cds])
        chunks = mock.call_args[0][0]
        # CDS region is 30 U's (converted from T's)
        assert chunks[0] == "U" * 30


@torch.no_grad()
def test_codonbert_max_length_boundary(model):
    """Test sequence at max_length boundary."""
    max_nt = (model.max_length - 2) * 3
    seq_at_boundary = "ATG" * (max_nt // 3)
    cds_at = make_cds(seq_at_boundary)
    output1 = embed_one(model, seq_at_boundary, cds=cds_at).cpu()
    assert output1.shape == (1, 768)

    seq_over_boundary = seq_at_boundary + "ATG"
    cds_over = make_cds(seq_over_boundary)
    output2 = embed_one(model, seq_over_boundary, cds=cds_over).cpu()
    assert output2.shape == (1, 768)

    assert not torch.allclose(output1, output2, atol=1e-5)


@torch.no_grad()
def test_codonbert_get_cds_full(model):
    """Test get_cds extracts the correct CDS region."""
    sequence = "CCUAUGCCG"
    cds = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0])
    cds_seq = model.get_cds(sequence, cds)
    assert cds_seq == "AUG"


@torch.no_grad()
def test_codonbert_get_cds_missing(model):
    """Test get_cds when no CDS is marked."""
    sequence = "CCGATGCC"  # 8 chars, truncated to 6
    cds = np.zeros(len(sequence), dtype=int)
    with pytest.warns(UserWarning, match="No CDS found"):
        cds_seq = model.get_cds(sequence, cds)
    assert cds_seq == "CCGATG"


@torch.no_grad()
def test_codonbert_embed_ragged_agg(model):
    """Test embed with identity agg_fn returns per-token 2D embeddings."""
    seqs = ["ATGATG", "GCGCGCGCGCGC"]
    cds = [make_cds(s) for s in seqs]
    out = model.embed(seqs, cds=cds, agg_fn=lambda x, **kwargs: x)
    assert_raw_batch_matches_single(model, seqs, out, cds=cds)
    assert out[0].dim() == 2  # (num_tokens, hidden_dim)
    assert out[1].dim() == 2
    assert out[0].shape[0] != out[1].shape[0]  # ragged: different codon counts
    assert out[0].shape[1] == out[1].shape[1]  # same hidden dim


def test_codonbert_gradient_flow(model):
    """Test that gradients can flow through the model."""
    model.set_train_mode()

    seq = "ATGATG"
    out = model.embed([seq], cds=[make_cds(seq)])
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


def test_codonbert_extract_structure(model):
    """extract() returns (dict, dict) with matching keys; hidden states are 2D."""
    model.set_inference_mode()
    seq = "ATGATG"
    cds = make_cds(seq)
    h, s = model.extract([seq], cds=[cds], layers=[0])
    assert isinstance(h, dict) and isinstance(s, dict)
    assert set(h.keys()) == set(s.keys())
    layer = next(iter(h))
    assert h[layer][0][0].dim() == 2
    assert h[layer][0][0].device.type == "cpu"


def test_codonbert_extract_layer_selection(model):
    """Requesting layers=[0] returns exactly 1 layer."""
    seq = "ATGATG"
    cds = make_cds(seq)
    h, _ = model.extract([seq], cds=[cds], layers=[0])
    assert len(h) == 1


def test_codonbert_extract_attention_weights(model):
    """return_attentions=True yields (H, T, T) tensors with rows summing to 1."""
    model.set_inference_mode()
    seq = "ATGATG"
    cds = make_cds(seq)
    h, s = model.extract([seq], cds=[cds], layers=[0], return_attentions=True)
    layer = next(iter(s))
    attn = s[layer]
    assert attn is not None
    w = attn[0][0]
    assert w.dim() == 3
    assert w.shape[1] == w.shape[2]
    assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-6)
