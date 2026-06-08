import pytest

from unittest.mock import patch

import numpy as np

pytest.importorskip("torch")
import torch

from mrna_bench.models.mrnabert import mRNABERT


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def model(device) -> mRNABERT:
    """Get mRNABERT model."""
    m = mRNABERT("mRNABERT", device, "eager")
    m.set_inference_mode()
    return m


def make_cds(seq: str) -> np.ndarray:
    """Make CDS array marking every codon start (assumes pure CDS input)."""
    arr = np.zeros(len(seq), dtype=int)
    arr[::3] = 1
    return arr


def test_mrnabert_forward(model):
    """Test mRNABERT forward pass using six track input."""
    out = model.embed_sequence_sixtrack(
        "ATG",
        np.array([0, 0, 0]),
        np.array([0, 0, 0])
    )
    assert out.shape == (768,)


def test_mrnabert_separate_utr_cds_single_codon(model):
    """UTR-CDS-UTR structure is spaced correctly for a single codon."""
    seq = "AACTGCGTG"
    cds = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0])
    assert model.separate_utr_cds(seq, cds) == ["A A CTG C G T G"]


def test_mrnabert_separate_utr_cds_multiple_codons(model):
    """Multiple contiguous CDS codons should be spaced in triples."""
    seq = "AACTGAAACCCGG"
    cds = np.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    assert model.separate_utr_cds(seq, cds) == ["A A CTG AAA C C C G G"]


def test_mrnabert_separate_utr_cds_at_start(model):
    """Handles CDS beginning at index 0, followed by UTR content."""
    seq = "ATGAAATTT"
    cds = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0])
    assert model.separate_utr_cds(seq, cds) == ["ATG AAA T T T"]


def test_mrnabert_separate_utr_cds_at_end(model):
    """Handles CDS at the end of the sequence without trailing codon padding."""
    seq = "GGGATGAAA"
    cds = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0])
    assert model.separate_utr_cds(seq, cds) == ["G G G ATG AAA"]


def test_mrnabert_separate_utr_cds_no_cds(model):
    """If no CDS sites exist, the whole sequence should be spaced as UTR."""
    seq = "ACGTAC"
    cds = np.zeros(len(seq), dtype=int)
    assert model.separate_utr_cds(seq, cds) == ["A C G T A C"]


def test_mrnabert_separate_utr_cds_truncated_codon(model):
    """Truncated codons at the end should not be extended or padded."""
    seq = "AAATGAA"
    cds = np.array([0, 0, 1, 0, 0, 1, 0])
    assert model.separate_utr_cds(seq, cds) == ["A A ATG AA"]


def test_mrnabert_cds_aware_chunking(model):
    """Ensure chunking never splits CDS codons across chunk boundaries."""
    seq = "A" * 5 + "ATG" * 400
    cds = np.zeros(len(seq), dtype=int)

    for i in range(5, len(seq), 3):
        cds[i] = 1  # mark codon start

    chunks = model.chunk_sequence_cds_aware(seq, cds, chunk_length=200)

    # No chunk should end mid-codon (CDS length always multiple of 3)
    for chunk_seq, chunk_cds in chunks:
        starts = np.where(chunk_cds != 0)[0]
        if len(starts) == 0:
            continue
        cds_start = starts[0]
        cds_end = min(starts[-1] + 3, len(chunk_seq))
        assert (cds_end - cds_start) % 3 == 0


def test_mrnabert_embed_batch(model):
    """Test mRNABERT batch embedding."""
    sequences = ["ATGATG", "GCGCGC", "AAACCC"]
    cds = [make_cds(s) for s in sequences]
    out = torch.stack(model.embed(sequences, cds=cds))
    assert out.shape == (3, 768)


@torch.no_grad()
def test_mrnabert_embed_batch_ragged(model):
    """Test ragged batches match individual embeddings."""
    sequences = ["ATGATG", "GCGCGCGCGCGC", "AAA", "ATGATGATG"]
    cds = [make_cds(s) for s in sequences]

    batch_out = torch.stack(model.embed(sequences, cds=cds)).cpu()
    assert batch_out.shape == (4, 768)

    for i, (seq, c) in enumerate(zip(sequences, cds)):
        single_out = torch.stack(model.embed([seq], cds=[c])).cpu()
        assert torch.allclose(
            batch_out[i:i + 1], single_out, atol=1e-5
        ), f"Mismatch at sequence {i} (len {len(seq)})"


def test_mrnabert_requires_cds(model):
    """Test that embed raises ValueError without CDS tracks."""
    with pytest.raises(ValueError, match="CDS tracks must be provided"):
        model.embed(["ATGATG"])


@torch.no_grad()
def test_mrnabert_embed_ragged_agg(model):
    """Test embed with identity agg_fn returns per-token embeddings (ragged)."""
    seqs = ["ATGATG", "GCGCGCGCGCGC"]
    cds = [make_cds(s) for s in seqs]
    out = model.embed(seqs, cds=cds, agg_fn=lambda x, **kwargs: x)
    assert out[0].dim() == 2  # (num_tokens, hidden_dim)
    assert out[1].dim() == 2
    assert out[0].shape[0] != out[1].shape[0]  # ragged: different token counts
    assert out[0].shape[1] == out[1].shape[1]  # same hidden dim


def test_mrnabert_gradient_flow(model):
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

def test_mrnabert_extract_structure(model):
    seq = "ATGATGATG"
    cds = make_cds(seq)
    h, s = model.extract([seq], cds=[cds], layers=[0])
    assert isinstance(h, dict) and isinstance(s, dict)
    assert set(h.keys()) == set(s.keys())
    layer = next(iter(h))
    assert h[layer][0][0].dim() == 2
    assert h[layer][0][0].device.type == "cpu"


def test_mrnabert_extract_layer_selection(model):
    seq = "ATGATGATG"
    cds = make_cds(seq)
    h, _ = model.extract([seq], cds=[cds], layers=[0])
    assert len(h) == 1


def test_mrnabert_extract_attention_weights(model):
    """mRNABERT with eager attn returns attention weights."""
    seq = "ATGATGATG"
    cds = make_cds(seq)
    h, s = model.extract([seq], cds=[cds], layers=[0], return_attentions=True)
    layer = next(iter(s))
    attn = s[layer]
    assert attn is not None
    w = attn[0][0]
    assert w.dim() == 3
    assert w.shape[1] == w.shape[2]
    assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-6)
