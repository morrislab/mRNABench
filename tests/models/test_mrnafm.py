import pytest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

pytest.importorskip("torch")
import torch
from mrna_bench.models.mrnafm import MRNAFM


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def model(device) -> MRNAFM:
    """Get mRNA-FM model."""
    model = MRNAFM("mRNA-FM", device, "eager")
    model.set_inference_mode()
    return model


def make_cds(seq: str) -> np.ndarray:
    """Make CDS array marking every codon start (assumes pure CDS input)."""
    arr = np.zeros(len(seq), dtype=int)
    arr[::3] = 1
    return arr


def test_mrnafm_requires_cds(model):
    """Test mRNA-FM raises error when CDS is not provided."""
    with pytest.raises(ValueError, match="mRNA-FM requires cds"):
        model.embed(["ATGATG"])


def test_mrnafm_embed_batch(model):
    """Test mRNA-FM batch embedding."""
    sequences = ["ATGATG", "GCGCGC", "AAACCC"]
    cds = [make_cds(s) for s in sequences]
    out = torch.stack(model.embed(sequences, cds=cds))
    assert out.shape == (3, 1280)


def test_mrnafm_converts_t_to_u(model):
    """Test mRNA-FM converts T->U before embedding."""
    seq = "ATGATG"
    with patch.object(
        model,
        "_forward_chunks",
        wraps=model._forward_chunks
    ) as mock:
        mock.return_value = (
            torch.zeros(1, 6, 1280),
            torch.ones(1, 6)
        )
        model.embed([seq], cds=[make_cds(seq)])
        chunks = mock.call_args[0][0]
        assert chunks[0] == "AUGAUG"


def test_mrnafm_cds_slice(model):
    """Test mRNA-FM correctly slices CDS region before embedding."""
    input_seq = "A" * 30 + "T" * 30 + "G" * 40
    cds = np.array([0] * 30 + [1, 0, 0] * 10 + [0] * 40)

    with patch.object(
        model,
        "_forward_chunks",
        wraps=model._forward_chunks
    ) as mock:
        mock.return_value = (
            torch.zeros(1, 30, 1280),
            torch.ones(1, 30)
        )
        model.embed([input_seq], cds=[cds])
        chunks = mock.call_args[0][0]
        # CDS region is 30 U's (converted from T's)
        assert chunks[0] == "U" * 30


def test_mrnafm_get_cds_full(model):
    """Test get_cds method."""
    sequence = "CCGATGCCG"
    cds = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0])

    cds_seq = model.get_cds(sequence, cds)
    assert cds_seq == "ATG"


def test_mrnafm_get_cds_missing(model):
    """Test get_cds method when cds is missing."""
    sequence_1 = "CCGATGCCG"
    cds_1 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])

    sequence_2 = "CCGATGCC"
    cds_2 = np.array([0, 0, 0, 0, 0, 0, 0, 0])

    with pytest.warns(UserWarning, match="No CDS found"):
        cds_seq_1 = model.get_cds(sequence_1, cds_1)
    assert cds_seq_1 == "CCGATGCCG"

    with pytest.warns(UserWarning, match="No CDS found"):
        cds_seq_2 = model.get_cds(sequence_2, cds_2)
    assert cds_seq_2 == "CCGATG"


def test_mrnafm_get_cds_irregular(model):
    """Test get_cds method when cds is not a multiple of 3."""
    sequence = "CCGATGCCG"
    cds_1 = np.array([0, 0, 0, 0, 1, 0, 0, 1, 0])
    cds_2 = np.array([0, 0, 0, 0, 0, 1, 0, 0, 1])

    with pytest.warns(UserWarning, match="Irregular CDS"):
        cds_1_seq = model.get_cds(sequence, cds_1)
    with pytest.warns(UserWarning, match="Irregular CDS"):
        cds_2_seq = model.get_cds(sequence, cds_2)

    assert cds_1_seq == "TGC"
    assert cds_2_seq == "GCC"


def test_mrnafm_mask_codon_tokenization(model):
    """Verify mask accounts for codon (3nt = 1 token) tokenization."""
    with patch.object(model, "model") as mock_model:
        # 6 nucleotides = 2 codons, so expect seq_len = 2 + 2 (CLS + EOS)
        mock_model.return_value = SimpleNamespace(
            last_hidden_state=torch.ones(1, 4, 1280, device=model.device)
        )

        _, mask = model._forward_chunks(["AUGAUG"])

        # Should have exactly 2 ones (for 2 codon tokens)
        assert mask.sum().item() == 2
        # Positions 1 and 2 should be 1, others 0
        assert mask[0, 0].item() == 0  # CLS
        assert mask[0, 1].item() == 1  # codon 1
        assert mask[0, 2].item() == 1  # codon 2
        assert mask[0, 3].item() == 0  # EOS


def test_mrnafm_mask_variable_lengths(model):
    """Test mask construction with different length sequences."""
    with patch.object(model, "model") as mock_model:
        # 3 sequences: 1, 2, 3 codons; padded to longest (3 codons + CLS + EOS = 5)
        mock_model.return_value = SimpleNamespace(
            last_hidden_state=torch.ones(3, 5, 1280, device=model.device)
        )

        chunks = ["AUG", "AUGAUG", "AUGAUGAUG"]  # 1, 2, 3 codons
        _, mask = model._forward_chunks(chunks)

        # Verify each sequence has correct number of content tokens masked
        assert mask[0].sum().item() == 1  # 1 codon
        assert mask[1].sum().item() == 2  # 2 codons
        assert mask[2].sum().item() == 3  # 3 codons


def test_mrnafm_embed_batch_ragged(model):
    """Test mRNA-FM batch embedding with variable length sequences."""
    sequences = ["ATGATG", "GCGCGCGCGCGC"]
    cds = [make_cds(s) for s in sequences]
    out = torch.stack(model.embed(sequences, cds=cds))
    assert out.shape == (2, 1280)


def test_mrnafm_chunking(model):
    """Test mRNA-FM with sequence requiring chunking."""
    # max_chunk_length = (1024 - 2) * 3 = 3066 nucleotides
    # Create sequence longer than that to trigger chunking
    long_seq = "ATG" * 1100  # 3300 nucleotides > 3066

    with patch.object(
        model,
        "_forward_chunks",
        wraps=model._forward_chunks
    ) as mock:
        # Return mock values for 2 chunks
        mock.return_value = (
            torch.zeros(2, 1024, 1280, device=model.device),
            torch.ones(2, 1024, device=model.device)
        )
        model.embed([long_seq], cds=[make_cds(long_seq)])

        # Should be called once with 2 chunks
        assert mock.call_count == 1
        chunks = mock.call_args[0][0]
        assert len(chunks) == 2
        assert len(chunks[0]) == 3066  # max chunk length
        assert len(chunks[1]) == 234  # remaining: 3300 - 3066


@torch.no_grad()
def test_mrnafm_embed_ragged_agg(model):
    """Test embed with identity agg_fn returns per-token embeddings (ragged)."""
    seqs = ["ATGATG", "GCGCGCATGATG"]  # 6 and 12 chars
    cds = [make_cds(s) for s in seqs]
    out = model.embed(seqs, cds=cds, agg_fn=lambda x, **kwargs: x)
    assert out[0].dim() == 2  # (num_tokens, hidden_dim)
    assert out[1].dim() == 2
    assert out[0].shape[0] != out[1].shape[0]  # ragged: different token counts
    assert out[0].shape[1] == out[1].shape[1]  # same hidden dim


def test_mrnafm_gradient_flow(model):
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

def test_mrnafm_extract_structure(model):
    seq = "ATGATG"
    cds = make_cds(seq)
    h, s = model.extract([seq], cds=[cds], layers=[0])
    assert isinstance(h, dict) and isinstance(s, dict)
    assert set(h.keys()) == set(s.keys())
    layer = next(iter(h))
    assert h[layer][0][0].dim() == 2
    assert h[layer][0][0].device.type == "cpu"


def test_mrnafm_extract_layer_selection(model):
    seq = "ATGATG"
    cds = make_cds(seq)
    h, _ = model.extract([seq], cds=[cds], layers=[0])
    assert len(h) == 1


def test_mrnafm_extract_attention_weights(model):
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
