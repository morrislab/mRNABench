import pytest

import numpy as np
import torch

from mrna_bench.models.helix_mrna import HelixmRNA


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def helix_mrna(device) -> HelixmRNA:
    """Get Helix-mRNA model."""
    model = HelixmRNA("helix-mrna", device)
    model.set_inference_mode()
    return model


def test_helix_mrna_forward(helix_mrna):
    """Test Helix-mRNA initialization and forward pass."""
    out = helix_mrna.embed_sequence("ATGATG")
    assert out.shape == (1, 256)

@torch.no_grad()
def test_helix_mrna_converts_t_to_u(helix_mrna):
    """Test that Helix-mRNA converts T->U for proper tokenization."""
    # DNA and RNA versions should produce identical embeddings
    dna_seq = "ATGATGATG"
    rna_seq = "AUGAUGAUG"

    dna_output = helix_mrna.embed_sequence(dna_seq).cpu()
    rna_output = helix_mrna.embed_sequence(rna_seq).cpu()

    assert torch.allclose(dna_output, rna_output, atol=1e-5), \
        "DNA (T) and RNA (U) sequences should produce identical embeddings"


@torch.no_grad()
def test_helix_mrna_embed_batch(helix_mrna):
    """Test Helix-mRNA batch embedding."""
    sequences = ["ATGATG", "ATGATGATG", "ATGATGATGATG"]
    output = torch.stack(helix_mrna.embed(sequences)).cpu()
    assert output.shape == (3, 256)


def test_helix_mrna_converter(helix_mrna):
    """Test Helix-mRNA sequence converter."""
    seq = "AUGUAG"
    cds_1 = np.array([0, 0, 0, 1, 0, 0])
    cds_2 = np.array([0, 0, 0, 0, 1, 1])
    cds_3 = np.array([0, 0, 0, 0, 0, 0])

    out_1 = helix_mrna._tokenize_cds(seq, cds_1)
    out_2 = helix_mrna._tokenize_cds(seq, cds_2)
    out_3 = helix_mrna._tokenize_cds(seq, cds_3)

    assert out_1 == "AUGEUAG"
    assert out_2 == "AUGUEAEG"
    assert out_3 == "AUGUAG"


@torch.no_grad()
def test_helix_mrna_embed_ragged_agg(helix_mrna):
    """Test embed with identity agg_fn returns per-token embeddings (ragged)."""
    seqs = ["ATGATG", "GCGCGCGCGCGC"]
    out = helix_mrna.embed(seqs, agg_fn=lambda x, **kwargs: x)
    assert out[0].dim() == 2  # (num_tokens, hidden_dim)
    assert out[1].dim() == 2
    assert out[0].shape[0] != out[1].shape[0]  # ragged: different token counts
    assert out[0].shape[1] == out[1].shape[1]  # same hidden dim


def test_helix_mrna_gradient_flow(helix_mrna):
    """Test that gradients can flow through the model."""
    helix_mrna.set_train_mode()

    out = helix_mrna.embed(["ATGATG"])
    assert out[0].requires_grad, "Output should require gradients"

    loss = torch.stack(out).sum()
    loss.backward()

    has_grad = False
    for param in helix_mrna.model.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break

    assert has_grad, "No gradients flowed to model parameters"
