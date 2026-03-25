import pytest

from unittest.mock import patch

pytest.importorskip("torch")
import torch
from mrna_bench.models.rnafm import RNAFM


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def model(device) -> RNAFM:
    """Get RNA-FM model."""
    model = RNAFM("rna-fm", device)
    model.set_inference_mode()
    return model


def test_rnafm_forward(model):
    """Test RNA-FM forward pass."""
    assert model.max_length == 1024

    out = model.embed(["ATGATG"])
    assert len(out) == 1 and out[0].shape == (640,)


def test_rnafm_embed_batch(model):
    """Test RNA-FM batch embedding."""
    out = model.embed(["ATGATG", "GCGCGC", "AAAA"])
    assert len(out) == 3 and out[0].shape == (640,)


def test_rnafm_converts_t_to_u(model):
    """Test RNA-FM converts T->U before embedding."""
    with patch.object(
        model,
        "_forward_chunks",
        wraps=model._forward_chunks
    ) as mock:
        mock.return_value = (
            torch.zeros(1, 6, 640),
            torch.ones(1, 6)
        )
        model.embed(["ATGATG"])
        chunks = mock.call_args[0][0]
        assert chunks[0] == "AUGAUG"


def test_rnafm_embed_batch_ragged(model):
    """Test RNA-FM batch embedding with variable length sequences."""
    out = model.embed(["ATGATG", "GCGCGCGCGCGC"])
    assert len(out) == 2 and out[0].shape == (640,)


def test_rnafm_mask_nucleotide_tokenization(model):
    """Verify mask accounts for nucleotide (1nt = 1 token) tokenization."""
    with patch.object(model, "model") as mock_model:
        # 6 nucleotides, so expect seq_len = 6 + 2 (CLS + EOS)
        mock_model.return_value = {
            "representations": {12: torch.ones(1, 8, 640, device=model.device)}
        }

        _, mask = model._forward_chunks(["AUGAUG"])

        # Should have exactly 6 ones (for 6 nucleotide tokens)
        assert mask.sum().item() == 6
        # Position 0 (CLS) and 7 (EOS) should be 0
        assert mask[0, 0].item() == 0  # CLS
        assert mask[0, 7].item() == 0  # EOS
        # Positions 1-6 should be 1
        for i in range(1, 7):
            assert mask[0, i].item() == 1


def test_rnafm_mask_variable_lengths(model):
    """Test mask construction with different length sequences."""
    with patch.object(model, "model") as mock_model:
        # 3 sequences: 3, 6, 9 nucleotides; padded to longest (9 + CLS + EOS = 11)
        mock_model.return_value = {
            "representations": {12: torch.ones(3, 11, 640, device=model.device)}
        }

        chunks = ["AUG", "AUGAUG", "AUGAUGAUG"]  # 3, 6, 9 nucleotides
        _, mask = model._forward_chunks(chunks)

        # Verify each sequence has correct number of content tokens masked
        assert mask[0].sum().item() == 3  # 3 nucleotides
        assert mask[1].sum().item() == 6  # 6 nucleotides
        assert mask[2].sum().item() == 9  # 9 nucleotides


@torch.no_grad()
def test_rnafm_embed_ragged_agg(model):
    """Test embed with identity agg_fn returns per-token embeddings (ragged)."""
    seqs = ["ATGATG", "GCGCGCGCGCGC"]
    out = model.embed(seqs, agg_fn=lambda x, **kwargs: x)
    assert out[0].dim() == 2  # (num_tokens, hidden_dim)
    assert out[1].dim() == 2
    assert out[0].shape[0] != out[1].shape[0]  # ragged: different token counts
    assert out[0].shape[1] == out[1].shape[1]  # same hidden dim


def test_rnafm_gradient_flow(model):
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
