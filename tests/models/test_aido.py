import pytest

from unittest.mock import patch

pytest.importorskip("torch")

import torch
from mrna_bench.models.aido import AIDORNA


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def aidomodel(device) -> AIDORNA:
    """Get AIDORNA model."""
    model = AIDORNA("aido_rna_650m", device)
    model.set_inference_mode()
    return model


def test_aido_forward(aidomodel):
    """Test AIDORNA initialization and forward pass."""

    out = aidomodel.embed_sequence("ATGATG")
    assert out.shape == (1, 1280)


def test_aido_forward_long(aidomodel):
    """Test AIDORNA forward pass."""
    long_sequence = "ATGC" * 257  # 1028 nucleotides
    out = aidomodel.embed_sequence(long_sequence)
    assert out.shape == (1, 1280)


def test_aido_embed_batch(aidomodel):
    """Test AIDORNA batch embedding."""
    sequences = ["ATGATG", "GCGCGC", "AAACCC"]
    out = torch.stack(aidomodel.embed(sequences))
    assert out.shape == (3, 1280)


@torch.no_grad()
def test_aido_embed_batch_ragged(aidomodel):
    """Test ragged batches match individual embeddings."""
    sequences = [
        "ACTG" * 5,
        "ACTG" * 50,
        "ACTG" * 200,
        "ACTG" * 10,
    ]

    batch_out = torch.stack(aidomodel.embed(sequences)).cpu()
    assert batch_out.shape == (4, 1280)

    for i, seq in enumerate(sequences):
        single_out = torch.stack(aidomodel.embed([seq])).cpu()
        assert torch.allclose(
            batch_out[i:i + 1], single_out, atol=1e-5
        ), f"Mismatch at sequence {i} (len {len(seq)})"


def test_aido_excludes_special_tokens(aidomodel):
    """Verify pooling mask excludes CLS and SEP special tokens."""
    with patch.object(aidomodel, "model") as mock_model:
        mock_model.return_value.last_hidden_state = torch.ones(
            1, 8, 1280, device=aidomodel.device
        )

        _, mask = aidomodel._forward_chunks(["ATGATG"])

        # Mask should be 0 at CLS (pos 0) and SEP (last real token pos)
        assert mask[0, 0].item() == 0  # CLS
        assert mask[0, -1].item() == 0  # SEP (last padding pos, but seq fills all)
        # Nucleotide positions (1 to seq_len-1) should be 1
        assert mask[0, 1:-1].sum().item() == 6


@torch.no_grad()
def test_aido_embed_ragged_agg(aidomodel):
    """Test embed with identity agg_fn returns per-token embeddings (ragged)."""
    seqs = ["ATGATG", "GCGCGCGCGCGC"]
    out = aidomodel.embed(seqs, agg_fn=lambda x, **kwargs: x)
    assert out[0].dim() == 2  # (num_tokens, hidden_dim)
    assert out[1].dim() == 2
    assert out[0].shape[0] != out[1].shape[0]  # ragged: different token counts
    assert out[0].shape[1] == out[1].shape[1]  # same hidden dim


def test_aido_gradient_flow(aidomodel):
    """Test that gradients can flow through the model."""
    aidomodel.set_train_mode()

    out = aidomodel.embed(["ATGATG"])
    assert out[0].requires_grad, "Output should require gradients"

    loss = torch.stack(out).sum()
    loss.backward()

    has_grad = False
    for param in aidomodel.model.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break

    assert has_grad, "No gradients flowed to model parameters"
