import pytest

from unittest.mock import patch

pytest.importorskip("torch")
import torch
from mrna_bench.models.generanno import GENERanno


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def model(device) -> GENERanno:
    """Get GENERanno model."""
    model = GENERanno("eukaryote-0.5b-base", device, "eager")
    model.set_inference_mode()
    return model


def test_generanno_forward(model):
    """Test GENERanno forward pass."""
    assert model.max_length == 8192

    out = model.embed(["ATGATG"])
    assert len(out) == 1 and out[0].shape == (1280,)


def test_generanno_embed_batch(model):
    """Test GENERanno batch embedding."""
    out = model.embed(["ATGATG", "GCGCGC", "AAACCC"])
    assert len(out) == 3 and out[0].shape == (1280,)


def test_generanno_embed_batch_ragged(model):
    """Test GENERanno batch embedding with variable length sequences."""
    out = model.embed(["ATGATG", "GCGCGCGCGCGC"])
    assert len(out) == 2 and out[0].shape == (1280,)


def test_generanno_excludes_special_tokens(model):
    """Verify pooling mask excludes CLS/SEP special tokens."""
    with patch.object(model, "model") as mock_model:
        mock_model.return_value.hidden_states = [
            torch.ones(1, 8, 1280, device=model.device)
        ]

        _, mask = model._forward_chunks(["ATGATG"])

        # Mask should be 0 at special token positions (CLS, SEP)
        assert mask[0, 0].item() == 0  # BOS
        assert mask[0, -1].item() == 0  # EOS
        # Nucleotide positions should be 1
        assert mask[0, 1:-1].sum().item() == 6


def test_generanno_chunking(model):
    """Test GENERanno with sequence requiring chunking."""
    long_seq = "A" * 9000  # > max_length of 8192

    with patch.object(
        model,
        "_forward_chunks",
        wraps=model._forward_chunks
    ) as mock:
        mock.return_value = (
            torch.zeros(1, 8190, 1280, device=model.device),
            torch.ones(1, 8190, device=model.device)
        )
        model.embed([long_seq])

        assert mock.call_count == 1
        chunks = mock.call_args[0][0]
        assert len(chunks) == 2
        assert len(chunks[0]) == 8190  # max_length - 2


@torch.no_grad()
def test_generanno_embed_ragged_agg(model):
    """Test embed with identity agg_fn returns per-token embeddings (ragged)."""
    seqs = ["ATGATG", "GCGCGCGCGCGC"]
    out = model.embed(seqs, agg_fn=lambda x, **kwargs: x)
    assert out[0].dim() == 2  # (num_tokens, hidden_dim)
    assert out[1].dim() == 2
    assert out[0].shape[0] != out[1].shape[0]  # ragged: different token counts
    assert out[0].shape[1] == out[1].shape[1]  # same hidden dim


def test_generanno_gradient_flow(model):
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
    

def test_generanno_extract_structure(model):
    """extract() returns (dict, dict) with matching keys; hidden states are 2D."""
    h, s = model.extract(["ATGATG"], layers=[0])
    assert isinstance(h, dict) and isinstance(s, dict)
    assert set(h.keys()) == set(s.keys())
    layer = next(iter(h))
    assert h[layer][0][0].dim() == 2
    assert h[layer][0][0].device.type == "cpu"


def test_generanno_extract_layer_selection(model):
    """Requesting layers=[0] returns exactly 1 layer."""
    h, _ = model.extract(["ATGATG"], layers=[0])
    assert len(h) == 1


def test_generanno_extract_attention_weights(model):
    """return_attentions=True with eager yields (H, T, T) tensors."""
    h, s = model.extract(["ATGATG"], layers=[0], return_attentions=True)
    layer = next(iter(s))
    attn = s[layer]
    assert attn is not None
    w = attn[0][0]
    assert w.dim() == 3
    assert w.shape[1] == w.shape[2]
    assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-4)
