import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
import torch
from mrna_bench.models.hyenadna import HyenaDNA


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def model(device) -> HyenaDNA:
    """Get HyenaDNA model."""
    return HyenaDNA("hyenadna-small-32k-seqlen-hf", device)


def test_hyenadna_forward(model):
    """Test HyenaDNA forward pass."""
    model.set_inference_mode()
    out = model.embed_sequence("ATGATG")
    assert out.shape == (1, 256)


def test_hyenadna_max_length(device):
    """Test HyenaDNA max_length is set correctly."""
    model_32k = HyenaDNA("hyenadna-small-32k-seqlen-hf", device)
    assert model_32k.max_length == 32000


def test_hyenadna_embed_batch(model):
    """Test batch embed matches individual embeddings."""
    model.set_inference_mode()
    sequences = [
        "ATGATG" * 10,
        "ATGATG" * 50,
        "ATGATG" * 100,
    ]

    batch_output = torch.stack(model.embed(sequences)).cpu()
    assert batch_output.shape == (3, 256)

    for i, seq in enumerate(sequences):
        single_output = model.embed_sequence(seq).cpu()
        assert torch.allclose(
            batch_output[i:i + 1],
            single_output,
            atol=1e-4
        ), "Mismatch at sequence {} (len {})".format(i, len(seq))


@torch.no_grad()
def test_hyenadna_embed_ragged_agg(model):
    """Test embed with identity agg_fn returns per-token embeddings (ragged)."""
    seqs = ["ATGATG", "GCGCGCGCGCGC"]
    out = model.embed(seqs, agg_fn=lambda x, **kwargs: x)
    assert out[0].dim() == 2  # (num_tokens, hidden_dim)
    assert out[1].dim() == 2
    assert out[0].shape[0] != out[1].shape[0]  # ragged: different token counts
    assert out[0].shape[1] == out[1].shape[1]  # same hidden dim


def test_hyenadna_gradient_flow(model):
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
