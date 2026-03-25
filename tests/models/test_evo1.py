import pytest

pytest.importorskip("torch")
pytest.importorskip("evo")
import torch
from mrna_bench.models.evo1 import Evo1


EVO1_VERSIONS = [
    "evo-1.5-8k-base",
    "evo-1-8k-base",
]


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module", params=EVO1_VERSIONS, ids=lambda x: x)
def model(request, device) -> Evo1:
    """Get Evo1 model."""
    return Evo1(request.param, device)


def test_evo1_forward(model):
    """Test Evo1 forward pass."""
    out = model.embed_sequence("ATGATG")
    assert out.shape == (1, 4096)


def test_evo1_max_length(device):
    """Test Evo1 max_length is set correctly."""
    model_8k = Evo1("evo-1-8k-base", device)
    assert model_8k.max_length == 8192


def test_evo1_embed_batch(model):
    """Test batch embed matches individual embeddings."""
    model.set_inference_mode()
    sequences = [
        "ATGATG" * 10,
        "ATGATG" * 50,
        "ATGATG" * 100,
    ]

    batch_output = torch.stack(model.embed(sequences)).cpu()
    assert batch_output.shape == (3, 4096)

    for i, seq in enumerate(sequences):
        single_output = model.embed_sequence(seq).cpu()
        assert torch.allclose(
            batch_output[i:i + 1],
            single_output,
            atol=1e-4
        ), "Mismatch at sequence {} (len {})".format(i, len(seq))


@torch.no_grad()
def test_evo1_embed_ragged_agg(model):
    """Test embed with identity agg_fn returns per-token embeddings (ragged)."""
    seqs = ["ATGATG", "GCGCGCGCGCGC"]
    out = model.embed(seqs, agg_fn=lambda x, **kwargs: x)
    assert out[0].dim() == 2  # (num_tokens, hidden_dim)
    assert out[1].dim() == 2
    assert out[0].shape[0] != out[1].shape[0]  # ragged: different token counts
    assert out[0].shape[1] == out[1].shape[1]  # same hidden dim


def test_evo1_gradient_flow(model):
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
