import pytest

pytest.importorskip("torch")
pytest.importorskip("evo2")
import torch
from mrna_bench.models.evo2 import Evo2


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def model(device) -> Evo2:
    """Get Evo2 model."""
    return Evo2("evo2_1b_base", device)


def test_evo2_forward(model):
    """Test Evo2 forward pass."""
    out = model.embed_sequence("ATGATG")
    # Evo2 concatenates middle + last layer embeddings
    # evo2_1b_base has hidden_dim=2048, so 2048*2=4096
    assert out.shape == (1, 4096)


def test_evo2_max_length(device):
    """Test Evo2 max_length is set correctly for base variant."""
    model = Evo2("evo2_1b_base", device)
    assert model.max_length == 8192


@torch.no_grad()
def test_evo2_embed_batch(model):
    """Test batch embedding matches individual embeddings."""
    sequences = [
        "ATGATG" * 10,
        "ATGATG" * 50,
    ]

    batch_output = model.embed(sequences).cpu()
    assert batch_output.shape == (2, 4096)

    for i, seq in enumerate(sequences):
        single_output = model.embed_sequence(seq).cpu()
        assert torch.allclose(
            batch_output[i:i + 1],
            single_output,
            atol=1e-5
        ), "Mismatch at sequence {} (len {})".format(i, len(seq))


def test_evo2_gradient_flow(model):
    """Test that gradients can flow through the model."""
    model.set_train_mode()

    out = model.embed(["ATGATG"])
    assert out.requires_grad, "Output should require gradients"

    loss = out.sum()
    loss.backward()

    has_grad = False
    for param in model.model.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break

    assert has_grad, "No gradients flowed to model parameters"
