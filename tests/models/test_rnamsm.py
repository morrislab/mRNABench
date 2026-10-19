import pytest

pytest.importorskip("torch")
import torch

from mrna_bench.models.rnamsm import RNAMSM


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def model(device) -> RNAMSM:
    """Get RNA-MSM model."""
    return RNAMSM("rnamsm", device)


def test_rnamsm_forward(model):
    """Test RNA-MSM initialization and forward pass."""
    model.set_inference_mode()
    text = "ACUUGGCCA"
    output = model.embed([text]).cpu()
    assert output.shape == (1, 768)


def test_rnamsm_forward_dna_input(model):
    """Test RNA-MSM forward pass with DNA input (T instead of U)."""
    model.set_inference_mode()
    text_rna = "ACUUGGCCA"
    text_dna = "ACTTGGCCA"

    output_rna = model.embed([text_rna]).cpu()
    output_dna = model.embed([text_dna]).cpu()

    assert torch.allclose(output_rna, output_dna, atol=1e-5)


def test_rnamsm_embed_batch(model):
    """Test batch embed matches individual embeddings."""
    model.set_inference_mode()
    sequences = [
        "ACUUGGCCA",
        "GGCCAAUUGG",
        "UUUAAAGGGCCC",
    ]

    batch_output = model.embed(sequences).cpu()
    assert batch_output.shape == (3, 768)

    for i, seq in enumerate(sequences):
        single_output = model.embed([seq]).cpu()
        assert torch.allclose(
            batch_output[i:i + 1],
            single_output,
            atol=1e-5
        ), "Mismatch at sequence {}".format(i)


def test_rnamsm_gradient_flow(model):
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
