import pytest

pytest.importorskip("torch")
import torch

from mrna_bench.models.utrlm import UTRLM


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def model(device) -> UTRLM:
    """Get UTR-LM model."""
    return UTRLM("utrlm-te_el", device)


def test_utrlm_forward(model):
    """Test UTR-LM initialization and forward pass."""
    model.set_inference_mode()
    text = "ACUUUGGCCA"
    output = torch.stack(model.embed([text])).cpu()
    assert output.shape == (1, 128)


def test_utrlm_forward_dna_input(model):
    """Test UTR-LM forward pass with DNA input (T instead of U)."""
    model.set_inference_mode()
    text_rna = "ACUUUGGCCA"
    text_dna = "ACTTTGGCCA"

    output_rna = torch.stack(model.embed([text_rna])).cpu()
    output_dna = torch.stack(model.embed([text_dna])).cpu()

    assert torch.allclose(output_rna, output_dna, atol=1e-5)


def test_utrlm_embed_batch(model):
    """Test batch embed matches individual embeddings."""
    model.set_inference_mode()
    sequences = [
        "ACUUUGGCCA",
        "GGCCAAUUGG",
        "UUUAAAGGGCCC",
    ]

    batch_output = torch.stack(model.embed(sequences)).cpu()
    assert batch_output.shape == (3, 128)

    for i, seq in enumerate(sequences):
        single_output = torch.stack(model.embed([seq])).cpu()
        assert torch.allclose(
            batch_output[i:i + 1],
            single_output,
            atol=1e-5
        ), "Mismatch at sequence {}".format(i)


@torch.no_grad()
def test_utrlm_embed_ragged_agg(model):
    """Test embed with identity agg_fn returns per-token embeddings (ragged)."""
    seqs = ["ATGATG", "GCGCGCGCGCGC"]
    out = model.embed(seqs, agg_fn=lambda x, **kwargs: x)
    assert out[0].dim() == 2  # (num_tokens, hidden_dim)
    assert out[1].dim() == 2
    assert out[0].shape[0] != out[1].shape[0]  # ragged: different token counts
    assert out[0].shape[1] == out[1].shape[1]  # same hidden dim


def test_utrlm_gradient_flow(model):
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
