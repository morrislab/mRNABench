import pytest

pytest.importorskip("torch")
import torch
from mrna_bench.models.dnabert_s import DNABERTS


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def model(device) -> DNABERTS:
    """Get DNABERT-S model."""
    model = DNABERTS("dnabert-s", device)
    model.set_inference_mode()
    return model


def test_dnaberts_forward(model):
    """Test DNABERT-S forward pass."""
    out = model.embed_sequence("ATGATG")
    assert out.shape == (1, 768)


@torch.no_grad()
def test_dnaberts_embed_batch_ragged(model):
    """Test ragged batches match individual embeddings."""
    sequences = [
        "ATGATG" * 10,
        "ATGATG" * 50,
        "ATGATG" * 100,
    ]

    batch_output = torch.stack(model.embed(sequences)).cpu()
    assert batch_output.shape == (3, 768)

    for i, seq in enumerate(sequences):
        single_output = model.embed_sequence(seq).cpu()
        assert torch.allclose(
            batch_output[i:i + 1],
            single_output,
            atol=1e-4
        ), "Mismatch at sequence {} (len {})".format(i, len(seq))


@torch.no_grad()
def test_dnaberts_excludes_special_tokens(model):
    """Test that CLS and SEP tokens are excluded from pooling."""
    text = "ATGATG" * 20

    toks = model.tokenizer([text], return_tensors="pt", padding=True)
    toks = toks.to(model.device)
    hidden_states = model.model(**toks)[0]

    # Mean over ALL tokens (including CLS/SEP)
    mean_all = hidden_states.mean(dim=1).cpu()

    # Mean excluding first and last (CLS/SEP)
    mean_no_special = hidden_states[:, 1:-1, :].mean(dim=1).cpu()

    output = model.embed_sequence(text).cpu()

    assert torch.allclose(output, mean_no_special, atol=1e-5), \
        "Output should exclude CLS/SEP tokens"
    assert not torch.allclose(output, mean_all, atol=1e-5), \
        "Output should differ from mean including special tokens"


@torch.no_grad()
def test_dnabert_s_embed_ragged_agg(model):
    """Test embed with identity agg_fn returns per-token embeddings (ragged)."""
    seqs = ["ATGATG", "GCGCGCGCGCGC"]
    out = model.embed(seqs, agg_fn=lambda x, **kwargs: x)
    assert out[0].dim() == 2  # (num_tokens, hidden_dim)
    assert out[1].dim() == 2
    assert out[0].shape[0] != out[1].shape[0]  # ragged: different token counts
    assert out[0].shape[1] == out[1].shape[1]  # same hidden dim


def test_dnaberts_gradient_flow(model):
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
