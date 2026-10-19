import pytest

pytest.importorskip("torch")

import torch
from mrna_bench.models.rnaernie import RNAErnie


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def model(device) -> RNAErnie:
    """Get RNAErnie model."""
    return RNAErnie("rnaernie", device)


def test_rnaernie_forward(model):
    """Test RNAErnie forward pass."""
    out = model.embed_sequence("ATGATG")
    assert out.shape == (1, 768)


@torch.no_grad()
def test_rnaernie_converts_t_to_u(model):
    """Test that RNAErnie converts T->U for proper tokenization."""
    dna_seq = "ATGATGATG"
    rna_seq = "AUGAUGAUG"

    dna_output = model.embed_sequence(dna_seq).cpu()
    rna_output = model.embed_sequence(rna_seq).cpu()

    assert torch.allclose(dna_output, rna_output, atol=1e-5), \
        "DNA (T) and RNA (U) sequences should produce identical embeddings"


@torch.no_grad()
def test_rnaernie_embed_batch_ragged(model):
    """Test ragged batches match individual embeddings."""
    sequences = [
        "ATGATG" * 10,
        "ATGATG" * 50,
        "ATGATG" * 100,
    ]

    batch_output = model.embed(sequences).cpu()
    assert batch_output.shape == (3, 768)

    for i, seq in enumerate(sequences):
        single_output = model.embed_sequence(seq).cpu()
        assert torch.allclose(
            batch_output[i:i + 1],
            single_output,
            atol=1e-4
        ), "Mismatch at sequence {}".format(i)


@torch.no_grad()
def test_rnaernie_excludes_special_tokens(model):
    """Test that CLS and SEP tokens are excluded from pooling."""
    text = "AUGAUG" * 20

    toks = model.tokenizer([text], return_tensors="pt", padding=True)
    toks = toks.to(model.device)
    hidden_states = model.model(**toks).last_hidden_state

    mean_all = hidden_states.mean(dim=1).cpu()
    mean_no_special = hidden_states[:, 1:-1, :].mean(dim=1).cpu()

    output = model.embed_sequence(text).cpu()

    assert torch.allclose(output, mean_no_special, atol=1e-5), \
        "Output should exclude CLS/SEP tokens"
    assert not torch.allclose(output, mean_all, atol=1e-5), \
        "Output should differ from mean including special tokens"


def test_rnaernie_gradient_flow(model):
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
