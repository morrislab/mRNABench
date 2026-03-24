import pytest

pytest.importorskip("torch")

import torch
from mrna_bench.models.omnigenome import OmniGenome


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def model_52m(device) -> OmniGenome:
    """Get OmniGenome 52M model."""
    return OmniGenome("omnigenome-52m", device)


@pytest.fixture(scope="module")
def model_186m(device) -> OmniGenome:
    """Get OmniGenome 186M model."""
    return OmniGenome("omnigenome-186m", device)


def test_omnigenome_52m_forward(model_52m):
    """Test OmniGenome 52M forward pass."""
    out = model_52m.embed_sequence("ATGATG")
    assert out.shape == (1, 480)


def test_omnigenome_186m_forward(model_186m):
    """Test OmniGenome 186M forward pass."""
    out = model_186m.embed_sequence("ATGATG")
    assert out.shape == (1, 720)


@torch.no_grad()
def test_omnigenome_converts_t_to_u(model_52m):
    """Test that OmniGenome converts T->U for proper tokenization."""
    dna_seq = "ATGATGATG"
    rna_seq = "AUGAUGAUG"

    dna_output = model_52m.embed_sequence(dna_seq).cpu()
    rna_output = model_52m.embed_sequence(rna_seq).cpu()

    assert torch.allclose(dna_output, rna_output, atol=1e-4), \
        "DNA (T) and RNA (U) sequences should produce identical embeddings"


@torch.no_grad()
def test_omnigenome_embed_batch_ragged(model_52m):
    """Test ragged batches match individual embeddings."""
    sequences = [
        "ATGATG" * 10,
        "ATGATG" * 50,
        "ATGATG" * 100,
    ]

    batch_output = model_52m.embed(sequences).cpu()
    assert batch_output.shape == (3, 480)

    for i, seq in enumerate(sequences):
        single_output = model_52m.embed_sequence(seq).cpu()
        assert torch.allclose(
            batch_output[i:i + 1],
            single_output,
            atol=1e-3
        ), "Mismatch at sequence {}".format(i)


@torch.no_grad()
def test_omnigenome_excludes_special_tokens(model_52m):
    """Test that CLS and SEP tokens are excluded from pooling."""
    text = "AUGAUG" * 20

    toks = model_52m.tokenizer([text], return_tensors="pt", padding=True)
    toks = toks.to(model_52m.device)
    hidden_states = model_52m.model(**toks).last_hidden_state

    mean_all = hidden_states.mean(dim=1).cpu()
    mean_no_special = hidden_states[:, 1:-1, :].mean(dim=1).cpu()

    output = model_52m.embed_sequence(text).cpu()

    assert torch.allclose(output, mean_no_special, atol=1e-4), \
        "Output should exclude CLS/SEP tokens"
    assert not torch.allclose(output, mean_all, atol=1e-4), \
        "Output should differ from mean including special tokens"


def test_omnigenome_gradient_flow(model_52m):
    """Test that gradients can flow through the model."""
    model_52m.set_train_mode()

    out = model_52m.embed(["ATGATG"])
    assert out.requires_grad, "Output should require gradients"

    loss = out.sum()
    loss.backward()

    has_grad = False
    for param in model_52m.model.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break

    assert has_grad, "No gradients flowed to model parameters"
