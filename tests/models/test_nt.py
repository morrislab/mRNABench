import pytest

pytest.importorskip("torch")

import torch
from mrna_bench.models.nucleotide_transformer import NucleotideTransformer


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def model(device) -> NucleotideTransformer:
    """Get NucleotideTransformer model."""
    return NucleotideTransformer("v2-50m-multi-species", device)


def test_nt_forward(model):
    """Test NucleotideTransformer forward pass."""
    out = model.embed_sequence("ATGATG")
    assert out.shape == (1, 512)


@torch.no_grad()
def test_nt_embed_batch_ragged(model):
    """Test ragged batches match individual embeddings."""
    sequences = [
        "ATGATG" * 10,
        "ATGATG" * 50,
        "ATGATG" * 100,
    ]

    batch_output = model.embed(sequences).cpu()
    assert batch_output.shape == (3, 512)

    for i, seq in enumerate(sequences):
        single_output = model.embed_sequence(seq).cpu()
        assert torch.allclose(
            batch_output[i:i + 1],
            single_output,
            atol=1e-4
        ), "Mismatch at sequence {}".format(i)


@torch.no_grad()
def test_nt_excludes_special_tokens(model):
    """Test that CLS and SEP tokens are excluded from pooling."""
    text = "ATGATG" * 20

    toks = model.tokenizer([text], return_tensors="pt", padding=True)
    toks = toks.to(model.device)
    torch_outs = model.model(
        toks["input_ids"],
        attention_mask=toks["attention_mask"],
        encoder_attention_mask=toks["attention_mask"],
        output_hidden_states=True
    )
    hidden_states = torch_outs["hidden_states"][-1]

    mean_all = hidden_states.mean(dim=1).cpu()
    mean_no_special = hidden_states[:, 1:-1, :].mean(dim=1).cpu()

    output = model.embed_sequence(text).cpu()

    assert torch.allclose(output, mean_no_special, atol=1e-5), \
        "Output should exclude CLS/SEP tokens"
    assert not torch.allclose(output, mean_all, atol=1e-5), \
        "Output should differ from mean including special tokens"


def test_nt_gradient_flow(model):
    """Test that gradients can flow through the model."""
    model.set_train_mode()

    out = model.embed(["ATGATG" * 10])
    assert out.requires_grad, "Output should require gradients"

    loss = out.sum()
    loss.backward()

    has_grad = False
    for param in model.model.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break

    assert has_grad, "No gradients flowed to model parameters"
