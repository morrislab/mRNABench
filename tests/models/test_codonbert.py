import pytest

pytest.importorskip("torch")
import torch
from mrna_bench.models.codonbert import CodonBERT


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def model(device) -> CodonBERT:
    """Get CodonBERT model."""
    return CodonBERT("codonbert", device)


def test_codonbert_forward(model):
    """Test CodonBERT forward pass."""
    out = model.embed_sequence("ATGATG")
    assert out.shape == (1, 768)


@torch.no_grad()
def test_codonbert_converts_t_to_u(model):
    """Test that CodonBERT converts T->U for proper tokenization."""
    # DNA and RNA versions should produce identical embeddings
    dna_seq = "ATGATGATG"
    rna_seq = "AUGAUGAUG"

    dna_output = model.embed_sequence(dna_seq).cpu()
    rna_output = model.embed_sequence(rna_seq).cpu()

    assert torch.allclose(dna_output, rna_output, atol=1e-5), \
        "DNA (T) and RNA (U) sequences should produce identical embeddings"


@torch.no_grad()
def test_codonbert_embed_batch(model):
    """Test CodonBERT batch embedding."""
    sequences = ["ATGATG", "ATGATGATG", "ATGATGATGATG"]
    output = model.embed(sequences).cpu()
    assert output.shape == (3, 768)


@torch.no_grad()
def test_codonbert_embed_batch_single_equals_embed_sequence(model):
    """Test that embed_batch with single sequence matches embed_sequence."""
    text = "ATGATGATG"
    single_output = model.embed_sequence(text).cpu()
    batch_output = model.embed([text]).cpu()
    assert torch.allclose(single_output, batch_output, atol=1e-5)


@torch.no_grad()
def test_codonbert_embed_batch_ragged(model):
    """Test ragged batches match individual embeddings."""
    sequences = [
        "ATG" * 10,
        "ATG" * 50,
        "ATG" * 100,
        "ATG" * 20,
    ]

    batch_output = model.embed(sequences).cpu()
    assert batch_output.shape == (4, 768)

    for i, seq in enumerate(sequences):
        single_output = model.embed_sequence(seq).cpu()
        assert torch.allclose(
            batch_output[i:i + 1],
            single_output,
            atol=1e-5
        ), "Mismatch at sequence {} (len {})".format(i, len(seq))


@torch.no_grad()
def test_codonbert_excludes_special_tokens(model):
    """Test that CLS and SEP tokens are excluded from pooling."""
    text = "AUG" * 20  # Use RNA notation to match vocab

    toks = model.tokenizer([text], return_tensors="pt", padding=True)
    toks = toks.to(model.device)
    hidden_states = model.model(**toks).last_hidden_state

    # Mean over ALL tokens (including CLS/SEP)
    mean_all = hidden_states.mean(dim=1).cpu()

    # Mean excluding first and last (CLS/SEP)
    mean_no_special = hidden_states[:, 1:-1, :].mean(dim=1).cpu()

    # Model output should match mean_no_special, not mean_all
    # Note: embed_sequence converts T->U internally, so use RNA input
    output = model.embed_sequence(text).cpu()

    assert torch.allclose(output, mean_no_special, atol=1e-5), \
        "Output should exclude CLS/SEP tokens"
    assert not torch.allclose(output, mean_all, atol=1e-5), \
        "Output should differ from mean including special tokens"


@torch.no_grad()
def test_codonbert_single_codon(model):
    """Test embedding a single codon."""
    output = model.embed_sequence("ATG").cpu()
    assert output.shape == (1, 768)
    assert not torch.isnan(output).any()


@torch.no_grad()
def test_codonbert_max_length_boundary(model):
    """Test sequence at max_length boundary."""
    # Exactly at boundary (1 chunk) - max_length_nt = 3066
    seq_at_boundary = "ATG" * (model.max_length_nt // 3)
    output1 = model.embed_sequence(seq_at_boundary).cpu()
    assert output1.shape == (1, 768)

    # One codon over boundary (2 chunks)
    seq_over_boundary = seq_at_boundary + "ATG"
    output2 = model.embed_sequence(seq_over_boundary).cpu()
    assert output2.shape == (1, 768)

    # Outputs should be different
    assert not torch.allclose(output1, output2, atol=1e-5)


def test_codonbert_gradient_flow(model):
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
