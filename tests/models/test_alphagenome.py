import pytest

pytest.importorskip("torch")
pytest.importorskip("alphagenome_pytorch")
import torch

from mrna_bench.models.alphagenome import AlphaGenome


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def alphagenome(device) -> AlphaGenome:
    """Get AlphaGenome model."""
    return AlphaGenome("alphagenome", device)


@torch.no_grad()
def test_alphagenome_forward(alphagenome):
    """Test AlphaGenome forward pass."""
    out = alphagenome.embed_sequence("ATGATG" * 10)
    assert out.shape == (1, 1536)


@torch.no_grad()
def test_alphagenome_embed_batch(alphagenome):
    """Test batch embedding matches individual embeddings."""
    sequences = ["ATGATG" * 10, "ACGTTGCA" * 400]

    batch_output = alphagenome.embed(sequences).cpu()
    assert batch_output.shape == (2, 1536)

    for i, seq in enumerate(sequences):
        single_output = alphagenome.embed_sequence(seq).cpu()
        assert torch.allclose(batch_output[i:i + 1], single_output, atol=1e-5)


@torch.no_grad()
def test_alphagenome_strips_padding(alphagenome):
    """Test token embeddings cover the sequence, not its 2048 padding."""
    sequence = "ACGT" * 300  # 1200 bases, padded to 2048
    tokens = alphagenome.embed_sequence(sequence, agg_fn=lambda x, dim: x)
    assert tokens.shape == (1, 1200, 1536)
