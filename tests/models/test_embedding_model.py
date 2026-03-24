import pytest

pytest.importorskip("torch")
import torch

from mrna_bench.models.embedding_model import EmbeddingModel


class MockEmbeddingModel(EmbeddingModel):
    """Mock embedding model for testing base class methods."""

    default_version = "mock"
    valid_versions = ["mock"]
    hidden_dim = 64

    def __init__(self, device: torch.device):
        super().__init__("mock", device)
        self.model = torch.nn.Identity()

    def embed(self, sequences, cds=None, splice=None, agg_fn=torch.mean):
        """Not used directly in these tests."""
        pass

    def mock_forward_chunks(
        self,
        chunks: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mock forward pass with character-based embeddings.

        Content tokens get ord(char), masked positions get -999.
        """
        batch_size = len(chunks)
        max_len = max(len(c) for c in chunks)

        hidden_states = torch.full(
            (batch_size, max_len, self.hidden_dim),
            -999.0,
            device=self.device
        )
        pooling_mask = torch.zeros(batch_size, max_len, device=self.device)

        for i, chunk in enumerate(chunks):
            vals = torch.tensor(
                [ord(c) for c in chunk],
                dtype=torch.float,
                device=self.device
            )
            hidden_states[i, :len(chunk)] = vals.unsqueeze(1).expand(
                -1,
                self.hidden_dim
            )
            pooling_mask[i, :len(chunk)] = 1

        return hidden_states, pooling_mask


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def model(device) -> MockEmbeddingModel:
    """Get mock embedding model."""
    return MockEmbeddingModel(device)


def test_embed_with_chunking_batch_no_chunking(model):
    """Test batch of sequences that all fit in one chunk."""
    sequences = ["ATG", "ATGATG", "A"]
    max_chunk_length = 100

    output = model._embed_with_chunking(
        sequences=sequences,
        max_chunk_length=max_chunk_length,
        embed_fn=model.mock_forward_chunks,
    )

    assert output.shape == (3, model.hidden_dim)

    for i, seq in enumerate(sequences):
        expected_mean = sum(ord(c) for c in seq) / len(seq)
        assert torch.allclose(
            output[i],
            torch.full((model.hidden_dim,), expected_mean, device=model.device),
            atol=1e-5
        ), "Mismatch at sequence {}".format(i)


def test_embed_with_chunking_batch_with_chunking(model):
    """Test batch where sequences require multiple chunks."""
    sequences = [
        "ATG",           # 1 chunk
        "ATGATGATG",     # 3 chunks with max_chunk_length=3
        "ATGATGA",       # 3 chunks, last chunk partial
    ]
    max_chunk_length = 3

    output = model._embed_with_chunking(
        sequences=sequences,
        max_chunk_length=max_chunk_length,
        embed_fn=model.mock_forward_chunks,
    )

    assert output.shape == (3, model.hidden_dim)

    for i, seq in enumerate(sequences):
        expected_mean = sum(ord(c) for c in seq) / len(seq)
        assert torch.allclose(
            output[i],
            torch.full((model.hidden_dim,), expected_mean, device=model.device),
            atol=1e-5
        ), "Mismatch at sequence {}".format(i)


def test_embed_with_chunking_applies_pooling_mask(model):
    """Test that pooling_mask is used to filter tokens."""
    sequences = ["A", "ATGATG"]
    max_chunk_length = 100

    output = model._embed_with_chunking(
        sequences=sequences,
        max_chunk_length=max_chunk_length,
        embed_fn=model.mock_forward_chunks,
    )

    # If masked positions (-999) were included, mean would be wrong
    expected_mean_a = float(ord("A"))
    assert torch.allclose(
        output[0],
        torch.full((model.hidden_dim,), expected_mean_a, device=model.device),
        atol=1e-5
    ), "Pooling mask should exclude masked positions"


def test_embed_with_chunking_preserves_gradient(model):
    """Test that gradients flow through _embed_with_chunking."""
    with torch.enable_grad():
        sequences = ["ATGATG"]
        max_chunk_length = 3

        # Create a simple model that we can backprop through
        class GradientMockModel(MockEmbeddingModel):
            def __init__(self, device):
                super().__init__(device)
                self.linear = torch.nn.Linear(64, 64).to(device)

            def mock_forward_chunks_with_grad(
                self,
                chunks: list[str]
            ) -> tuple[torch.Tensor, torch.Tensor]:
                _, pooling_mask = self.mock_forward_chunks(chunks)
                # Create fresh tensor with gradients enabled
                batch_size = len(chunks)
                max_len = max(len(c) for c in chunks)
                hidden_states = torch.randn(
                    batch_size, max_len, self.hidden_dim,
                    device=self.device, requires_grad=True
                )
                # Pass through linear layer to create gradient connection
                hidden_states = self.linear(hidden_states)
                return hidden_states, pooling_mask

        grad_model = GradientMockModel(model.device)
        grad_model.linear.weight.requires_grad = True

        output = grad_model._embed_with_chunking(
            sequences=sequences,
            max_chunk_length=max_chunk_length,
            embed_fn=grad_model.mock_forward_chunks_with_grad,
        )

        # Backprop should work
        loss = output.sum()
        loss.backward()

        assert grad_model.linear.weight.grad is not None
        assert not torch.all(grad_model.linear.weight.grad == 0)


def test_chunk_sequence(model):
    """Test chunk_sequence utility method."""
    sequence = "ATGATGATG"

    chunks = model.chunk_sequence(sequence, 3)
    assert chunks == ["ATG", "ATG", "ATG"]

    chunks = model.chunk_sequence(sequence, 4)
    assert chunks == ["ATGA", "TGAT", "G"]

    chunks = model.chunk_sequence(sequence, 100)
    assert chunks == ["ATGATGATG"]


def test_chunk_tokens(model):
    """Test chunk_tokens utility method."""
    tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    chunks = model.chunk_tokens(tokens, 3)
    assert chunks == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    chunks = model.chunk_tokens(tokens, 4)
    assert chunks == [[1, 2, 3, 4], [5, 6, 7, 8], [9]]

    chunks = model.chunk_tokens(tokens, 100)
    assert chunks == [[1, 2, 3, 4, 5, 6, 7, 8, 9]]


def test_invalid_version_raises_value_error(device):
    """Test that invalid model version raises ValueError with valid versions."""
    class MultiVersionModel(EmbeddingModel):
        default_version = "v1"
        valid_versions = ["v1", "v2", "v3"]

        def __init__(self, model_version: str, device: torch.device):
            super().__init__(model_version, device)
            self.model = torch.nn.Identity()

        def embed(self, sequences, cds=None, splice=None, agg_fn=torch.mean):
            pass

    with pytest.raises(ValueError) as exc_info:
        MultiVersionModel("invalid", device)

    error_msg = str(exc_info.value)
    assert "invalid" in error_msg
    assert "v1" in error_msg
    assert "v2" in error_msg
    assert "v3" in error_msg
