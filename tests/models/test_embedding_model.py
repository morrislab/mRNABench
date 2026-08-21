import pytest
from types import SimpleNamespace

import numpy as np

pytest.importorskip("torch")
import torch

from mrna_bench.models.embedding_model import (
    EmbeddingModel,
    ModelBehavior,
    mean_pool,
)


class MockEmbeddingModel(EmbeddingModel):
    """Mock embedding model for testing base class methods."""

    default_version = "mock"
    valid_versions = ["mock"]
    default_attn_implementation = "mock"
    valid_attn_implementations = ["mock"]
    supported_behaviors = frozenset({ModelBehavior.EMBEDDING})
    hidden_dim = 64

    def __init__(self, device: torch.device):
        super().__init__("mock", device)
        self.model = torch.nn.Identity()

    def embed(
        self,
        sequences,
        cds=None,
        splice=None,
        agg_fn=EmbeddingModel.mean_pool,
    ):
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


def test_embed_with_chunking_batch(model):
    """Test pooled batches with and without sequence chunking."""
    cases = [
        (["ATG", "ATGATG", "A"], 100),
        (["ATG", "ATGATGATG", "ATGATGA"], 3),
    ]
    for sequences, max_chunk_length in cases:
        output = model._embed_with_chunking(
            sequences=sequences,
            max_chunk_length=max_chunk_length,
            embed_fn=model.mock_forward_chunks,
        )

        assert len(output) == 3 and output[0].shape == (model.hidden_dim,)

        for i, seq in enumerate(sequences):
            expected_mean = sum(ord(c) for c in seq) / len(seq)
            assert torch.allclose(
                output[i],
                torch.full(
                    (model.hidden_dim,),
                    expected_mean,
                    device=model.device,
                ),
                atol=1e-5,
            ), "Mismatch at sequence {}".format(i)


def test_mean_pool_uses_float32():
    """Default pooling preserves half-precision token differences."""
    hidden = torch.tensor([[1.0, 2.0], [1.01, 2.01]], dtype=torch.float16)
    pooled = mean_pool(hidden)

    assert pooled.dtype == torch.float32
    torch.testing.assert_close(pooled, hidden.float().mean(dim=0))


def test_flash_attention_uses_float16(model):
    """FlashAttention uses FP16 while other backends retain FP32."""
    original = model.attn_implementation
    try:
        model.attn_implementation = "flash_attention_2"
        assert model._get_inference_dtype() == torch.float16
        model.attn_implementation = "eager"
        assert model._get_inference_dtype() == torch.float32
    finally:
        model.attn_implementation = original


def test_resolve_layer_paths_rejects_undeclared_strings(model):
    """String layer paths must name a declared hookable layer."""
    model.hookable_layer_patterns = [r""]

    assert model._resolve_layer_paths([""]) == [""]
    with pytest.raises(ValueError, match="Unknown layer"):
        model._resolve_layer_paths(["weight"])


def test_run_with_layer_capture_supports_multiple_paths(model):
    """Selected layer outputs can be captured without detaching gradients."""
    model.model = torch.nn.Sequential(
        torch.nn.Linear(4, 4),
        torch.nn.ReLU(),
    )
    inputs = torch.randn(2, 4, requires_grad=True)

    output, activations = model._run_with_layer_capture(
        ["0", "1"],
        lambda: model.model(inputs),
        detach=False,
    )

    assert set(activations) == {"0", "1"}
    assert len(activations["0"]) == len(activations["1"]) == 1
    assert activations["0"][0].requires_grad
    assert activations["1"][0].requires_grad
    output.sum().backward()
    assert inputs.grad is not None


def test_standard_hf_extract_captures_only_requested_layers():
    """HF extraction uses hooks instead of materializing all hidden states."""
    class ExtractBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Identity(),
                torch.nn.ReLU(),
            )
            self.requested_hidden_states = None

        def forward(
            self,
            input_ids,
            output_hidden_states=False,
            output_attentions=False,
        ):
            self.requested_hidden_states = output_hidden_states
            hidden = self.layers[0](input_ids.float())
            hidden = self.layers[1](hidden.relu().transpose(0, 1))
            return SimpleNamespace(
                last_hidden_state=hidden,
                attentions=None,
            )

    extract_model = MockEmbeddingModel(torch.device("cpu"))
    extract_model.model = ExtractBackbone()
    extract_model.hookable_layer_patterns = [r"layers\.\d+"]

    hidden, scores = extract_model._standard_hf_extract(
        sequences=["AC"],
        tokenize_fn=lambda seqs: {
            "input_ids": -torch.ones(1, len(seqs[0]), 2)
        },
        max_chunk_length=10,
        layers=[1],
    )

    assert extract_model.model.requested_hidden_states is False
    assert list(hidden) == ["layers.1"]
    torch.testing.assert_close(
        hidden["layers.1"][0][0],
        torch.zeros(2, 2),
    )
    assert scores["layers.1"] is None


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


def test_embed_with_chunking_raw_batch_matches_single(model):
    """Return identical unpooled embeddings with and without padding."""
    sequences = ["A", "ATGATG"]
    batch = model._embed_with_chunking(
        sequences=sequences,
        max_chunk_length=100,
        embed_fn=model.mock_forward_chunks,
        agg_fn=lambda hidden: hidden,
    )
    singles = [
        model._embed_with_chunking(
            sequences=[sequence],
            max_chunk_length=100,
            embed_fn=model.mock_forward_chunks,
            agg_fn=lambda hidden: hidden,
        )[0]
        for sequence in sequences
    ]

    for batched, single in zip(batch, singles):
        assert torch.equal(batched, single)


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
        loss = torch.stack(output).sum()
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


def test_score_tracks_are_validated_and_chunked(model):
    sequences = ["ACGT", "AA"]
    cds = [np.zeros(4), np.zeros(2)]
    splice = [np.arange(4), np.arange(2)]

    model._validate_score_tracks(sequences, cds, splice)
    with pytest.raises(ValueError, match="one track per sequence"):
        model._validate_score_tracks(sequences, cds[:1], splice)
    with pytest.raises(ValueError, match="match sequence lengths"):
        model._validate_score_tracks(sequences, [np.zeros(3), cds[1]], splice)

    model.sequence_score_chunk_length = 2
    chunks = model._score_chunks(sequences[0], cds[0], splice[0])
    assert [chunk[0] for chunk in chunks] == ["AC", "GT"]
    np.testing.assert_array_equal(chunks[1][2], np.array([2, 3]))


def test_invalid_version_raises_value_error(device):
    """Test that invalid model version raises ValueError with valid versions."""
    class MultiVersionModel(EmbeddingModel):
        default_version = "v1"
        valid_versions = ["v1", "v2", "v3"]
        supported_behaviors = frozenset({ModelBehavior.EMBEDDING})

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


def test_causal_sequence_score(device):
    class Tokenizer:
        def __call__(self, sequence, **kwargs):
            ids = [{"A": 0, "C": 1}[base] for base in sequence]
            return {"input_ids": torch.tensor([ids])}

    class Model(torch.nn.Module):
        def forward(self, input_ids, **kwargs):
            logits = torch.full(
                (*input_ids.shape, 2),
                -10.0,
                device=input_ids.device,
            )
            for idx in range(input_ids.shape[1] - 1):
                logits[0, idx, input_ids[0, idx + 1]] = 10.0
            return SimpleNamespace(logits=logits)

    class ScoringModel(MockEmbeddingModel):
        supported_behaviors = frozenset({
            ModelBehavior.EMBEDDING,
            ModelBehavior.CAUSAL_LIKELIHOOD,
        })

        def __init__(self, device):
            super().__init__(device)
            self.tokenizer = Tokenizer()
            self.model = Model()

    scorer = ScoringModel(device)
    mean_score = scorer.sequence_score(["ACAC"])[0]
    sum_score = scorer.sequence_score(
        ["ACAC"], normalization="sum"
    )[0]
    assert scorer.logits(["ACAC"])[0].shape == (4, 2)

    assert mean_score == pytest.approx(0.0, abs=1e-6)
    assert sum_score == pytest.approx(mean_score * 3)
    with pytest.raises(ValueError):
        scorer.sequence_score(["ACAC"], ModelBehavior.PSEUDO_LIKELIHOOD)

    scorer.sequence_score_chunk_length = 2
    with pytest.warns(RuntimeWarning, match="independently per chunk"):
        scorer.sequence_score(["ACAC"])
    assert scorer.logits(["ACAC"])[0].shape == (4, 2)

    class FixedModel(torch.nn.Module):
        def forward(self, input_ids, **kwargs):
            logits = torch.full(
                (*input_ids.shape, 2),
                -10.0,
                device=input_ids.device,
            )
            logits[..., 0] = 10.0
            return SimpleNamespace(logits=logits)

    scorer.model = FixedModel()
    with pytest.warns(RuntimeWarning):
        reference = scorer.sequence_score(["AAAA"], normalization="sum")[0]
    with pytest.warns(RuntimeWarning):
        alternate = scorer.sequence_score(["AACA"], normalization="sum")[0]
    assert reference > alternate


def test_pseudo_log_likelihood(device):
    class Tokenizer:
        mask_token_id = 2

        def __call__(self, sequence, **kwargs):
            ids = [{"A": 0, "C": 1}[base] for base in sequence]
            return {"input_ids": torch.tensor([ids])}

        def get_special_tokens_mask(self, ids, **kwargs):
            return [0] * len(ids)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, input_ids):
            self.calls += 1
            logits = torch.zeros(
                (*input_ids.shape, 3),
                device=input_ids.device,
            )
            logits[..., 0] = 2.0
            return SimpleNamespace(logits=logits)

    class ScoringModel(MockEmbeddingModel):
        sequence_score_batch_size = 2
        supported_behaviors = frozenset({
            ModelBehavior.EMBEDDING,
            ModelBehavior.PSEUDO_LIKELIHOOD,
        })

        def __init__(self, device):
            super().__init__(device)
            self.tokenizer = Tokenizer()
            self.model = Model()

    scorer = ScoringModel(device)
    score_a, score_c = scorer.sequence_score(["AAA", "CCC"])

    assert score_a > score_c
    assert scorer.model.calls == 4


def test_masked_marginal_llr_handles_kmer_tokens(device):
    """A substitution scores every overlapping tokenizer token it changes."""
    class KmerTokenizer:
        mask_token_id = 4
        vocab = {"AA": 0, "AC": 1, "CA": 2}

        def __call__(self, sequence, **kwargs):
            ids = [
                self.vocab[sequence[idx:idx + 2]]
                for idx in range(len(sequence) - 1)
            ]
            return {"input_ids": torch.tensor([ids])}

        def get_special_tokens_mask(self, ids, **kwargs):
            return [0] * len(ids)

    class Model(torch.nn.Module):
        def forward(self, input_ids):
            logits = torch.zeros(
                (*input_ids.shape, 5),
                device=input_ids.device,
            )
            logits[..., 0] = 3.0
            return SimpleNamespace(logits=logits)

    class ScoringModel(MockEmbeddingModel):
        supported_behaviors = frozenset({
            ModelBehavior.EMBEDDING,
            ModelBehavior.PSEUDO_LIKELIHOOD,
        })

        def __init__(self, device):
            super().__init__(device)
            self.tokenizer = KmerTokenizer()
            self.model = Model()

    scorer = ScoringModel(device)
    with pytest.warns(RuntimeWarning, match="multiple tokenizer tokens"):
        score = scorer.masked_marginal_llr(["AAAA"], ["ACAA"])[0]

    assert score > 0
    with pytest.raises(ValueError, match="substitutions only"):
        scorer.masked_marginal_llr(["AAAA"], ["AAA"])
