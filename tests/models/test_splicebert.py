from unittest.mock import patch

import pytest

pytest.importorskip("torch")
import torch

from mrna_bench.models.splicebert import SpliceBERT


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def model_1024(device) -> SpliceBERT:
    """Get SpliceBERT 1024nt model."""
    return SpliceBERT("SpliceBERT-1024nt", device, "eager")


@pytest.fixture(scope="module")
def model_510(device) -> SpliceBERT:
    """Get SpliceBERT 510nt model."""
    return SpliceBERT("SpliceBERT-510nt", device, "eager")


def test_splicebert_1024_forward(model_1024):
    """Test SpliceBERT 1024nt initialization and forward pass."""
    model_1024.set_inference_mode()
    assert model_1024.max_length == 1024

    text = "ATGATGATGATG"
    output = torch.stack(model_1024.embed([text])).cpu()
    assert output.shape == (1, 512)


def test_splicebert_510_forward(model_510):
    """Test SpliceBERT 510nt initialization and forward pass."""
    model_510.set_inference_mode()
    assert model_510.max_length == 510

    text = "A" * 510
    output = torch.stack(model_510.embed([text])).cpu()
    assert output.shape == (1, 512)


def test_splicebert_1024_embed_batch(model_1024):
    """Test batch embed matches individual embeddings for 1024nt model."""
    model_1024.set_inference_mode()
    sequences = [
        "ATGATGATGATG",
        "GGCCAATTGGCC",
        "TTTAAAGGGCCCAAA",
    ]

    batch_output = torch.stack(model_1024.embed(sequences)).cpu()
    assert batch_output.shape == (3, 512)

    for i, seq in enumerate(sequences):
        single_output = torch.stack(model_1024.embed([seq])).cpu()
        assert torch.allclose(
            batch_output[i:i + 1],
            single_output,
            atol=1e-5
        ), "Mismatch at sequence {}".format(i)


def test_splicebert_510_embed_batch(model_510):
    """Test batch embed matches individual embeddings for 510nt model."""
    model_510.set_inference_mode()
    sequences = [
        "A" * 510,
        "G" * 510,
        "C" * 510,
    ]

    batch_output = torch.stack(model_510.embed(sequences)).cpu()
    assert batch_output.shape == (3, 512)

    for i, seq in enumerate(sequences):
        single_output = torch.stack(model_510.embed([seq])).cpu()
        assert torch.allclose(
            batch_output[i:i + 1],
            single_output,
            atol=1e-5
        ), "Mismatch at sequence {}".format(i)


def test_splicebert_510_overlap_handling(model_510):
    """Test 510nt model handles sequences requiring overlap correctly."""
    model_510.set_inference_mode()

    seq_with_overlap = "A" * 600
    output = torch.stack(model_510.embed([seq_with_overlap])).cpu()
    assert output.shape == (1, 512)


def test_splicebert_510_overlap_chunks(model_510):
    """Test 510nt overlap creates correctly padded chunks."""
    model_510.set_inference_mode()

    spillover = 10
    input_seq = "A" * spillover + "G" * 510

    captured_chunks = []

    original_forward = model_510._forward_chunks

    def capture_forward(chunks):
        captured_chunks.extend(chunks)
        return original_forward(chunks)

    with patch.object(model_510, "_forward_chunks", side_effect=capture_forward):
        model_510.embed([input_seq])

    assert len(captured_chunks) == 2
    assert len(captured_chunks[0]) == 510
    assert len(captured_chunks[1]) == 510

    assert captured_chunks[0] == "A" * 10 + "G" * 500
    assert captured_chunks[1] == "G" * 510


def test_splicebert_1024_chunking(model_1024):
    """Test 1024nt model handles long sequences with chunking."""
    model_1024.set_inference_mode()

    long_seq = "ATGC" * 500
    output = torch.stack(model_1024.embed([long_seq])).cpu()
    assert output.shape == (1, 512)


@torch.no_grad()
def test_splicebert_embed_ragged_agg(model_1024):
    """Test embed with identity agg_fn returns per-token embeddings (ragged)."""
    seqs = ["ATGATG", "GCGCGCGCGCGC"]
    out = model_1024.embed(seqs, agg_fn=lambda x, **kwargs: x)
    assert out[0].dim() == 2  # (num_tokens, hidden_dim)
    assert out[1].dim() == 2
    assert out[0].shape[0] != out[1].shape[0]  # ragged: different token counts
    assert out[0].shape[1] == out[1].shape[1]  # same hidden dim


def test_splicebert_gradient_flow(model_1024):
    """Test that gradients can flow through the model."""
    model_1024.set_train_mode()

    out = model_1024.embed(["ATGATG"])
    assert out[0].requires_grad, "Output should require gradients"

    loss = torch.stack(out).sum()
    loss.backward()

    has_grad = False
    for param in model_1024.model.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break

    assert has_grad, "No gradients flowed to model parameters"
    model_1024.set_inference_mode()


def test_splicebert_extract_structure(model_1024):
    """extract() returns (dict, dict) with matching keys; hidden states are 2D."""
    h, s = model_1024.extract(["ATGATG"], layers=[0])
    assert isinstance(h, dict) and isinstance(s, dict)
    assert set(h.keys()) == set(s.keys())
    layer = next(iter(h))
    assert h[layer][0][0].dim() == 2
    assert h[layer][0][0].device.type == "cpu"


def test_splicebert_extract_layer_selection(model_1024):
    """Requesting layers=[0] returns exactly 1 layer."""
    h, _ = model_1024.extract(["ATGATG"], layers=[0])
    assert len(h) == 1


def test_splicebert_extract_attention_weights(model_1024):
    """return_attentions=True yields (H, T, T) tensors with rows summing to 1."""
    h, s = model_1024.extract(["ATGATG"], layers=[0], return_attentions=True)
    layer = next(iter(s))
    attn = s[layer]
    assert attn is not None
    w = attn[0][0]
    assert w.dim() == 3
    assert w.shape[1] == w.shape[2]
    assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-6)
