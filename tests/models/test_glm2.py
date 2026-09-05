import pytest

pytest.importorskip("torch")

import torch

from tests.model_utils import (
    assert_pooled_batch_matches_single,
    assert_raw_batch_matches_single,
    embed_one,
)
from mrna_bench.models.glm2 import GLM2

HIDDEN_DIM = 640


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def model(device) -> GLM2:
    """Get gLM2-150M model."""
    model = GLM2("gLM-150M", device, "eager")
    model.set_inference_mode()
    return model


def test_glm2_forward(model):
    """Test gLM2 forward pass."""
    out = embed_one(model, "ATGATG")
    assert out.shape == (1, HIDDEN_DIM)


@torch.no_grad()
def test_glm2_embed_batch_ragged(model):
    """Test ragged batches match individual embeddings."""
    sequences = [
        "ATGATG" * 10,
        "ATGATG" * 50,
        "ATGATG" * 100,
    ]
    assert_pooled_batch_matches_single(model, sequences, atol=2e-5)


@torch.no_grad()
def test_glm2_excludes_special_tokens(model):
    """Strand marker and special tokens are excluded from pooling."""
    seq = "ACGTACGT" * 10
    toks = model.tokenizer(
        [seq], return_tensors="pt", padding=True
    ).to(model.device)
    hidden = model.model(**toks).last_hidden_state

    special_ids = torch.tensor(
        model.tokenizer.all_special_ids,
        device=model.device,
    )
    mask = ~torch.isin(toks["input_ids"][0], special_ids)
    mean_no_special = hidden[0, mask].float().mean(0).unsqueeze(0).cpu()
    mean_all = hidden.float().mean(dim=1).cpu()
    output = embed_one(model, seq).cpu()

    assert torch.allclose(output, mean_no_special, atol=1e-5)
    assert not torch.allclose(output, mean_all, atol=1e-5)


@torch.no_grad()
def test_glm2_embed_ragged_agg(model):
    """Identity agg_fn returns per-token embeddings (ragged)."""
    seqs = ["ATGATG", "GCGCGCGCGCGC"]
    out = model.embed(seqs, agg_fn=lambda x, **kwargs: x)
    assert_raw_batch_matches_single(model, seqs, out)
    assert out[0].dim() == 2
    assert out[1].dim() == 2
    assert out[0].shape[0] != out[1].shape[0]
    assert out[0].shape[1] == out[1].shape[1] == HIDDEN_DIM


def test_glm2_gradient_flow(model):
    """Gradients flow through the model."""
    model.set_train_mode()

    out = model.embed(["ATGATG"])
    assert out[0].requires_grad

    loss = torch.stack(out).sum()
    loss.backward()

    has_grad = False
    for param in model.model.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break

    assert has_grad, "No gradients flowed to model parameters"
    model.set_inference_mode()


def test_glm2_extract_structure(model):
    """extract() returns (dict, dict) with 2D hidden states on CPU."""
    h, s = model.extract(["ATGATG"], layers=[0])
    assert isinstance(h, dict) and isinstance(s, dict)
    assert set(h.keys()) == set(s.keys())
    layer = next(iter(h))
    assert h[layer][0][0].dim() == 2
    assert h[layer][0][0].device.type == "cpu"


def test_glm2_extract_layer_selection(model):
    """layers=[0] returns exactly 1 layer."""
    h, _ = model.extract(["ATGATG"], layers=[0])
    assert len(h) == 1


def test_glm2_extract_attention_weights(model):
    """Attention weights are (H, T, T) with rows summing to 1."""
    _, s = model.extract(
        ["ATGATG"], layers=[0], return_attentions=True,
    )
    layer = next(iter(s))
    attn = s[layer]
    assert attn is not None
    w = attn[0][0]
    assert w.dim() == 3
    assert w.shape[1] == w.shape[2]
    assert torch.allclose(
        w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-6,
    )
