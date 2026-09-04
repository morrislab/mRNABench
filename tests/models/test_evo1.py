import pytest

pytest.importorskip("torch")

from unittest.mock import patch

import torch

from tests.model_utils import assert_raw_batch_matches_single, embed_one

if not torch.cuda.is_available():
    pytest.skip(
        "Evo1 7B integration test requires a CUDA GPU",
        allow_module_level=True,
    )

from mrna_bench.models.evo1 import Evo1

EVO1_VERSION = "Evo1-1-7B-8K"
HIDDEN_DIM = 4096


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device."""
    return torch.device("cuda")


@pytest.fixture(scope="module")
def model(device) -> Evo1:
    """Get Evo1 model (eager backend so attention weights are materializable)."""
    model = Evo1(EVO1_VERSION, device, "eager")
    model.set_inference_mode()
    return model


def test_evo1_forward(model):
    """Evo1 embeds to the final-layer hidden dimension."""
    out = embed_one(model, "ATGATG")
    assert out.shape == (1, HIDDEN_DIM)


def test_evo1_max_length(model):
    """Evo1-1-7B-8K uses an 8192nt context window."""
    assert model.max_length == 8192


@torch.no_grad()
def test_evo1_embed_batch(model):
    """Test batch embed matches individual embeddings."""
    sequences = [
        "ATGATG" * 10,
        "ATGATG" * 10 + "ATG",
    ]

    with patch.object(
        model.model,
        "forward",
        wraps=model.model.forward,
    ) as forward:
        batch_output = torch.stack(model.embed(sequences)).cpu()
    assert forward.call_args.kwargs["input_ids"].shape[0] == 2
    assert batch_output.shape == (2, HIDDEN_DIM)

    for i, seq in enumerate(sequences):
        single_output = embed_one(model, seq).cpu()
        cosine = torch.nn.functional.cosine_similarity(
            batch_output[i:i + 1].float(),
            single_output.float(),
        )
        assert cosine.item() >= 0.999


@torch.no_grad()
def test_evo1_embed_ragged_agg(model):
    """Test embed with identity agg_fn returns per-token embeddings (ragged)."""
    seqs = ["ATGATG", "GCGCGCG"]
    out = model.embed(seqs, agg_fn=lambda x, **kwargs: x)
    # BF16 kernels preserve direction more reliably than elementwise values.
    assert_raw_batch_matches_single(
        model, seqs, out, min_cosine=0.999
    )
    assert out[0].dim() == 2  # (num_tokens, hidden_dim)
    assert out[1].dim() == 2
    assert out[0].shape[0] != out[1].shape[0]  # ragged: different token counts
    assert out[0].shape[1] == out[1].shape[1] == HIDDEN_DIM  # same feature dim


def test_evo1_gradient_flow(model):
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
    model.set_inference_mode()


def test_evo1_extract_structure(model):
    """extract() returns (dict, dict) with matching keys; hidden states are 2D."""
    h, s = model.extract(["ATGATG"], layers=[0])
    assert isinstance(h, dict) and isinstance(s, dict)
    assert set(h.keys()) == set(s.keys())
    layer = next(iter(h))
    assert h[layer][0][0].dim() == 2
    assert h[layer][0][0].device.type == "cpu"


def test_evo1_extract_layer_selection(model):
    """Requesting layers=[0] returns exactly 1 layer."""
    h, _ = model.extract(["ATGATG"], layers=[0])
    assert len(h) == 1


def test_evo1_extract_transformer_attention(model):
    """Transformer blocks return causal (H, T, T) attention, rows ~sum to 1."""
    attn_idx = model.model.config.attn_layer_idxs[0]
    path = f"blocks.{attn_idx}"
    _, s = model.extract(
        ["ATGCATGCATGCATGCATGC"], layers=[path], return_attentions=True
    )
    assert s[path] is not None
    w = s[path][0][0]  # (H, T, T)
    assert w.dim() == 3
    assert w.shape[1] == w.shape[2]
    # Evo1 is autoregressive: attention is causal (strict upper triangle == 0).
    seq_len = w.shape[1]
    upper = w.float()[:, torch.triu(torch.ones(seq_len, seq_len), 1).bool()]
    assert upper.abs().max() == 0.0
    # Rows are a softmax distribution (sum to 1 within bf16 tolerance).
    rowsum = w.float().sum(-1)
    assert torch.allclose(rowsum, torch.ones_like(rowsum), atol=1e-6)


def test_evo1_extract_hyena_scores_none(model):
    """Hyena/SSM blocks always return None scores regardless of return_attentions."""
    _, s = model.extract(["ATGCATGC"], layers=["blocks.0"], return_attentions=True)
    assert s["blocks.0"] is None


def test_evo1_peft_target(model):
    """get_peft_target returns the HF model; set_peft_target writes it back."""
    target = model.get_peft_target()
    assert target is model.model

    original = model.model
    sentinel = torch.nn.Linear(1, 1)
    model.set_peft_target(sentinel)
    assert model.model is sentinel

    model.set_peft_target(original)  # restore for any subsequent use
    assert model.model is original
