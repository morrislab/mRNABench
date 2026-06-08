import pytest

pytest.importorskip("torch")

import torch

# The Evo2 HuggingFace ports load via ``AutoModel.from_pretrained`` and do not
# require the ``evo2``/``vortex``/TransformerEngine packages. The smallest
# publicly runnable variant on non-Hopper GPUs is the 7B model (the 1B variant
# uses FP8 input projections that require an H100), and it needs a CUDA device.
if not torch.cuda.is_available():
    pytest.skip(
        "Evo2 7B integration test requires a CUDA GPU",
        allow_module_level=True,
    )

from mrna_bench.models.evo2 import Evo2

# Evo2-7B-8K: hidden_size = 4096; embeddings concatenate the middle block's
# pre-norm representation with the final-layer hidden state -> 2 * 4096.
EVO2_VERSION = "Evo2-7B-8K"
HIDDEN_DIM = 4096
CONCAT_DIM = HIDDEN_DIM * 2


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device."""
    return torch.device("cuda")


@pytest.fixture(scope="module")
def model(device) -> Evo2:
    """Get Evo2 model (eager backend so attention weights are materializable)."""
    model = Evo2(EVO2_VERSION, device, "eager")
    model.set_inference_mode()
    return model


def test_evo2_forward(model):
    """Evo2 embeds to the concatenated middle-pre-norm + final dimension."""
    out = model.embed_sequence("ATGATG")
    assert out.shape == (1, CONCAT_DIM)


def test_evo2_max_length(model):
    """Evo2-7B-8K uses an 8192nt context window."""
    assert model.max_length == 8192


def test_evo2_middle_layer(model):
    """Middle block index is num_layers // 2 (16 for the 32-block 7B)."""
    assert model.middle_layer_idx == 16


@torch.no_grad()
def test_evo2_embed_batch(model):
    """Test batch embedding matches individual embeddings."""
    sequences = [
        "ATGATG" * 10,
        "ATGATG" * 50,
    ]

    batch_output = torch.stack(model.embed(sequences)).cpu()
    assert batch_output.shape == (2, CONCAT_DIM)

    for i, seq in enumerate(sequences):
        single_output = model.embed_sequence(seq).cpu()
        assert torch.allclose(
            batch_output[i:i + 1],
            single_output,
            atol=1e-2,
        ), "Mismatch at sequence {} (len {})".format(i, len(seq))


@torch.no_grad()
def test_evo2_embed_ragged_agg(model):
    """Test embed with identity agg_fn returns per-token embeddings (ragged)."""
    seqs = ["ATGATG", "GCGCGCGCGCGC"]
    out = model.embed(seqs, agg_fn=lambda x, **kwargs: x)
    assert out[0].dim() == 2  # (num_tokens, concat_dim)
    assert out[1].dim() == 2
    assert out[0].shape[0] != out[1].shape[0]  # ragged: different token counts
    assert out[0].shape[1] == out[1].shape[1] == CONCAT_DIM  # same feature dim


def test_evo2_gradient_flow(model):
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


def test_evo2_extract_structure(model):
    """extract() returns (dict, dict) with matching keys; hidden states are 2D."""
    h, s = model.extract(["ATGATG"], layers=[0])
    assert isinstance(h, dict) and isinstance(s, dict)
    assert set(h.keys()) == set(s.keys())
    layer = next(iter(h))
    assert h[layer][0][0].dim() == 2
    assert h[layer][0][0].device.type == "cpu"


def test_evo2_extract_layer_selection(model):
    """Requesting layers=[0] returns exactly 1 layer."""
    h, _ = model.extract(["ATGATG"], layers=[0])
    assert len(h) == 1


def test_evo2_extract_transformer_attention(model):
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
    # Evo2 is autoregressive: attention is causal (strict upper triangle == 0).
    seq_len = w.shape[1]
    upper = w.float()[:, torch.triu(torch.ones(seq_len, seq_len), 1).bool()]
    assert upper.abs().max() == 0.0
    # Rows are a softmax distribution (sum to 1 within bf16 tolerance).
    rowsum = w.float().sum(-1)
    assert torch.allclose(rowsum, torch.ones_like(rowsum), atol=1e-2)


def test_evo2_extract_hyena_scores_none(model):
    """Hyena/SSM blocks always return None scores regardless of return_attentions."""
    _, s = model.extract(["ATGCATGC"], layers=["blocks.0"], return_attentions=True)
    assert s["blocks.0"] is None


def test_evo2_peft_target(model):
    """get_peft_target returns the HF model; set_peft_target writes it back."""
    target = model.get_peft_target()
    assert target is model.model

    original = model.model
    sentinel = torch.nn.Linear(1, 1)
    model.set_peft_target(sentinel)
    assert model.model is sentinel

    model.set_peft_target(original)  # restore for any subsequent use
    assert model.model is original
