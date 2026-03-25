import importlib.util

import pytest
from functools import partial

pytest.importorskip("torch")

# Use find_spec instead of importorskip so that the evo2 package (and
# TransformerEngine, which it imports at module level) is NOT loaded
# during pytest collection.  TE's module-level initialisation rewrites
# PyTorch's CUDA op dispatcher and corrupts the flash_attn schema used
# by Evo1/StripedHyena, causing all Evo1 tests to fail when evo2 is
# collected in the same session.  With find_spec, evo2 is imported only
# when the module-scoped fixture first runs - by which point all Evo1
# tests have already completed.
if importlib.util.find_spec("evo2") is None:
    pytest.skip("evo2 not installed", allow_module_level=True)

import torch
from mrna_bench.models.evo2 import Evo2


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def model(device) -> Evo2:
    """Get Evo2 model."""
    model = Evo2("evo2_1b_base", device)
    model.set_inference_mode()
    return model


def test_evo2_forward(model):
    """Test Evo2 forward pass."""
    out = model.embed_sequence("ATGATG")
    # Evo2 concatenates middle + last layer embeddings
    # evo2_1b_base has hidden_dim=1920, so 1920*2=3840
    assert out.shape == (1, 3840)


def test_evo2_max_length(device):
    """Test Evo2 max_length is set correctly for base variant."""
    model = Evo2("evo2_1b_base", device)
    assert model.max_length == 8192


@torch.no_grad()
def test_evo2_embed_batch(model):
    """Test batch embedding matches individual embeddings."""
    sequences = [
        "ATGATG" * 10,
        "ATGATG" * 50,
    ]

    batch_output = torch.stack(model.embed(sequences)).cpu()
    assert batch_output.shape == (2, 3840)

    for i, seq in enumerate(sequences):
        single_output = model.embed_sequence(seq).cpu()
        assert torch.allclose(
            batch_output[i:i + 1],
            single_output,
            atol=1e-5
        ), "Mismatch at sequence {} (len {})".format(i, len(seq))


@torch.no_grad()
def test_evo2_embed_ragged_agg(model):
    """Test embed with identity agg_fn returns per-token embeddings (ragged)."""
    seqs = ["ATGATG", "GCGCGCGCGCGC"]
    out = model.embed(seqs, agg_fn=lambda x, **kwargs: x)
    assert out[0].dim() == 2  # (num_tokens, hidden_dim)
    assert out[1].dim() == 2
    assert out[0].shape[0] != out[1].shape[0]  # ragged: different token counts
    assert out[0].shape[1] == out[1].shape[1]  # same hidden dim


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

