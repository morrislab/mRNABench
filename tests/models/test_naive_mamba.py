import pytest

import numpy as np

pytest.importorskip("torch")
pytest.importorskip("mamba_ssm")
import torch

from tests.model_utils import (
    assert_pooled_batch_matches_single,
    assert_raw_batch_matches_single,
    embed_one,
)
from mrna_bench.models.naive_mamba import NaiveMamba


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def model(device) -> NaiveMamba:
    """Get NaiveMamba model."""
    model = NaiveMamba("naive-mamba", device, None)
    model.set_inference_mode()
    return model


def make_cds(seq: str) -> np.ndarray:
    """Make CDS array marking every codon start (assumes pure CDS input)."""
    arr = np.zeros(len(seq), dtype=int)
    arr[::3] = 1
    return arr


def make_splice(seq: str) -> np.ndarray:
    """Make all-zero splice array."""
    return np.zeros(len(seq), dtype=int)


def test_naive_mamba_single_sequence(model):
    """Test NaiveMamba with one sequence."""
    sequence = "ATGATG"
    out = embed_one(model, sequence, make_cds(sequence), make_splice(sequence))
    assert out.shape == (1, 64)


def test_naive_mamba_requires_cds_splice(model):
    """Test NaiveMamba raises error without cds/splice tracks."""
    seq = "ATGATG"
    with pytest.raises(ValueError):
        model.embed([seq])

    with pytest.raises(ValueError):
        model.embed([seq], cds=[make_cds(seq)])

    with pytest.raises(ValueError):
        model.embed([seq], splice=[make_splice(seq)])


def test_naive_mamba_embed_batch_ragged(model):
    """Test ragged batches match individual embeddings."""
    sequences = ["ATGATG", "ATGATGATGATGATGATG"]
    cds = [make_cds(s) for s in sequences]
    splice = [make_splice(s) for s in sequences]

    assert_pooled_batch_matches_single(
        model,
        sequences,
        cds=cds,
        splice=splice,
    )


def test_naive_mamba_custom_agg_fn(model):
    """Test NaiveMamba with custom aggregation function."""
    from functools import partial
    seq = "ATGATG"
    cds = make_cds(seq)
    splice = make_splice(seq)

    out_mean = torch.stack(model.embed([seq], [cds], [splice], agg_fn=partial(torch.mean, dim=0)))
    out_sum = torch.stack(model.embed([seq], [cds], [splice], agg_fn=partial(torch.sum, dim=0)))

    assert out_mean.shape == (1, 64)
    assert out_sum.shape == (1, 64)
    assert not torch.allclose(out_mean, out_sum)


@torch.no_grad()
def test_naive_mamba_embed_ragged_agg(model):
    """Test embed with identity agg_fn returns per-token embeddings (ragged)."""
    seqs = ["ATGATG", "GCGCGCGCGCGC"]
    cds = [make_cds(s) for s in seqs]
    splice = [make_splice(s) for s in seqs]
    out = model.embed(seqs, cds=cds, splice=splice, agg_fn=lambda x, **kwargs: x)
    assert_raw_batch_matches_single(
        model, seqs, out, cds=cds, splice=splice
    )
    assert out[0].dim() == 2  # (num_tokens, hidden_dim)
    assert out[1].dim() == 2
    assert out[0].shape[0] != out[1].shape[0]  # ragged: different token counts
    assert out[0].shape[1] == out[1].shape[1]  # same hidden dim


def test_naive_mamba_gradient_flow(model):
    """Test that gradients can flow through the model."""
    model.set_train_mode()

    seq = "ATGATG"
    out = model.embed([seq], cds=[make_cds(seq)], splice=[make_splice(seq)])
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

def test_naive_mamba_extract_structure(model):
    """extract() returns (dict, dict) with matching keys; hidden states are 2D."""
    seq = "ATGATG"
    cds = [make_cds(seq)]
    splice = [make_splice(seq)]
    h, s = model.extract([seq], cds=cds, splice=splice, layers=[0])
    assert isinstance(h, dict) and isinstance(s, dict)
    assert set(h.keys()) == set(s.keys())
    layer = next(iter(h))
    assert h[layer][0][0].dim() == 2
    assert h[layer][0][0].device.type == "cpu"


def test_naive_mamba_extract_layer_selection(model):
    """Requesting layers=[0] returns exactly 1 layer."""
    seq = "ATGATG"
    h, _ = model.extract([seq], cds=[make_cds(seq)], splice=[make_splice(seq)], layers=[0])
    assert len(h) == 1


def test_naive_mamba_extract_scores_none(model):
    """NaiveMamba scores are None for all layers (SSM, no attention matrix)."""
    seq = "ATGATG"
    _, s = model.extract(
        [seq], cds=[make_cds(seq)], splice=[make_splice(seq)],
        layers=[0], return_attentions=True
    )
    assert all(v is None for v in s.values())
