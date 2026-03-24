import pytest

import numpy as np

pytest.importorskip("torch")
pytest.importorskip("mamba_ssm")
import torch
from mrna_bench.models.naive_mamba import NaiveMamba


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def model(device) -> NaiveMamba:
    """Get NaiveMamba model."""
    return NaiveMamba("naive-mamba", device)


def test_naive_mamba_forward(model):
    """Test NaiveMamba forward pass with batch."""
    sequences = ["ATGATG", "ATGATGATG"]
    cds = [np.array([1, 0, 0, 1, 0, 0]), np.array([1, 0, 0, 1, 0, 0, 1, 0, 0])]
    splice = [np.array([0, 0, 0, 0, 0, 0]), np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])]

    out = model.embed(sequences, cds, splice)
    assert out.shape == (2, 64)


def test_naive_mamba_single_sequence(model):
    """Test NaiveMamba with single sequence via embed_sequence."""
    sequence = "ATGATG"
    cds = np.array([1, 0, 0, 1, 0, 0])
    splice = np.array([0, 0, 0, 0, 0, 0])

    out = model.embed_sequence(sequence, cds, splice)
    assert out.shape == (1, 64)


def test_naive_mamba_requires_cds_splice(model):
    """Test NaiveMamba raises error without cds/splice tracks."""
    with pytest.raises(ValueError):
        model.embed(["ATGATG"])

    with pytest.raises(ValueError):
        model.embed(["ATGATG"], cds=[np.array([1, 0, 0, 1, 0, 0])])

    with pytest.raises(ValueError):
        model.embed(["ATGATG"], splice=[np.array([0, 0, 0, 0, 0, 0])])


def test_naive_mamba_batch_ragged(model):
    """Test that batch embedding matches individual embeddings."""
    sequences = ["ATGATG", "ATGATGATGATGATGATG"]
    cds = [
        np.array([1, 0, 0, 1, 0, 0]),
        np.array([1, 0, 0] * 6)
    ]
    splice = [
        np.array([0] * 6),
        np.array([0] * 18)
    ]

    batch_out = model.embed(sequences, cds, splice)

    individual_outs = []
    for seq, c, s in zip(sequences, cds, splice):
        out = model.embed([seq], [c], [s])
        individual_outs.append(out)

    individual_stacked = torch.cat(individual_outs, dim=0)

    assert torch.allclose(batch_out, individual_stacked, atol=1e-5)


def test_naive_mamba_custom_agg_fn(model):
    """Test NaiveMamba with custom aggregation function."""
    sequence = "ATGATG"
    cds = np.array([1, 0, 0, 1, 0, 0])
    splice = np.array([0, 0, 0, 0, 0, 0])

    out_mean = model.embed([sequence], [cds], [splice], agg_fn=torch.mean)
    out_sum = model.embed([sequence], [cds], [splice], agg_fn=torch.sum)

    assert out_mean.shape == (1, 64)
    assert out_sum.shape == (1, 64)
    assert not torch.allclose(out_mean, out_sum)


def test_naive_mamba_gradient_flow(model):
    """Test that gradients can flow through the model."""
    model.set_train_mode()

    cds = np.array([1, 0, 0, 1, 0, 0])
    splice = np.array([0, 0, 0, 0, 0, 0])
    out = model.embed(["ATGATG"], cds=[cds], splice=[splice])
    assert out.requires_grad, "Output should require gradients"

    loss = out.sum()
    loss.backward()

    has_grad = False
    for param in model.model.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break

    assert has_grad, "No gradients flowed to model parameters"
