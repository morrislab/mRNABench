import pytest

import numpy as np

pytest.importorskip("torch")
import torch

from mrna_bench.models.orthrus import Orthrus


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def orthrus_6(device) -> Orthrus:
    """Get Orthrus 6-track model."""
    return Orthrus("orthrus-large-6-track", device)


@pytest.fixture(scope="module")
def orthrus_4(device) -> Orthrus:
    """Get Orthrus 4-track model."""
    return Orthrus("orthrus-base-4-track", device)


def test_orthrus_forward_six(orthrus_6):
    """Test Orthrus 6-track forward pass with batch."""
    sequences = ["ATG", "ATGATG"]
    cds = [np.array([1, 0, 0]), np.array([1, 0, 0, 1, 0, 0])]
    splice = [np.array([0, 0, 0]), np.array([0, 0, 0, 0, 0, 0])]

    out = orthrus_6.embed(sequences, cds, splice)

    assert out.shape == (2, 512)


def test_orthrus_forward_four(orthrus_4):
    """Test Orthrus 4-track forward pass with batch."""
    sequences = ["ATG", "ATGATG"]

    out = orthrus_4.embed(sequences)

    assert out.shape == (2, 256)


def test_orthrus_single_sequence_six(orthrus_6):
    """Test Orthrus 6-track with single sequence via embed_sequence."""
    sequence = "ATG"
    cds = np.array([1, 0, 0])
    splice = np.array([0, 0, 0])

    out = orthrus_6.embed_sequence(sequence, cds, splice)

    assert out.shape == (1, 512)


def test_orthrus_single_sequence_four(orthrus_4):
    """Test Orthrus 4-track with single sequence via embed_sequence."""
    sequence = "ATG"

    out = orthrus_4.embed_sequence(sequence)

    assert out.shape == (1, 256)


def test_orthrus_six_requires_tracks(orthrus_6):
    """Test Orthrus 6-track raises error without cds/splice tracks."""
    with pytest.raises(ValueError):
        orthrus_6.embed(["ATG"])

    with pytest.raises(ValueError):
        orthrus_6.embed(["ATG"], cds=[np.array([1, 0, 0])])

    with pytest.raises(ValueError):
        orthrus_6.embed(["ATG"], splice=[np.array([0, 0, 0])])


def test_orthrus_four_ignores_tracks(orthrus_4):
    """Test Orthrus 4-track ignores cds/splice tracks if provided."""
    sequences = ["ATG"]
    cds = [np.array([1, 0, 0])]
    splice = [np.array([0, 0, 0])]

    out_without = orthrus_4.embed(sequences)
    out_with = orthrus_4.embed(sequences, cds, splice)

    assert torch.allclose(out_without, out_with)


def test_orthrus_batch_ragged_six(orthrus_6):
    """Test 6-track batch embedding matches individual embeddings."""
    sequences = ["ATG", "ATGATGATG"]
    cds = [np.array([1, 0, 0]), np.array([1, 0, 0, 1, 0, 0, 1, 0, 0])]
    splice = [np.array([0, 0, 0]), np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])]

    batch_out = orthrus_6.embed(sequences, cds, splice)

    individual_outs = []
    for seq, c, s in zip(sequences, cds, splice):
        out = orthrus_6.embed([seq], [c], [s])
        individual_outs.append(out)

    individual_stacked = torch.cat(individual_outs, dim=0)

    assert torch.allclose(batch_out, individual_stacked, atol=1e-5)


def test_orthrus_batch_ragged_four(orthrus_4):
    """Test 4-track batch embedding matches individual embeddings."""
    sequences = ["ATG", "ATGATGATG"]

    batch_out = orthrus_4.embed(sequences)

    individual_outs = []
    for seq in sequences:
        out = orthrus_4.embed([seq])
        individual_outs.append(out)

    individual_stacked = torch.cat(individual_outs, dim=0)

    assert torch.allclose(batch_out, individual_stacked, atol=1e-5)


def test_orthrus_gradient_flow(orthrus_4):
    """Test that gradients can flow through the model."""
    orthrus_4.set_train_mode()

    out = orthrus_4.embed(["ATGATG"])
    assert out.requires_grad, "Output should require gradients"

    loss = out.sum()
    loss.backward()

    has_grad = False
    for param in orthrus_4.model.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break

    assert has_grad, "No gradients flowed to model parameters"
