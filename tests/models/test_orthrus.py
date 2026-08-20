import pytest

import numpy as np

pytest.importorskip("torch")
import torch

from tests.model_utils import (
    assert_pooled_batch_matches_single,
    assert_raw_batch_matches_single,
    embed_one,
)

from mrna_bench.models.orthrus import Orthrus


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def orthrus_6(device) -> Orthrus:
    """Get Orthrus 6-track model."""
    model = Orthrus("orthrus-large-6-track", device, None)
    model.set_inference_mode()
    return model


@pytest.fixture(scope="module")
def orthrus_4(device) -> Orthrus:
    """Get Orthrus 4-track model."""
    model = Orthrus("orthrus-large-4-track", device, None)
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


def test_orthrus_single_sequence_six(orthrus_6):
    """Test Orthrus 6-track with one sequence."""
    sequence = "ATG"
    out = embed_one(orthrus_6, sequence, make_cds(sequence), make_splice(sequence))
    assert out.shape == (1, 512)


def test_orthrus_single_sequence_four(orthrus_4):
    """Test Orthrus 4-track with one sequence."""
    sequence = "ATG"
    out = embed_one(orthrus_4, sequence)
    assert out.shape == (1, 512)


def test_orthrus_six_requires_tracks(orthrus_6):
    """Test Orthrus 6-track raises error without cds/splice tracks."""
    seq = "ATG"
    with pytest.raises(ValueError):
        orthrus_6.embed([seq])

    with pytest.raises(ValueError):
        orthrus_6.embed([seq], cds=[make_cds(seq)])

    with pytest.raises(ValueError):
        orthrus_6.embed([seq], splice=[make_splice(seq)])


def test_orthrus_four_ignores_tracks(orthrus_4):
    """Test Orthrus 4-track ignores cds/splice tracks if provided."""
    sequence = "ATG"
    cds = [make_cds(sequence)]
    splice = [make_splice(sequence)]

    out_without = torch.stack(orthrus_4.embed([sequence]))
    out_with = torch.stack(orthrus_4.embed([sequence], cds, splice))

    assert torch.allclose(out_without, out_with)


def test_orthrus_batch_ragged_six(orthrus_6):
    """Test 6-track batch embedding matches individual embeddings."""
    sequences = ["ATG", "ATGATGATG"]
    cds = [make_cds(s) for s in sequences]
    splice = [make_splice(s) for s in sequences]

    assert_pooled_batch_matches_single(
        orthrus_6,
        sequences,
        cds=cds,
        splice=splice,
    )


def test_orthrus_batch_ragged_four(orthrus_4):
    """Test 4-track batch embedding matches individual embeddings."""
    sequences = ["ATG", "ATGATGATG"]

    assert_pooled_batch_matches_single(
        orthrus_4,
        sequences,
    )


@torch.no_grad()
def test_orthrus_embed_ragged_agg(orthrus_4):
    """Test embed with identity agg_fn returns per-token embeddings (ragged)."""
    seqs = ["ATGATG", "GCGCGCGCGCGC"]
    out = orthrus_4.embed(seqs, agg_fn=lambda x, **kwargs: x)
    assert_raw_batch_matches_single(orthrus_4, seqs, out)
    assert out[0].dim() == 2  # (num_tokens, hidden_dim)
    assert out[1].dim() == 2
    assert out[0].shape[0] != out[1].shape[0]  # ragged: different token counts
    assert out[0].shape[1] == out[1].shape[1]  # same hidden dim


def test_orthrus_gradient_flow(orthrus_4):
    """Test that gradients can flow through the model."""
    orthrus_4.set_train_mode()

    out = orthrus_4.embed(["ATGATG"])
    assert out[0].requires_grad, "Output should require gradients"

    loss = torch.stack(out).sum()
    loss.backward()

    has_grad = False
    for param in orthrus_4.model.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break

    assert has_grad, "No gradients flowed to model parameters"
    orthrus_4.set_inference_mode()

def test_orthrus_extract_structure(orthrus_6):
    """extract() returns (dict, dict) with matching keys; hidden states are 2D."""
    seq = "ATGATG"
    cds = [make_cds(seq)]
    splice = [make_splice(seq)]
    h, s = orthrus_6.extract([seq], cds=cds, splice=splice, layers=[0])
    assert isinstance(h, dict) and isinstance(s, dict)
    assert set(h.keys()) == set(s.keys())
    layer = next(iter(h))
    assert h[layer][0][0].dim() == 2
    assert h[layer][0][0].device.type == "cpu"


def test_orthrus_extract_layer_selection(orthrus_6):
    """Requesting layers=[0] returns exactly 1 layer."""
    seq = "ATGATG"
    h, _ = orthrus_6.extract([seq], cds=[make_cds(seq)], splice=[make_splice(seq)], layers=[0])
    assert len(h) == 1


def test_orthrus_extract_scores_none(orthrus_6):
    """Orthrus scores are None for all layers (Mamba SSM, no attention matrix)."""
    seq = "ATGATG"
    _, s = orthrus_6.extract(
        [seq], cds=[make_cds(seq)], splice=[make_splice(seq)],
        layers=[0], return_attentions=True
    )
    assert all(v is None for v in s.values())
