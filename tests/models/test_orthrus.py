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
    """Get Orthrus model."""
    return Orthrus("orthrus-large-6-track", device)


@pytest.fixture(scope="module")
def orthrus_4(device) -> Orthrus:
    """Get Orthrus model."""
    return Orthrus("orthrus-base-4-track", device)


def test_orthrus_forward_six(orthrus_6):
    """Test Orthrus forward pass using six track input."""
    out = orthrus_6.embed_sequence_sixtrack(
        "ATG",
        np.array([0, 0, 0]),
        np.array([0, 0, 0])
    )

    assert orthrus_6.is_sixtrack is True
    assert out.shape == (1, 512)


def test_orthrus_forward_four(orthrus_4):
    """Test Orthrus forward pass using four track input."""
    out = orthrus_4.embed_sequence("ATG")

    assert orthrus_4.is_sixtrack is False
    assert out.shape == (1, 256)


def test_orthrus_no_overlap(orthrus_4, orthrus_6):
    """Test Orthrus throws error when using overlap."""
    with pytest.raises(ValueError):
        orthrus_6.embed_sequence_sixtrack(
            "ATG",
            np.array([0, 0, 0]),
            np.array([0, 0, 0]),
            overlap=1
        )

    with pytest.raises(ValueError):
        orthrus_4.embed_sequence(
            "ATG",
            overlap=1
        )
