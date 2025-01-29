import pytest

import numpy as np

pytest.importorskip("torch")
import torch

from mrna_bench.models.orthrus import Orthrus


@pytest.fixture
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def test_orthrus_forward_six(device):
    """Test Orthrus forward pass using six track input."""
    model = Orthrus("orthrus-large-6-track", device)

    out = model.embed_sequence_sixtrack(
        "ATG",
        np.array([0, 0, 0]),
        np.array([0, 0, 0])
    )

    assert out.shape == (1, 512)

def test_orthrus_forward_four(device):
    """Test Orthrus forward pass using four track input."""
    model = Orthrus("orthrus-base-4-track", device)

    out = model.embed_sequence("ATG")

    assert out.shape == (1, 256)


def test_orthrus_no_overlap(device):
    """Test Orthrus throws error when using overlap."""
    model_6 = Orthrus("orthrus-large-6-track", device)

    with pytest.raises(ValueError):
        model_6.embed_sequence_sixtrack(
            "ATG",
            np.array([0, 0, 0]),
            np.array([0, 0, 0]),
            overlap=1
        )

    model_4 = Orthrus("orthrus-base-4-track", device)

    with pytest.raises(ValueError):
        model_4.embed_sequence(
            "ATG",
            overlap=1
        )


def test_orthrus_correct_track_info(device):
    """Test Orthrus sets sixtrack flag correctly."""
    model_6 = Orthrus("orthrus-large-6-track", device)

    assert model_6.is_sixtrack is True

    model_4 = Orthrus("orthrus-base-4-track", device)

    assert model_4.is_sixtrack is False
