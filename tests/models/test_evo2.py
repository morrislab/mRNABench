import pytest
import numpy as np


import torch
from mrna_bench.models.evo2 import Evo2

# pytestmark = pytest.mark.skip(reason="temporary skip")

EVO2_VERSIONS = [
    "evo2_7b",
    "evo2_7b_262k",
    "evo2_7b_base",
    "evo2_1b_base"
]


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module", params=EVO2_VERSIONS, ids=lambda x: x)
def model(request, device) -> Evo2:
    """Get Evo2 model."""
    return Evo2(request.param, device)


def test_evo2_forward_small(model):
    """Forward pass on a short sequence returns correct shape."""
    seq = "ACGTAC"
    out = model.embed_sequence(seq)

    assert isinstance(out, torch.Tensor)
    assert out.ndim == 2 # shape (1, H)
    assert out.shape[0] == 1 # batch dim OK
    assert out.shape[1] > 0 # has embedding dimension


def test_evo2_max_length_setting(model):
    """Max length updates correctly depending on version."""
    if model.model_version in ["evo2_40b", "evo2_7b"]:
        assert model.max_length == 1_000_000
    elif model.model_version == "evo2_7b_262k":
        assert model.max_length == 262_144
    else:
        assert model.max_length == 8_192

def test_evo2_sixtrack_not_supported(model):
    """Six-track embedding should raise NotImplementedError."""
    with pytest.raises(NotImplementedError):
        model.embed_sequence_sixtrack("ATG", np.array([0]), np.array([0]))
