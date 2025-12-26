import pytest

pytest.importorskip("torch")
pytest.importorskip("evo")
import torch
from mrna_bench.models.evo1 import Evo1


EVO1_VERSIONS = [
    "evo-1.5-8k-base",
    "evo-1-8k-base",
]


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module", params=EVO1_VERSIONS, ids=lambda x: x)
def model(request, device) -> Evo1:
    """Get Evo1 model."""
    return Evo1(request.param, device)


def test_evo1_forward(model):
    """Test Evo1 forward pass."""
    out = model.embed_sequence("ATGATG")
    assert out.shape == (1, 4096)


def test_evo1_max_length(device):
    """Test Evo1 max_length is set correctly."""
    model_8k = Evo1("evo-1-8k-base", device)
    assert model_8k.max_length == 8192


def test_evo1_sixtrack_not_supported(model):
    """Test Evo1 sixtrack raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        model.embed_sequence_sixtrack("ATGATG", None, None, None)
