import pytest
from unittest.mock import patch


pytest.importorskip("torch")
import torch

from mrna_bench.models.rinalmo import RiNALMo


RINALMO_VERSIONS = [
    ("rinalmo-giga", 1280),
    ("rinalmo-mega", 640),
    ("rinalmo-micro", 480),
]


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module", params=RINALMO_VERSIONS, ids=lambda x: x[0])
def rinalmo(request, device):
    """Get RiNALMo model and expected embedding dim."""
    model_version, embed_dim = request.param
    model = RiNALMo(model_version, device)
    return model, embed_dim


@pytest.fixture(scope="module")
def rinalmo_giga(device) -> RiNALMo:
    """Get RiNALMo giga model for specific output tests."""
    return RiNALMo("rinalmo-giga", device)


def test_rinalmo_forward(rinalmo):
    """Test RiNALMo forward pass."""
    model, embed_dim = rinalmo
    assert model.is_sixtrack is False

    text = "ACTTTGGCCA"
    output = model.embed_sequence(text, agg_fn=torch.mean).cpu()
    assert output.shape == (1, embed_dim)


def test_rinalmo_giga_output(rinalmo_giga):
    """Test RiNALMo giga produces expected output."""
    text = "ACTTTGGCCA"
    output = rinalmo_giga.embed_sequence(text, agg_fn=torch.mean).cpu()
    assert output.shape == (1, 1280)

    # Matches output from official release
    assert torch.allclose(
        torch.Tensor([-0.00032]),
        torch.mean(output),
        atol=0.0001
    )


def test_rinalmo_forward_converts_tu(rinalmo):
    """Test that RiNALMo forward pass automatically converts T->U."""
    model, _ = rinalmo
    text = "ACTTTGGCCA"
    with patch.object(
        model,
        "tokenizer",
        side_effect=model.tokenizer
    ) as mock:
        model.embed_sequence(text)
        mock.assert_called_once_with("ACUUUGGCCA", return_tensors="pt")
