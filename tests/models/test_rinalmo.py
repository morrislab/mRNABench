import pytest
from unittest.mock import patch


pytest.importorskip("torch")
import torch

from mrna_bench.models.rinalmo import RiNALMo


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def rinalmo(device) -> RiNALMo:
    """Get RiNALMo model."""
    return RiNALMo("rinalmo", device)


def test_rinalmo_forward(rinalmo):
    """Test RiNALMo forward pass."""
    assert rinalmo.is_sixtrack is False

    text = "ACTTTGGCCA"
    output = rinalmo.embed_sequence(text, agg_fn=torch.mean).cpu()
    assert output.shape == (1, 1280)

    # Matches output from official release
    assert torch.allclose(
        torch.Tensor([-0.00032]),
        torch.mean(output),
        atol=0.0001
    )


def test_rinalmo_errors_overlap(rinalmo):
    """Test RiNALMo throws error if overlap is not zero."""
    text = "ACTTTGGCCA"

    with pytest.raises(ValueError):
        rinalmo.embed_sequence(text, overlap=1, agg_fn=torch.mean).cpu()


def test_rinalmo_forward_converts_tu(rinalmo):
    """Test that RiNALMo forward pass automatically converts T->U."""
    text = "ACTTTGGCCA"
    with patch.object(
        rinalmo,
        "tokenizer",
        side_effect=rinalmo.tokenizer
    ) as mock:
        rinalmo.embed_sequence(text)
        mock.assert_called_once_with("ACUUUGGCCA", return_tensors="pt")
