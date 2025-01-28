import pytest

pytest.importorskip("torch")
import torch

from mrna_bench.models.rinalmo import RiNALMo


@pytest.fixture
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def test_rinalmo_forward(device):
    """Test RiNALMo initialization and forward pass."""
    text = "ACUUUGGCCA"
    model = RiNALMo("rinalmo", device)

    output = model.embed_sequence(text, agg_fn=torch.mean).cpu()
    assert output.shape == (1, 1280)

    # Matches output from official release
    assert torch.allclose(
        torch.Tensor([-0.00032]),
        torch.mean(output),
        atol=0.0001
    )
