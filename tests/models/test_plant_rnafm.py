from unittest.mock import patch

import pytest
import torch

from mrna_bench.models.plant_rnafm import PlantRNAFM


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def plant_rnafm(device) -> PlantRNAFM:
    """Get PlantRNAFM model."""
    return PlantRNAFM("plant_rnafm", device)


def test_plant_rnafm_forward(plant_rnafm):
    """Test PlantRNAFM forward pass."""
    assert plant_rnafm.is_sixtrack is False

    text = "ACUUGGCCA"
    output = plant_rnafm.embed_sequence(text)
    assert output.shape[0] == 1
    assert output.shape[1] == 480


def test_plant_rnafm_forward_conversion(plant_rnafm):
    """Test PlantRNAFM forward pass converts T->U."""
    text = "ACTTGGCCA"

    with patch.object(
        plant_rnafm,
        "chunk_sequence",
        side_effect=plant_rnafm.chunk_sequence
    ) as mock:
        plant_rnafm.embed_sequence(text)
        mock.assert_called_once_with("ACUUGGCCA", plant_rnafm.max_length - 2)
