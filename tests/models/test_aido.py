from collections import namedtuple

import pytest
from unittest.mock import patch

pytest.importorskip("torch")
pytest.importorskip("modelgenerator")

import torch

from mrna_bench.models.aido import AIDORNA


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def aidorna(device) -> AIDORNA:
    """Get AIDO.RNA model."""
    return AIDORNA("aido_rna_650m", device)


def test_aidorna_forward(aidorna):
    """Test AIDO.RNA forward pass."""
    assert aidorna.is_sixtrack is False

    text = "ACTTGGCCA"
    output = aidorna.embed_sequence(text)
    assert output.shape == (1, 1280)
