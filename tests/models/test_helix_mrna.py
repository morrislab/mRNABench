import pytest

import numpy as np
import torch

from mrna_bench.models.helix_mrna import HelixmRNAWrapper


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def helix_mrna(device) -> HelixmRNAWrapper:
    """Get Helix-mRNA model."""
    return HelixmRNAWrapper("helix-mrna", device)


def test_helix_mrna_forward(helix_mrna):
    """Test Helix-mRNA initialization and forward pass."""
    assert helix_mrna.is_sixtrack is True

    out = helix_mrna.embed_sequence("ATGATG")
    assert out.shape == (1, 256)


def test_helix_mrna_dna_to_rna(helix_mrna):
    """Test Helix-mRNA sequence conversion from DNA to RNA."""
    assert helix_mrna.convert_dna_to_rna("ATGATG") == "AUGAUG"


def test_helix_mrna_converter(helix_mrna):
    """Test Helix-mRNA sequence converter."""
    seq = "AUGUAG"
    cds_1 = np.array([0, 0, 0, 1, 0, 0])
    cds_2 = np.array([0, 0, 0, 0, 1, 1])
    cds_3 = np.array([0, 0, 0, 0, 0, 0])

    out_1 = helix_mrna.tokenize_cds(seq, cds_1)
    out_2 = helix_mrna.tokenize_cds(seq, cds_2)
    out_3 = helix_mrna.tokenize_cds(seq, cds_3)

    assert out_1 == "AUGEUAG"
    assert out_2 == "AUGUEAEG"
    assert out_3 == "AUGUAG"
