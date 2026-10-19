import pytest

import numpy as np

pytest.importorskip("torch")
import torch

from mrna_bench.models.mrnabert import mRNABERT


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def mrnabert(device) -> mRNABERT:
    """Get mRNABERT model."""
    return mRNABERT("mRNABERT", device)


def test_mrnabert_forward_six(mrnabert):
    """Test mRNABERT forward pass using six track input."""
    sequence = "GGGAAA" + "ATGGCATAA" + "CCCGGG"
    cds = np.zeros(len(sequence), dtype=np.int32)
    cds[6:15:3] = 1

    out = mrnabert.embed_sequence_sixtrack(
        sequence, cds, np.zeros(len(sequence))
    )

    assert mrnabert.is_sixtrack is True
    assert out.shape == (1, 768)


def test_mrnabert_four_track_unsupported(mrnabert):
    """Test mRNABERT rejects four track input."""
    with pytest.raises(NotImplementedError):
        mrnabert.embed_sequence("ATG", torch.mean)
