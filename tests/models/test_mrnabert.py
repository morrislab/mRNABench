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


def make_cds(sequence: str, utr5_len: int, utr3_len: int) -> np.ndarray:
    """Mark codon starts between the UTRs."""
    cds = np.zeros(len(sequence), dtype=np.int32)
    cds[utr5_len:len(sequence) - utr3_len:3] = 1
    return cds


def test_mrnabert_requires_cds(mrnabert):
    """Test mRNABERT raises error when CDS is not provided."""
    with pytest.raises(ValueError, match="mRNABERT requires cds"):
        mrnabert.embed(["ATGATG"])


def test_mrnabert_forward_batch(mrnabert):
    """Test mRNABERT embeds a ragged batch with CDS tracks."""
    sequences = [
        "GGGAAA" + "ATGGCATAA" + "CCCGGG",
        "ACGT" + "ATG" + "GCA" * 10 + "TAA" + "TTTT",
    ]
    cds = [make_cds(sequences[0], 6, 6), make_cds(sequences[1], 4, 4)]

    out = mrnabert.embed(sequences, cds=cds)

    assert out.shape == (2, 768)
