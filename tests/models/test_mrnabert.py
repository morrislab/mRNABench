import numpy as np

import pytest
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
    return mRNABERT("mRNA-BERT", device)


def test_mrna_bert_forward_six(mrnabert):
    """Test mRNABERT forward pass using six track input."""
    out = mrnabert.embed_sequence_sixtrack(
        "ATG",
        np.array([0, 0, 0]),
        np.array([0, 0, 0])
    )

    assert mrnabert.is_sixtrack is True
    assert out.shape == (1, 768)


def test_single_codon(mrnabert):
    """UTR-CDS-UTR structure is spaced correctly for a single codon."""
    seq = "AACTGCGTG"
    cds = np.array([0,0,1,0,0,0,0,0,0])
    assert mrnabert.separate_utr_cds(seq, cds) == ["A A CTG C G T G"]


def test_multiple_codons(mrnabert):
    """Multiple contiguous CDS codons should be spaced in triples."""
    seq = "AACTGAAACCCGG"
    cds = np.array([0,0,1,0,0,1,0,0,0,0,0,0,0])
    assert mrnabert.separate_utr_cds(seq, cds) == ["A A CTG AAA C C C G G"]


def test_cds_at_start(mrnabert):
    """Handles CDS beginning at index 0, followed by UTR content."""
    seq = "ATGAAATTT"
    cds = np.array([1,0,0,1,0,0,0,0,0])
    assert mrnabert.separate_utr_cds(seq, cds) == ["ATG AAA T T T"]


def test_cds_at_end(mrnabert):
    """Handles CDS at the end of the sequence without trailing codon padding."""
    seq = "GGGATGAAA"
    cds = np.array([0,0,0,1,0,0,1,0,0])
    assert mrnabert.separate_utr_cds(seq, cds) == ["G G G ATG AAA"]


def test_no_cds(mrnabert):
    """If no CDS sites exist, the whole sequence should be spaced as UTR."""
    seq = "ACGTAC"
    cds = np.zeros(len(seq), dtype=int)
    assert mrnabert.separate_utr_cds(seq, cds) == ["A C G T A C"]


def test_truncated_last_codon(mrnabert):
    """Truncated codons at the end should not be extended or padded."""
    seq = "AAATGAA"
    cds = np.array([0,0,1,0,0,1,0])
    assert mrnabert.separate_utr_cds(seq, cds) == ["A A ATG AA"]

def test_cds_aware_chunking_does_not_split_codons(mrnabert):
    """Ensure chunking never splits CDS codons across chunk boundaries."""
    seq = "A" * 5 + "ATG" * 400
    cds = np.zeros(len(seq), dtype=int)

    for i in range(5, len(seq), 3):
        cds[i] = 1  # mark codon start

    chunks = mrnabert.chunk_sequence_cds_aware(
        seq,
        cds,
        chunk_length=200
    )

    # No chunk should end mid-codon (CDS length always multiple of 3)
    for chunk_seq, chunk_cds in chunks:
        starts = np.where(chunk_cds != 0)[0]
        if len(starts) == 0:
            continue
        cds_start = starts[0]
        cds_end = min(starts[-1] + 3, len(chunk_seq))
        assert (cds_end - cds_start) % 3 == 0
