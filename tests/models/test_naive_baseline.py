import pytest

import numpy as np

pytest.importorskip("torch")
import torch
from mrna_bench.models.naive_baseline import (
    NaiveBaseline,
    NaiveBaselineSixTrack,
    generate_kmer_vocab,
)


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def model_4track(device) -> NaiveBaseline:
    """Get NaiveBaseline 4-track model."""
    return NaiveBaseline("naive-4-track", device)


@pytest.fixture(scope="module")
def model_6track(device) -> NaiveBaselineSixTrack:
    """Get NaiveBaselineSixTrack model."""
    return NaiveBaselineSixTrack("naive-6-track", device)


def test_naive_baseline_4track_forward(model_4track):
    """Test NaiveBaseline 4-track forward pass."""
    out = model_4track.embed(["ATGATG"])
    # k-mer counts (3-7) + GC content
    # vocab size = 4^3 + 4^4 + 4^5 + 4^6 + 4^7 = 64 + 256 + 1024 + 4096 + 16384
    # = 21824, plus 1 for GC content = 21825
    assert out.shape == (1, 21825)


def test_naive_baseline_6track_forward(model_6track):
    """Test NaiveBaselineSixTrack forward pass."""
    sequence = "ATGATG"
    cds = np.array([1, 0, 0, 1, 0, 0])
    splice = np.array([0, 0, 0, 0, 0, 0])

    out = model_6track.embed([sequence], cds=[cds], splice=[splice])
    # k-mer counts + GC content + cds_length + exon_count = 21825 + 2 = 21827
    assert out.shape == (1, 21827)


def test_naive_baseline_gc_content(model_4track):
    """Test GC content calculation."""
    # All G and C should give GC ratio of 1.0
    out_gc = model_4track.embed(["GCGCGC"])
    gc_value = out_gc[0, -1].item()
    assert gc_value == 1.0

    # All A and T should give GC ratio of 0.0
    out_at = model_4track.embed(["ATATAT"])
    at_value = out_at[0, -1].item()
    assert at_value == 0.0


def test_naive_baseline_cds_length(model_6track):
    """Test CDS length calculation."""
    sequence = "ATGATGATG"
    # CDS from position 3 to 6 (codon at 3, ends at 6)
    cds = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0])
    splice = np.array([0] * 9)

    out = model_6track.embed([sequence], cds=[cds], splice=[splice])
    # CDS length should be 3 (one codon)
    cds_length = out[0, -2].item()
    assert cds_length == 3.0


def test_naive_baseline_exon_count(model_6track):
    """Test exon count calculation."""
    sequence = "ATGATGATG"
    cds = np.array([0] * 9)
    # Two splice sites
    splice = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0])

    out = model_6track.embed([sequence], cds=[cds], splice=[splice])
    exon_count = out[0, -1].item()
    assert exon_count == 2.0


def test_naive_baseline_no_cds(model_6track):
    """Test behavior when no CDS is present."""
    sequence = "ATGATG"
    cds = np.array([0, 0, 0, 0, 0, 0])
    splice = np.array([0, 0, 0, 0, 0, 0])

    out = model_6track.embed([sequence], cds=[cds], splice=[splice])
    cds_length = out[0, -2].item()
    assert cds_length == 0.0


def test_naive_baseline_6track_requires_tracks(model_6track):
    """Test that NaiveBaselineSixTrack raises error without tracks."""
    with pytest.raises(ValueError):
        model_6track.embed(["ATGATG"])

    with pytest.raises(ValueError):
        model_6track.embed(["ATGATG"], cds=[np.array([0] * 6)])


def test_generate_vocab():
    """Test k-mer vocabulary generation."""
    vocab = generate_kmer_vocab(kmer_list=[3], alphabet="AC")
    expected = ["AAA", "AAC", "ACA", "ACC", "CAA", "CAC", "CCA", "CCC"]
    assert vocab == expected


def test_naive_baseline_batch(model_4track):
    """Test batch embed matches individual embeddings."""
    sequences = ["ATGATG", "GCGCGC", "ATATAT"]

    batch_output = model_4track.embed(sequences)
    assert batch_output.shape == (3, 21825)

    for i, seq in enumerate(sequences):
        single_output = model_4track.embed([seq])
        assert torch.allclose(
            batch_output[i:i + 1],
            single_output,
            atol=1e-6
        ), "Mismatch at sequence {}".format(i)


def test_naive_baseline_6track_batch(model_6track):
    """Test batch embed matches individual embeddings for 6-track."""
    sequences = ["ATGATG", "GCGCGC"]
    cds_list = [np.array([1, 0, 0, 1, 0, 0]), np.array([0, 0, 0, 0, 0, 0])]
    splice_list = [np.array([0, 0, 0, 0, 0, 0]), np.array([1, 0, 0, 0, 0, 0])]

    batch_output = model_6track.embed(sequences, cds=cds_list, splice=splice_list)
    assert batch_output.shape == (2, 21827)

    for i in range(len(sequences)):
        single_output = model_6track.embed(
            [sequences[i]],
            cds=[cds_list[i]],
            splice=[splice_list[i]]
        )
        assert torch.allclose(
            batch_output[i:i + 1],
            single_output,
            atol=1e-6
        ), "Mismatch at sequence {}".format(i)
