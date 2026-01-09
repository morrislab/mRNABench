import pytest

import numpy as np

pytest.importorskip("torch")
import torch
from mrna_bench.models.naive_baseline import NaiveBaseline


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
def model_6track(device) -> NaiveBaseline:
    """Get NaiveBaseline 6-track model."""
    return NaiveBaseline("naive-6-track", device)


def test_naive_baseline_4track_forward(model_4track):
    """Test NaiveBaseline 4-track forward pass."""
    out = model_4track.embed_sequence("ATGATG")
    # k-mer counts (3-7) + GC content
    # vocab size = 4^3 + 4^4 + 4^5 + 4^6 + 4^7 = 64 + 256 + 1024 + 4096 + 16384
    # = 21824, plus 1 for GC content = 21825
    assert out.shape == (1, 21825)


def test_naive_baseline_6track_forward(model_6track):
    """Test NaiveBaseline 6-track forward pass."""
    sequence = "ATGATG"
    cds = np.array([1, 0, 0, 1, 0, 0])
    splice = np.array([0, 0, 0, 0, 0, 0])

    out = model_6track.embed_sequence_sixtrack(sequence, cds, splice)
    # k-mer counts + GC content + cds_length + exon_count = 21825 + 2 = 21827
    assert out.shape == (1, 21827)


def test_naive_baseline_is_sixtrack_flag(model_4track, model_6track):
    """Test is_sixtrack flag is set correctly."""
    assert model_4track.is_sixtrack is False
    assert model_6track.is_sixtrack is True


def test_naive_baseline_gc_content(model_4track):
    """Test GC content calculation."""
    # All G and C should give GC ratio of 1.0
    out_gc = model_4track.embed_sequence("GCGCGC")
    gc_value = out_gc[0, -1].item()
    assert gc_value == 1.0

    # All A and T should give GC ratio of 0.0
    out_at = model_4track.embed_sequence("ATATAT")
    at_value = out_at[0, -1].item()
    assert at_value == 0.0


def test_naive_baseline_cds_length(model_6track):
    """Test CDS length calculation."""
    sequence = "ATGATGATG"
    # CDS from position 3 to 6 (codon at 3, ends at 6)
    cds = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0])
    splice = np.array([0] * 9)

    out = model_6track.embed_sequence_sixtrack(sequence, cds, splice)
    # CDS length should be 3 (one codon)
    cds_length = out[0, -2].item()
    assert cds_length == 3.0


def test_naive_baseline_exon_count(model_6track):
    """Test exon count calculation."""
    sequence = "ATGATGATG"
    cds = np.array([0] * 9)
    # Two splice sites
    splice = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0])

    out = model_6track.embed_sequence_sixtrack(sequence, cds, splice)
    exon_count = out[0, -1].item()
    assert exon_count == 2.0


def test_naive_baseline_no_cds(model_6track):
    """Test behavior when no CDS is present."""
    sequence = "ATGATG"
    cds = np.array([0, 0, 0, 0, 0, 0])
    splice = np.array([0, 0, 0, 0, 0, 0])

    out = model_6track.embed_sequence_sixtrack(sequence, cds, splice)
    cds_length = out[0, -2].item()
    assert cds_length == 0.0


def test_naive_baseline_agg_fn_not_supported(model_4track, model_6track):
    """Test that custom agg_fn raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        model_4track.embed_sequence("ATGATG", agg_fn=torch.sum)

    with pytest.raises(NotImplementedError):
        model_6track.embed_sequence_sixtrack(
            "ATGATG",
            np.array([0] * 6),
            np.array([0] * 6),
            agg_fn=torch.sum
        )


def test_generate_vocab():
    """Test k-mer vocabulary generation."""
    vocab = NaiveBaseline.generate_vocab(kmer_list=[3], alphabet="AC")
    expected = ["AAA", "AAC", "ACA", "ACC", "CAA", "CAC", "CCA", "CCC"]
    assert vocab == expected
