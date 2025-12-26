import numpy as np
import pytest

from mrna_bench.datasets.dataset_utils import (
    create_cds_track,
    create_sequence,
    create_splice_track,
    get_top_n_priority_transcripts,
    ohe_to_str,
    str_to_ohe,
)


def test_str_to_ohe():
    """Test str_to_ohe function."""
    out_1 = str_to_ohe("ACGTN")

    assert out_1.shape == (5, 4)
    assert out_1.sum() == 4

    assert out_1[0, 0] == 1
    assert out_1[1, 1] == 1
    assert out_1[2, 2] == 1
    assert out_1[3, 3] == 1
    assert out_1[4].sum() == 0


def test_str_to_ohe_null():
    """Test str_to_ohe function."""
    out_1 = str_to_ohe("NNNN")

    assert out_1.shape == (4, 4)
    assert out_1.sum() == 0


def test_ohe_to_str():
    """Test ohe_to_str function."""
    ohe = np.array([
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]],
    ])
    result = ohe_to_str(ohe)

    assert result == ["ACGT", "TGCA"]


def test_ohe_to_str_with_n():
    """Test ohe_to_str function with N (all zeros)."""
    ohe = np.array([
        [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
    ])
    result = ohe_to_str(ohe)

    assert result == ["ANG"]


def test_ohe_to_str_trailing_n():
    """Test ohe_to_str strips trailing Ns."""
    ohe = np.array([
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    ])
    result = ohe_to_str(ohe)

    assert result == ["AC"]


@pytest.fixture
def genome():
    """Load gencode.v41 genome."""
    gk = pytest.importorskip("genome_kit")
    return gk.Genome("gencode.v41")


@pytest.fixture
def coding_transcript(genome):
    """Get a coding transcript with CDS."""
    for t in genome.transcripts:
        if len(t.cdss) > 0:
            return t
    pytest.fail("No coding transcript found in genome")


@pytest.fixture
def noncoding_transcript(genome):
    """Get a non-coding transcript without CDS."""
    for t in genome.transcripts:
        if len(t.cdss) == 0:
            return t
    pytest.fail("No non-coding transcript found in genome")


@pytest.fixture
def gene_with_mane(genome):
    """Get a gene that has a MANE transcript and multiple transcripts."""
    for gene in genome.genes:
        mane = genome.mane_select_transcripts(gene)
        if len(mane) > 0 and len(list(gene.transcripts)) > 1:
            return gene
    pytest.fail("No gene with MANE transcript found in genome")


@pytest.fixture
def gene_with_multiple_appris(genome):
    """Get a gene with multiple APPRIS transcripts but no MANE."""
    for gene in genome.genes:
        mane = genome.mane_select_transcripts(gene)
        if len(mane) == 0:
            transcripts = list(gene.transcripts)
            appris_transcripts = [
                t for t in transcripts
                if genome.appris_principality(t) is not None
            ]
            if len(appris_transcripts) >= 2:
                return gene
    pytest.fail("No gene with multiple APPRIS transcripts found in genome")


def test_create_cds_track_coding(coding_transcript):
    """Test create_cds_track on a coding transcript."""
    result = create_cds_track(coding_transcript)

    cds_len = sum(len(x) for x in coding_transcript.cdss)
    utr5_len = sum(len(x) for x in coding_transcript.utr5s)
    utr3_len = sum(len(x) for x in coding_transcript.utr3s)

    assert len(result) == cds_len + utr5_len + utr3_len
    assert result[:utr5_len].sum() == 0
    if utr3_len > 0:
        assert result[-utr3_len:].sum() == 0
    assert result[utr5_len::3].sum() > 0


def test_create_cds_track_noncoding(noncoding_transcript):
    """Test create_cds_track on a non-coding transcript."""
    result = create_cds_track(noncoding_transcript)

    exon_len = sum(len(x) for x in noncoding_transcript.exons)
    assert len(result) == exon_len
    assert result.sum() == 0


def test_create_splice_track(coding_transcript):
    """Test create_splice_track marks exon boundaries."""
    result = create_splice_track(coding_transcript)
    num_exons = len(coding_transcript.exons)

    assert result.sum() == num_exons


def test_create_sequence(coding_transcript, genome):
    """Test create_sequence returns valid DNA sequence."""
    result = create_sequence(coding_transcript, genome)

    assert len(result) > 0
    assert set(result.upper()).issubset({"A", "C", "G", "T", "N"})

    expected_len = sum(len(exon) for exon in coding_transcript.exons)
    assert len(result) == expected_len


def test_get_top_n_priority_transcripts(gene_with_mane, genome):
    """Test get_top_n_priority_transcripts returns transcripts."""
    result = get_top_n_priority_transcripts(gene_with_mane, genome, n=3)

    assert isinstance(result, list)
    assert len(result) <= 3
    assert len(result) <= len(list(gene_with_mane.transcripts))


def test_get_top_n_priority_transcripts_mane_first(gene_with_mane, genome):
    """Test MANE transcripts come before APPRIS."""
    mane = genome.mane_select_transcripts(gene_with_mane)
    result = get_top_n_priority_transcripts(gene_with_mane, genome, n=5)

    assert result[0].id == mane[0].id


def test_get_top_n_priority_transcripts_appris_order(
    gene_with_multiple_appris,
    genome
):
    """Test APPRIS transcripts are sorted by principality."""
    result = get_top_n_priority_transcripts(
        gene_with_multiple_appris,
        genome,
        n=5
    )

    priorities = [
        genome.appris_principality(t)
        for t in result
        if genome.appris_principality(t) is not None
    ]

    assert len(priorities) >= 2
    assert priorities == sorted(priorities)
