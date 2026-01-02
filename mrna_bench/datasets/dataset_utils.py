from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from genome_kit import Gene, Genome, Transcript

import numpy as np


def ohe_to_str(
    ohe: np.ndarray,
    nucs: list[str] = ["A", "C", "G", "T", "N"]
) -> list[str]:
    """Convert OHE sequence to string representation.

    Args:
        ohe: One hot encoded sequence to convert.
        nucs: List of nucleotides corresponding to OHE position.

    Returns:
        List of string tokens representing nucleotides.
    """
    indices = np.where(ohe.sum(axis=-1) == 0, 4, np.argmax(ohe, axis=-1))
    sequences = ["".join(nucs[i] for i in row) for row in indices]
    sequences = [seq.rstrip("N") for seq in sequences]
    return sequences


def str_to_ohe(
    sequence: str,
    nucs: list[str] = ["A", "C", "G", "T"]
) -> np.ndarray:
    """Convert sequence to OHE. Represents "N" as all zeros.

    Args:
        sequence: Sequence to convert.
        nucs: Nucleotides corresponding to their one hot position.

    Returns:
        One hot encoded sequence.
    """
    mapping = {nuc: i for i, nuc in enumerate(nucs)}
    num_classes = len(mapping)

    mapping["N"] = -1

    # Convert sequence to indices
    indices = np.array([mapping[base] for base in sequence])

    # Create one-hot encoding
    one_hot = np.zeros((len(sequence), num_classes), dtype=int)

    for i in range(len(sequence)):
        if indices[i] == -1:
            continue
        one_hot[i, indices[i]] = 1

    return one_hot


def create_cds_track(transcript: "Transcript") -> np.ndarray:
    """Generate CDS track for a transcript.

    Args:
        transcript: Transcript object.

    Returns:
        CDS track for the transcript.
    """
    if len(transcript.cdss) == 0:
        return np.zeros(sum([len(x) for x in transcript.exons]), dtype=np.int8)

    cds_intervals = transcript.cdss
    utr3_intervals = transcript.utr3s
    utr5_intervals = transcript.utr5s

    len_utr3 = sum([len(x) for x in utr3_intervals])
    len_utr5 = sum([len(x) for x in utr5_intervals])
    len_cds = sum([len(x) for x in cds_intervals])

    # create a track where first position of the codon is one
    cds_track = np.zeros(len_cds, dtype=np.int8)
    # set every third position to 1
    cds_track[0::3] = 1
    # concat with zeros of utr3 and utr5
    full_track = np.concatenate([
        np.zeros(len_utr5, dtype=np.int8),
        cds_track,
        np.zeros(len_utr3, dtype=np.int8)
    ])
    return full_track


def create_splice_track(transcript: "Transcript") -> np.ndarray:
    """Generate splicing track for a transcript.

    Args:
        transcript: Transcript object.

    Returns:
        Splicing track for the transcript.
    """
    len_utr3 = sum([len(x) for x in transcript.utr3s])
    len_utr5 = sum([len(x) for x in transcript.utr5s])
    len_cds = sum([len(x) for x in transcript.cdss])

    if len(transcript.cdss) == 0:
        len_mrna = sum([len(x) for x in transcript.exons])
    else:
        len_mrna = len_utr3 + len_utr5 + len_cds
    splicing_track = np.zeros(len_mrna, dtype=np.int8)
    cumulative_len = 0
    for exon in transcript.exons:
        cumulative_len += len(exon)
        splicing_track[cumulative_len - 1:cumulative_len] = 1

    return splicing_track


def create_sequence(transcript: "Transcript", genome: "Genome") -> str:
    """Generate sequence for a transcript.

    Args:
        transcript: Transcript object.
        genome: Genome object.

    Returns:
        Sequence for the transcript.
    """
    seq = "".join([genome.dna(exon) for exon in transcript.exons])
    return seq


def get_top_n_priority_transcripts(
    gene: "Gene",
    genome: "Genome",
    n: int = 3
) -> list["Transcript"]:
    """Get up to N priority transcripts for a gene.

    The selection process sorts all unique transcripts for a gene and returns
    the top N. The sorting hierarchy is:
    1. MANE select transcripts.
    2. APPRIS transcripts (sorted by principality: 1, 2, 3, etc.).
    3. Other transcripts.

    Within each category, transcripts are sorted by their ID.

    Args:
        gene: Gene object.
        genome: Genome object.
        n: Number of top transcripts to return.

    Returns:
        List of up to N top-priority transcripts.
    """
    unique_transcripts_map = {t.id: t for t in gene.transcripts}

    if not unique_transcripts_map:
        return []

    mane_ids = {t.id for t in genome.mane_select_transcripts(gene)}

    def get_sort_key(transcript: "Transcript") -> tuple:
        if transcript.id in mane_ids:
            return (0, 0, transcript.id)

        appris_priority = genome.appris_principality(transcript)
        if appris_priority is not None:
            return (1, appris_priority, transcript.id)

        return (2, 0, transcript.id)

    all_unique_transcripts = list(unique_transcripts_map.values())
    all_unique_transcripts.sort(key=get_sort_key)

    return all_unique_transcripts[:n]
