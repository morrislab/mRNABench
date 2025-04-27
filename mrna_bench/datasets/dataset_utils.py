from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from genome_kit import Genome, Transcript

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
        return np.zeros(sum([len(x) for x in transcript.exons]), dtype=int)

    cds_intervals = transcript.cdss
    utr3_intervals = transcript.utr3s
    utr5_intervals = transcript.utr5s

    len_utr3 = sum([len(x) for x in utr3_intervals])
    len_utr5 = sum([len(x) for x in utr5_intervals])
    len_cds = sum([len(x) for x in cds_intervals])

    # create a track where first position of the codon is one
    cds_track = np.zeros(len_cds, dtype=int)
    # set every third position to 1
    cds_track[0::3] = 1
    # concat with zeros of utr3 and utr5
    cds_track = np.concatenate([
        np.zeros(len_utr5, dtype=int),
        cds_track,
        np.zeros(len_utr3, dtype=int)
    ])
    return cds_track


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
    splicing_track = np.zeros(len_mrna, dtype=int)
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
