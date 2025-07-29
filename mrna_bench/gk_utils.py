import numpy as np
from typing import List, Tuple, Optional
import genome_kit as gk
import re


def find_transcript(genome, transcript_id):
    """Find a transcript in a genome by transcript ID.

    Args:
        genome (object): The genome object containing a list of transcripts.
        transcript_id (str): The ID of the transcript to find.

    Returns:
        object: The transcript object, if found.

    Raises:
        ValueError: If no transcript with the given ID is found.

    Example:
        >>> # Create sample transcripts and a genome
        >>> transcript1 = 'ENST00000263946'
        >>> genome = Genome("gencode.v29")
        >>> result = find_transcript(genome, 'ENST00000335137')
        >>> print(result.id)
        <Transcript ENST00000263946.7 of PKP1>
        >>> # If transcript ID is not found
        >>> find_transcript(genome, 'ENST00000000000')
        ValueError: Transcript with ID ENST00000000000 not found.
    """
    transcripts = [x for x in genome.transcripts if x.id.split(".")[0] == transcript_id]
    if not transcripts:
        raise ValueError(f"Transcript with ID {transcript_id} not found.")

    return transcripts[0]


def find_transcript_by_gene_name(genome, gene_name):
    """Find all transcripts in a genome by gene name.

    Args:
        genome (object): The genome object containing a list of transcripts.
        gene_name (str): The name of the gene whose transcripts are to be found.

    Returns:
        list: A list of transcript objects corresponding to the given gene name.

    Raises:
        ValueError: If no transcripts for the given gene name are found.

    Example:
        >>> # Find transcripts by gene name
        >>> transcripts = find_transcript_by_gene_name(genome, 'PKP1')
        >>> print(transcripts)
        [<Transcript ENST00000367324.7 of PKP1>,
        <Transcript ENST00000263946.7 of PKP1>,
        <Transcript ENST00000352845.3 of PKP1>,
        <Transcript ENST00000475988.1 of PKP1>,
        <Transcript ENST00000477817.1 of PKP1>]
        >>> # If gene name is not found
        >>> find_transcript_by_gene_name(genome, 'XYZ')
        ValueError: No transcripts found for gene name XYZ.
    """
    genes = [x for x in genome.genes if x.name == gene_name]
    if not genes:
        raise ValueError(f"No genes found for gene name {gene_name}.")
    if len(genes) > 1:
        print(f"Warning: More than one gene found for gene name {gene_name}.")
        print("Concatenating transcripts from all genes.")

    transcripts = []
    for gene in genes:
        transcripts += gene.transcripts
    return transcripts


def get_transcript_regions(
    t: Optional[gk.Transcript] = None,
    list_of_exons: Optional[List[gk.Interval]] = None,
    list_of_cds_intervals: Optional[List[gk.Interval]] = None,
) -> Tuple[List[gk.Interval], List[gk.Interval], List[gk.Interval]]:
    """
    Identifies and returns the genomic intervals corresponding to the
    5' UTR, CDS, and 3' UTR regions of a transcript.

    Args:
        t (gk.Transcript, optional): The transcript object.
            If provided, its exons and cdss attributes are used unless
            overridden by list_of_exons/list_of_cds_intervals.
        list_of_exons (list, optional): List of exon intervals.
            Required if t is None. Overrides t.exons if t is provided.
        list_of_cds_intervals (list, optional): List of CDS intervals.
            Required if t is None. Overrides t.cdss if t is provided.

    Returns:
        Tuple[List[gk.Interval], List[gk.Interval], List[gk.Interval]]:
            A tuple containing three lists of intervals:
            (five_prime_utr_intervals, cds_intervals_sorted, three_prime_utr_intervals)
    """
    # --- 1. Input Validation and Initialization ---
    if t is None and (list_of_exons is None or list_of_cds_intervals is None):
        raise ValueError(
            "If t is None, both list_of_exons and list_of_cds_intervals must be provided."
        )

    exon_list = list_of_exons if list_of_exons is not None else (t.exons if t else [])
    cds_list = (
        list_of_cds_intervals
        if list_of_cds_intervals is not None
        else (t.cdss if t else [])
    )

    if not exon_list:
        return [], [], []  # No exons, no regions

    # Determine strand and reference genome from the first exon
    strand = exon_list[0].strand
    chrom = exon_list[0].chrom
    ref_genome = exon_list[0].reference_genome

    # Assert consistency
    assert all(exon.strand == strand for exon in exon_list), (
        "All exons must be on the same strand"
    )
    assert all(exon.chrom == chrom for exon in exon_list), (
        "All exons must be on the same chromosome"
    )
    if cds_list:
        assert all(cds.strand == strand for cds in cds_list), (
            "All CDS intervals must be on the same strand as exons"
        )
        assert all(cds.chrom == chrom for cds in cds_list), (
            "All CDS intervals must be on the same chromosome"
        )

    # --- 2. Sort Exons and CDS Intervals (5' to 3') ---
    if strand == "+":
        sorted_exons = sorted(exon_list, key=lambda x: x.start)
        cds_intervals_sorted = sorted(cds_list, key=lambda x: x.start)
    else:  # negative strand
        sorted_exons = sorted(exon_list, key=lambda x: x.end, reverse=True)
        cds_intervals_sorted = sorted(cds_list, key=lambda x: x.end, reverse=True)

    # --- 3. Handle No CDS Case ---
    if not cds_intervals_sorted:
        # If no CDS, all exons are considered UTR. Let's call them 5'UTR.
        return sorted_exons, [], []

    # --- 4. Identify CDS Region Boundaries ---
    first_cds = cds_intervals_sorted[0]
    last_cds = cds_intervals_sorted[-1]

    # Define the genomic coordinates corresponding to the 5' and 3' ends of the *entire* CDS region
    if strand == "+":
        cds_five_prime_genomic_coord = first_cds.start
        cds_three_prime_genomic_coord = last_cds.end
    else:  # negative strand
        cds_five_prime_genomic_coord = first_cds.end  # 5' end is the largest coordinate
        cds_three_prime_genomic_coord = (
            last_cds.start
        )  # 3' end is the smallest coordinate

    # --- 5. Identify 5' UTR Intervals ---
    five_prime_utr_intervals = []
    for exon in sorted_exons:
        exon_start = exon.start
        exon_end = exon.end

        if strand == "+":
            if exon_end <= cds_five_prime_genomic_coord:
                # Exon is entirely before the CDS start
                five_prime_utr_intervals.append(exon)
            elif exon_start < cds_five_prime_genomic_coord:
                # Exon overlaps the CDS start -> partial 5' UTR
                utr_part = gk.Interval(
                    start=exon_start,
                    end=cds_five_prime_genomic_coord,  # End at the CDS start coord
                    chromosome=chrom,
                    reference_genome=ref_genome,
                    strand=strand,
                )
                if len(utr_part) > 0:
                    five_prime_utr_intervals.append(utr_part)
                break  # Stop after finding the first overlap for 5' UTR
            else:
                # Exon starts at or after the CDS start
                break  # No more 5' UTR possible
        else:  # negative strand
            if exon_start >= cds_five_prime_genomic_coord:
                # Exon is entirely 5' of the CDS start (genomically higher coord)
                five_prime_utr_intervals.append(exon)
            elif exon_end > cds_five_prime_genomic_coord:
                # Exon overlaps the CDS start (genomically higher coord) -> partial 5' UTR
                utr_part = gk.Interval(
                    start=cds_five_prime_genomic_coord,  # Start at the CDS start coord
                    end=exon_end,
                    chromosome=chrom,
                    reference_genome=ref_genome,
                    strand=strand,
                )
                if len(utr_part) > 0:
                    five_prime_utr_intervals.append(utr_part)
                break  # Stop after finding the first overlap for 5' UTR
            else:
                # Exon ends at or before the CDS start coord (is 3' of CDS start)
                break  # No more 5' UTR possible

    # --- 6. Identify 3' UTR Intervals ---
    three_prime_utr_intervals = []
    for exon in sorted_exons:  # Still iterating 5'->3'
        exon_start = exon.start
        exon_end = exon.end

        if strand == "+":
            if exon_start >= cds_three_prime_genomic_coord:
                # Exon is entirely after the CDS end
                three_prime_utr_intervals.append(exon)
            elif exon_end > cds_three_prime_genomic_coord:
                # Exon overlaps the CDS end -> partial 3' UTR
                utr_part = gk.Interval(
                    start=cds_three_prime_genomic_coord,  # Start at CDS end coord
                    end=exon_end,
                    chromosome=chrom,
                    reference_genome=ref_genome,
                    strand=strand,
                )
                if len(utr_part) > 0:
                    three_prime_utr_intervals.append(utr_part)
                # Continue checking subsequent exons as they might be fully 3' UTR
            # else: Exon is entirely before or within CDS -> ignore for 3' UTR

        else:  # negative strand
            if exon_end <= cds_three_prime_genomic_coord:
                # Exon is entirely 3' of the CDS end (genomically lower coord)
                three_prime_utr_intervals.append(exon)
            elif exon_start < cds_three_prime_genomic_coord:
                # Exon overlaps the CDS end (genomically lower coord) -> partial 3' UTR
                utr_part = gk.Interval(
                    start=exon_start,
                    end=cds_three_prime_genomic_coord,  # End at the CDS end coord
                    chromosome=chrom,
                    reference_genome=ref_genome,
                    strand=strand,
                )
                if len(utr_part) > 0:
                    three_prime_utr_intervals.append(utr_part)
                # Continue checking subsequent exons as they might be fully 3' UTR
            # else: Exon is entirely 5' or within CDS -> ignore for 3' UTR

    # Sort the found 3' UTR intervals correctly AFTER the loop
    # This handles cases where partial overlaps were found before full overlaps
    if strand == "+":
        three_prime_utr_intervals.sort(key=lambda x: x.start)
    else:
        three_prime_utr_intervals.sort(key=lambda x: x.end, reverse=True)

    # --- 7. Return Results ---
    # The CDS list was already sorted 5'->3'
    return five_prime_utr_intervals, cds_intervals_sorted, three_prime_utr_intervals


def create_cds_track(t, list_of_exons=None, list_of_cds_intervals=None):
    """
    Create a track for the coding sequence of a transcript, handling both strands correctly.

    - The final track length = sum of all exon lengths.
    - The region before the CDS is zeros (the '5′ UTR').
    - The CDS region is an every-third=1 pattern.
    - The region after is zeros (the '3′ UTR').

    Args:
        t (gk.Transcript): The transcript object. Must have `t.cdss` for coding intervals.
        list_of_exons (list, optional): List of exon intervals to use instead of t.exons.
        list_of_cds_intervals (list, optional): List of CDS intervals to use instead of t.cdss.

    Returns:
        np.ndarray: 1D array of shape (transcript_length,).
                    0 for noncoding positions, 1 every third base in the CDS region.
    """
    # 1) Compute total length of the transcript (sum of exon lengths)
    if t is None:
        assert list_of_exons is not None
        transcript_length = sum(len(exon) for exon in list_of_exons)
        exon_list = list_of_exons
        # If t is None, we need to infer strand from the exons
        strand = exon_list[0].strand
    else:
        transcript_length = sum(len(exon) for exon in t.exons)
        exon_list = t.exons
        strand = t.strand

    # Assert all intervals are on the same strand
    assert all(exon.strand == strand for exon in exon_list), (
        "All exons must be on the same strand"
    )

    if transcript_length == 0:
        return np.array([], dtype=int)

    # 2) If there are no CDS intervals, return an all-zero track
    if t is None:
        assert list_of_cds_intervals is not None
        cds_intervals = list_of_cds_intervals
    else:
        cds_intervals = t.cdss
    if not cds_intervals:
        return np.zeros(transcript_length, dtype=int)

    # Assert all CDS intervals are on the same strand as exons
    assert all(cds.strand == strand for cds in cds_intervals), (
        "All CDS intervals must be on the same strand as exons"
    )

    # Sort CDS intervals by 5' to 3' direction
    if strand == "+":
        sorted_cds_intervals = sorted(cds_intervals, key=lambda x: x.start)
        first_cds = sorted_cds_intervals[0]  # Most 5' CDS interval
        assert first_cds.end5.start == first_cds.start, (
            "On positive strand, end5 should equal start"
        )
    else:  # negative strand
        sorted_cds_intervals = sorted(cds_intervals, key=lambda x: x.end, reverse=True)
        first_cds = sorted_cds_intervals[0]  # Most 5' CDS interval
        assert first_cds.end5.start == first_cds.end, (
            "On negative strand, end5 should equal end"
        )

    # 3) Sum the lengths of all CDS intervals
    cds_length = sum(len(c) for c in sorted_cds_intervals)

    # Sort exons in 5' to 3' direction
    if strand == "+":
        sorted_exons = sorted(exon_list, key=lambda x: x.start)
    else:
        sorted_exons = sorted(exon_list, key=lambda x: x.end, reverse=True)

    # Find the 5' UTR length by calculating the total length of exons or parts of exons
    # that come before the first CDS region in 5' to 3' direction
    five_utr_length = 0
    for exon in sorted_exons:
        if strand == "+":
            if exon.end <= first_cds.start:
                # This exon is entirely in the 5' UTR
                five_utr_length += len(exon)
            elif exon.overlaps(first_cds):
                # This exon contains the start of the first CDS
                five_utr_length += max(0, first_cds.start - exon.start)
                break
            else:
                # This exon comes after the first CDS, stop counting
                break
        else:  # negative strand
            if exon.start >= first_cds.end:
                # This exon is entirely in the 5' UTR
                five_utr_length += len(exon)
            elif exon.overlaps(first_cds):
                # This exon contains the start of the first CDS
                five_utr_length += max(0, exon.end - first_cds.end)
                break
            else:
                # This exon comes after the first CDS, stop counting
                break

    # 6) The remainder after we place the CDS is the "3′ UTR" length
    three_utr_length = transcript_length - (five_utr_length + cds_length)
    assert three_utr_length >= 0, "3' UTR length cannot be negative"

    # 7) Build the CDS region track: every 3rd base is 1
    cds_track = np.zeros(cds_length, dtype=int)
    cds_track[0::3] = 1

    # 8) Concatenate: 5′ zeros, the CDS track, 3′ zeros
    track = np.concatenate(
        [
            np.zeros(five_utr_length, dtype=int),
            cds_track,
            np.zeros(three_utr_length, dtype=int),
        ]
    )

    return track


def create_splice_track(t, list_of_exons=None):
    """Create a track of the splice sites of a transcript.
    The track is a 1D array where the positions of the splice sites are 1.

    Args:
        t (gk.Transcript): The transcript object.
    """
    if list_of_exons is None:
        len_mrna = sum([len(x) for x in t.exons])
        list_of_exons = t.exons
    else:
        len_mrna = sum([len(x) for x in list_of_exons])

    splicing_track = np.zeros(len_mrna, dtype=int)
    cumulative_len = 0
    for exon in list_of_exons:
        cumulative_len += len(exon)
        splicing_track[cumulative_len - 1 : cumulative_len] = 1

    return splicing_track


# convert to one hot
def seq_to_oh(seq: str) -> np.ndarray:
    """
    Convert a sequence string to a one-hot encoded numpy array.
    This function is optimized for performance using NumPy vectorization.

    Args:
        seq (str): The input sequence (e.g., 'ACGT').

    Returns:
        np.ndarray: A 2D numpy array of shape (len(seq), 4) with dtype=int.

    Raises:
        ValueError: If the sequence contains characters other than 'A', 'C', 'G', 'T'.
    """
    seq = seq.upper()
    nuc_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    # Find unique characters in the sequence to efficiently check for invalid ones.
    unique_chars = set(seq)
    unfamiliar_chars = unique_chars - nuc_map.keys()
    if unfamiliar_chars:
        raise ValueError(f"Unfamiliar bases found in sequence: {', '.join(sorted(list(unfamiliar_chars)))}")

    # Create an array of indices from the sequence.
    # We can use direct mapping as we have already validated the characters.
    indices = np.fromiter((nuc_map[base] for base in seq), dtype=np.int8, count=len(seq))

    # Create the one-hot encoded matrix.
    one_hot = np.zeros((len(seq), 4), dtype=int)

    # Use numpy's advanced indexing to set the '1's in a single, vectorized operation.
    one_hot[np.arange(len(seq)), indices] = 1

    return one_hot


def ohe_to_str(ohe_seq: np.ndarray, channels_last: bool = False) -> str:
    """
    Convert a one-hot encoded sequence back to a string using vectorization.

    Args:
        ohe_seq (np.ndarray): 2D array of the OHE sequence.
        channels_last (bool): If True, shape is (L, 4). If False, shape is (4, L).
                              Defaults to False.

    Returns:
        str: The DNA sequence. 'N' for unknown bases (all-zero vectors).

    Raises:
        AssertionError: For invalid input shape or type.
    """
    assert isinstance(ohe_seq, np.ndarray), "Input must be a numpy array."
    assert ohe_seq.ndim == 2, f"Input must be 2D, but has shape {ohe_seq.shape}."

    if channels_last:
        assert ohe_seq.shape[1] == 4, f"Channels (dim 1) must be 4 for channels_last, but shape is {ohe_seq.shape}."
        ohe = ohe_seq
    else:
        assert ohe_seq.shape[0] == 4, f"Channels (dim 0) must be 4 for channels_first, but shape is {ohe_seq.shape}."
        ohe = ohe_seq.T

    bases = np.array(['A', 'C', 'G', 'T', 'N'])
    indices = np.argmax(ohe, axis=1)

    # Handle all-zero vectors, which argmax incorrectly maps to index 0.
    # We map them to index 4, which corresponds to 'N'.
    indices[ohe.sum(axis=1) == 0] = 4

    return "".join(bases[indices])


def create_one_hot_encoding(t, genome, list_of_exons=None):
    """Create a track of the sequence of a transcript.
    The track is a 2D array where the rows are the positions
    and the columns are the one-hot encoding of the bases.

    Args
        t (gk.Transcript): The transcript object.
    """
    if list_of_exons is None:
        seq = "".join([genome.dna(exon) for exon in t.exons])
    else:
        seq = "".join([genome.dna(exon) for exon in list_of_exons])
    oh = seq_to_oh(seq)
    return oh


def create_six_track_encoding(
    t,
    genome,
    list_of_exons=None,
    list_of_cds_intervals=None,
    channels_last=False,
):
    """Create a track of the sequence of a transcript.

    Produces an array of shape (L,6) if channels_last=True
    or (6,L) if channels_last=False.

    Args:
        t (gk.Transcript): The transcript object.
        genome (gk.Genome): Genome reference.
        channels_last (bool): If True, output is (L, 6). Otherwise, (6, L).

    Returns:
        np.ndarray: A 2D array with 6 channels (one-hot base encoding + CDS + splice).
    """
    if t is not None:
        # Step 1: Generate base tracks
        oh = create_one_hot_encoding(t, genome)  # shape is (L, 4)
        cds_track = create_cds_track(t)  # shape is (L,)
        splice_track = create_splice_track(t)  # shape is (L,)
    else:
        assert list_of_exons is not None
        assert list_of_cds_intervals is not None
        oh = create_one_hot_encoding(
            t=None, list_of_exons=list_of_exons, genome=genome
        )  # shape is (L, 4)
        cds_track = create_cds_track(
            t=None,
            list_of_exons=list_of_exons,
            list_of_cds_intervals=list_of_cds_intervals,
        )  # shape is (L,)
        splice_track = create_splice_track(
            t=None, list_of_exons=list_of_exons
        )  # shape is (L,)

    # Step 2: Create final track based on channels_last
    if channels_last:
        # Channels along axis=1 => shape (L, 6)
        # (L, 4), (L, 1), (L, 1) -> (L, 6)
        six_track = np.concatenate(
            [oh, cds_track[:, None], splice_track[:, None]], axis=1
        )
    else:
        # Channels along axis=0 => shape (6, L)
        # first transpose one-hot from (L, 4) to (4, L)
        oh = oh.T
        # reshape cds/splice from (L,) to (1, L)
        cds_track = cds_track[None, :]
        splice_track = splice_track[None, :]
        # now concatenate on axis=0 => shape (6, L)
        six_track = np.concatenate([oh, cds_track, splice_track], axis=0)

    return six_track


def get_transcript_sequence(
    t: Optional[gk.Transcript],
    genome: gk.Genome,
    list_of_exons: Optional[List[gk.Interval]] = None,
) -> str:
    """Get the sequence of a transcript.

    Args:
        t (gk.Transcript, optional): The transcript object.
        genome (gk.Genome): Genome reference.
        list_of_exons (list, optional): List of exon intervals.
            If provided, these are used instead of t.exons.

    Returns:
        str: The transcript sequence.
    """
    exons_to_use = list_of_exons if list_of_exons is not None else (t.exons if t else [])
    if not exons_to_use:
        if t is None and list_of_exons is None:
            raise ValueError("Either t or list_of_exons must be provided.")
        return ""  # Return empty string if there are no exons
    return "".join([genome.dna(exon) for exon in exons_to_use])


def get_transcript_cds_sequence(
    t: Optional[gk.Transcript],
    genome: gk.Genome,
    list_of_exons: Optional[List[gk.Interval]] = None,
    list_of_cds_intervals: Optional[List[gk.Interval]] = None,
) -> str:
    """Get the CDS sequence of a transcript.

    Args:
        t (gk.Transcript, optional): The transcript object.
        genome (gk.Genome): Genome reference.
        list_of_exons (list, optional): List of exon intervals.
            If provided, these are used instead of t.exons.
        list_of_cds_intervals (list, optional): List of CDS intervals.
            If provided, these are used instead of t.cdss.

    Returns:
        str: The CDS sequence.
    """
    five_prime_utr, cds, three_prime_utr = get_transcript_regions(
        t=t,
        list_of_exons=list_of_exons,
        list_of_cds_intervals=list_of_cds_intervals,
    )
    return "".join([genome.dna(exon) for exon in cds])


def _get_numerical_transcript_id(transcript):
    """Helper to extract the numerical part of a transcript ID for sorting."""
    try:
        # Find the first sequence of digits in the ID string. This is more
        # robust than assuming a specific prefix like "ENST".
        match = re.search(r"\d+", transcript.id)
        if match:
            return int(match.group(0))
        # If no digits are found, fall back to infinity.
        return float("inf")
    except (ValueError, IndexError, TypeError):
        # Return infinity for non-standard IDs to place them at the end
        return float("inf")


def get_priority_transcript(gene, genome):
    """
    Get the priority transcript for a gene based on the following hierarchy:
    1. MANE select transcript (if exactly one exists)
    2. APPRIS principal transcript (if exactly one exists)
    3. If multiple APPRIS, choose APPRIS transcript with lowest numerical ID
    4. If no MANE/APPRIS, choose transcript with lowest numerical ID from all transcripts
    
    Parameters
    ----------
    gene : genome_kit.Gene
        The gene object to get priority transcript for
    genome : genome_kit.Genome
        The genome object
        
    Returns
    -------
    transcript or None
        The priority transcript, or None if no transcripts exist
    """
    # Check for MANE select transcripts
    mane_transcripts = genome.mane_select_transcripts(gene)
    if len(mane_transcripts) == 1:
        return mane_transcripts[0]
    
    # Check for APPRIS principal transcripts
    appris_transcripts = genome.appris_transcripts(gene)
    if len(appris_transcripts) == 1:
        return appris_transcripts[0]
    elif len(appris_transcripts) > 1:
        # Multiple APPRIS transcripts - use numerical ID tiebreaker on APPRIS set
        candidates = appris_transcripts
    else:
        # No MANE/APPRIS - use all transcripts
        candidates = list(gene.transcripts)
    
    if not candidates:
        return None
    
    # Sort by numerical ID and return transcript with lowest ID
    candidates.sort(key=_get_numerical_transcript_id)
    return candidates[0]


def get_top_n_priority_transcripts(gene, genome, n=3):
    """
    Get up to N priority transcripts for a gene, based on a hierarchy.

    The selection process sorts all unique transcripts for a gene and returns
    the top N. The sorting hierarchy is:
    1. MANE select transcripts.
    2. APPRIS principal transcripts.
    3. Other transcripts.

    Within each category, transcripts are sorted by their numerical ID.

    Parameters
    ----------
    gene : genome_kit.Gene
        The gene object.
    genome : genome_kit.Genome
        The genome object.
    n : int, optional
        The number of top transcripts to return (default is 3).

    Returns
    -------
    list[genome_kit.Transcript]
        A list containing up to N of the top-priority transcripts.
    """
    # Use a dictionary to get unique transcripts from the gene by ID
    unique_transcripts_map = {t.id: t for t in gene.transcripts}

    if not unique_transcripts_map:
        return []

    # Get MANE and APPRIS sets for efficient lookup
    mane_ids = {t.id for t in genome.mane_select_transcripts(gene)}
    appris_ids = {t.id for t in genome.appris_transcripts(gene)}

    def get_sort_key(transcript):
        """Defines the sorting criteria for a transcript."""
        # Priority levels: 0 for MANE, 1 for APPRIS, 2 for other
        priority = 2
        if transcript.id in mane_ids:
            priority = 0
        elif transcript.id in appris_ids:
            priority = 1

        numerical_id = _get_numerical_transcript_id(transcript)
        return (priority, numerical_id)

    # Get a list of unique transcript objects
    all_unique_transcripts = list(unique_transcripts_map.values())

    # Sort all unique transcripts using the defined key
    all_unique_transcripts.sort(key=get_sort_key)

    return all_unique_transcripts[:n]
