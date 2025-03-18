from pathlib import Path
import shutil
import os 
from tqdm import tqdm
from typing import List, Tuple

import pandas as pd
import numpy as np
import parasail

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from mrna_bench import get_data_path
from mrna_bench.data_splitter.data_splitter import DataSplitter

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

def compute_kmer_similarity_matrix(
    sequences: List[str],
    transcripts: List[str], 
    k_small : int = 3,
    k_large : int = 3,
    dtype: np.dtype = np.float16
):
    """Create kmer count encoding for every sequence in the list and then compute 
    pairwise similarity between sequences using cosine similarity.

    Args:
        sequences: List of sequences.
        transcripts: List of transcript IDs.
        k_small: Smallest k-mer size.
        k_large: Largest k-mer size.
        dtype: Data type for the similarity matrix.
    
    Returns:
        A pandas DataFrame of shape (n, n) where n is the number of sequences. The index
        and columns are the transcript IDs and the values are the similarity scores.
    """
    # 1) Vectorize without converting to array (stays in CSR)
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(k_small, k_large))
    kmer_counts = vectorizer.fit_transform(sequences)  # shape = (N, num_kmers)

    # 2) Normalize each row to have unit L2 norm
    kmer_counts = normalize(kmer_counts, norm='l2', axis=1)

    # 3) Cosine similarity matrix = dot product of normalized vectors
    #    This yields an (N x N) sparse matrix (CSR).
    sim_matrix = kmer_counts.dot(kmer_counts.T)

    # Optionally cast to smaller dtype, but watch out for precision
    sim_matrix = sim_matrix.astype(dtype)

    # convert to dataframe
    sim_matrix = pd.DataFrame(sim_matrix.toarray(), index=transcripts, columns=transcripts)

    return sim_matrix


def calculate_similarity(seq1, seq2, gap_open, gap_extend, matrix):
    """
    Calculate the percent identity between two sequences using global alignment.
    """
    # Perform global alignment with traceback enabled using 16-bit, then 32-/64-bit if saturated.
    result = parasail.nw_trace_scan_16(seq1, seq2, gap_open, gap_extend, matrix)
    if result.saturated:
        result = parasail.nw_trace_scan_32(seq1, seq2, gap_open, gap_extend, matrix)
        if result.saturated:
            print("Saturated result, using 64-bit")
            result = parasail.nw_trace_scan_64(seq1, seq2, gap_open, gap_extend, matrix)

    # Extract the aligned sequences from the traceback and compute percent identity.
    aligned_q = result.traceback.query
    aligned_t = result.traceback.ref
    matches = sum(1 for a, b in zip(aligned_q, aligned_t) if a == b)
    percent_identity = (matches / len(aligned_q))
    return percent_identity

def compute_similarity_matrix(
    sequences : List[str],
    transcripts : List[str],
    gap_open : int = 7,
    gap_extend : int = 2,
    matrix : parasail.Matrix = parasail.dnafull, 
    dtype : np.dtype = np.float16
):
    """
    Compute a symmetric similarity matrix for a list of sequences.
    The matrix is stored as a pandas DataFrame with the transcript IDs as the index and columns.

    This code assumes that the order of the sequences in the similarity matrix is the same as 
    the order of the transcripts in the list.
    
    Parameters:
      sequences: list of sequences (strings)
      transcripts: list of transcript IDs (strings)
      gap_open: gap opening penalty
      gap_extend: gap extension penalty
      matrix: substitution matrix to use (e.g., parasail.dnafull)
      dtype: the data type for the stored similarity values (e.g., np.float16)
      
    Returns:
      A pandas.DataFrame of shape (n, n) where n is the number of sequences. The index
        and columns are the transcript IDs and the values are the similarity scores. 
    """
    n = len(sequences)
    similarity_matrix = np.zeros((n, n), dtype=dtype)
    
    if len(sequences) != len(transcripts):
        raise ValueError("Sequences and transcripts must have the same length.")

    # Process each pair only once. We compute the diagonal and lower triangle and mirror into the upper triangle.
    for i in tqdm(range(n), desc="Computing similarity matrix"):
        
        diag_val = 1.0
        similarity_matrix[i, i] = diag_val

        for j in range(i + 1, n):
            sim = calculate_similarity(sequences[i], sequences[j],
                                       gap_open=gap_open,
                                       gap_extend=gap_extend,
                                       matrix=matrix)

            # Store the similarity in both the lower and upper triangle.
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim

    similarity_matrix = pd.DataFrame(similarity_matrix, index=transcripts, columns=transcripts)

    return similarity_matrix

def train_test_split_ss(
    transcripts: List[str],
    ss_matrix: pd.DataFrame,
    threshold: float = 0.75,
    test_size: float = 0.3,
    random_state: int | None = None,
    overshoot_to_train: bool = True,
) -> Tuple[List[int], List[int]]:
    """
    Split transcripts into two sets such that any sequences that are similar (i.e. with similarity
    >= threshold) end up in the same set.

    The function first grabs the corresponding indices for the transcripts in the similarity matrix,
    converts to csr, thresholds the submatrix, and computes connected components so that sequences
    in the same connected component (i.e. group) are kept together. Then, groups are assigned to test
    or train based on the target test size. If overshoot occurs—that is, if adding an entire group would
    push the test set count above the target—then the behavior depends on the flag:
    
      - If overshoot_to_train is True (default in this version), then the entire group is assigned to
        train (i.e. overshoot happens in train).
      - If overshoot_to_train is False, then the group is assigned to test even if that means overshooting
        the target.

    Args:
        transcripts: List of transcript IDs.
        ss_matrix: A square similarity matrix in DataFrame format.
        threshold: Similarity threshold above which two sequences are considered similar.
        test_size: Fraction of data allocated to the test split.
        random_state: Optional integer seed for randomization.
        overshoot_to_train: If True, any group that would cause the test set to exceed the target
                            size is instead assigned to train. If False, groups are added to test
                            even if they overshoot the target.

    Returns:
        A tuple (train_indices, test_indices) where each is a list of row indices (relative to df_seqs)
        assigned to the training or test split.
    """

    # 1. Grab the corresponding indices for the transcripts in the similarity matrix and convert to CSR.
    ss_subset = csr_matrix(ss_matrix.loc[transcripts, transcripts].values)

    # In case the matrix only contains the upper triangle, symmetrize it.
    ss_subset = ss_subset.maximum(ss_subset.T)

    # 2. Binarize the matrix at the given threshold.
    binary_ss = (ss_subset >= threshold).astype(int)

    # 3. Compute connected components so that connected sequences are in the same group.
    _, labels = connected_components(csgraph=binary_ss, directed=False)

    # Group dataframe indices by their connected component label.
    group_to_indices = {}
    for df_idx, comp_label in enumerate(labels):
        group_to_indices.setdefault(comp_label, []).append(df_idx)
    groups = list(group_to_indices.values())

    # if we have less than 2 groups, we can't split the data
    if len(groups) < 2:
        raise ValueError("Cannot split data into two groups with the given similarity threshold.")

    print(f"Splitting data into {len(groups)} groups.") #if you notice that the number of groups is very small, you may want to increase the threshold

    # 4. Determine the target number of test samples.
    total = len(transcripts)
    target_test_count = int(round(test_size * total))

    # 5. Shuffle the groups (using the provided random_state).
    np.random.seed(random_state)
    np.random.shuffle(groups)

    # 6. Assign whole groups to test until reaching the target.
    #    If overshoot_to_train is True, a group that would exceed the target is assigned to train.
    test_groups = []
    train_groups = []
    current_test_count = 0
    for group in groups:
        if current_test_count < target_test_count:
            # Check whether adding this group would overshoot the target.
            if overshoot_to_train and (current_test_count + len(group) > target_test_count):
                train_groups.append(group)
            else:
                test_groups.append(group)
                current_test_count += len(group)
        else:
            train_groups.append(group)

    # Flatten the group lists into flat lists of indices.
    test_indices = [idx for group in test_groups for idx in group]
    train_indices = [idx for group in train_groups for idx in group]

    return train_indices, test_indices


class SSSplitter(DataSplitter):

    SS_LNCRNA_ESS_URL = "/home/dalalt1/Orthrus_eval/essentiality/lncRNA_homology/output/similarity_matrix.csv"

    def __init__(
        self,
        sequences: List[str] | None,
        transcripts: List[str] | None,
        ss_map_path: str | None,
        threshold: float,
        dataset_name: str | None = None,
        force_redownload: bool = False,
    ):
        """Initialize SSSplitter.

        Sequence similarity splitting requires a similarity matrix. This matrix
        should be in a pandas.DataFrame format (symmetric). If the similarity
        matrix is not specified, parasail will be used to calculate the global
        similarity between every pair of sequences in the dataset. However, this 
        is not recommended for datasets with more than 1000 sequences as it is very 
        computationally expensive. Ideally, the similarity matrix should be precomputed 
        and saved in the pandas.DataFrame format.

        Args:
            sequences: List of sequences.
            transcripts: List of transcript IDs.
            ss_map_path: Path to similarity matrix.
            threshold: Similarity threshold for categorizing sequences as similar.
            force_redownload: Forces redownload of similarity matrix.
        """
        super().__init__()

        self.data_storage_path = get_data_path()
        self.sequences = sequences
        self.transcripts = transcripts
        self.threshold = threshold

        if sequences is not None and transcripts is None:
            raise ValueError("Transcript IDs must be provided if sequences are provided.")
        elif sequences is None and transcripts is not None:
            raise ValueError("Sequences must be provided if transcript IDs are provided.")

        if ss_map_path is None and (sequences is not None and transcripts is not None):
            print("No similarity matrix provided. Calculating similarity matrix.")
            
            if dataset_name is None:
                raise ValueError("Dataset name must be provided if no similarity matrix is provided.")
            
            self.ss_map_path = self.data_storage_path + "/ss_maps/" + dataset_name + ".csv"

            # manually compute similarity matrix here
            if len(sequences) > 1000:
                print("Warning: Calculating similarity matrix for large dataset using k-mer frequency + cosine similarity.")
                self.ss_matrix = compute_kmer_similarity_matrix(
                    sequences = sequences, 
                    transcripts = transcripts, 
                    k_small=3, 
                    k_large=3, 
                    dtype=np.float16
                )
            else:
                self.ss_matrix = compute_similarity_matrix(
                    sequences = sequences, 
                    transcripts = transcripts,
                    gap_open=7, 
                    gap_extend=2, 
                    matrix=parasail.dnafull, 
                    dtype=np.float16
                )

            print(f"Saving similarity matrix to {self.ss_map_path}.")
            os.makedirs(self.data_storage_path + "/ss_maps", exist_ok=True)

            self.ss_matrix.to_csv(self.ss_map_path, index_label='index')

        elif Path(ss_map_path) != self.data_storage_path + "/ss_maps/" + Path(ss_map_path).name: # if the path is not in the data storage path, copy it there

            self.raw_data_src_path = ss_map_path

            self.ss_map_path = self.data_storage_path + "/ss_maps/" + Path(ss_map_path).name

            if not Path(self.ss_map_path).exists():
                os.makedirs(self.data_storage_path + "/ss_maps", exist_ok=True)
                shutil.copy(self.raw_data_src_path, self.ss_map_path)
            else:
                print(f"Similarity matrix already exists at {self.ss_map_path}.")

            self.ss_matrix = pd.read_csv(Path(self.ss_map_path), index_col='index')
        else:
            self.ss_map_path = ss_map_path
            self.ss_matrix = pd.read_csv(Path(self.ss_map_path), index_col='index')


    def split_df(
        self,
        df: pd.DataFrame,
        test_size: float,
        random_seed: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        df_seqs = df["sequence"].tolist()

        train_indices, test_indices = train_test_split_ss(
            df["transcript"].tolist(),
            self.ss_matrix,
            self.threshold,
            test_size,
            random_seed
        )

        assert len(set(train_indices).intersection(set(test_indices))) == 0

        train_df = df.iloc[train_indices].reset_index(drop=True)
        test_df = df.iloc[test_indices].reset_index(drop=True)

        return train_df, test_df
