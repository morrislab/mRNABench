from pathlib import Path
import shutil
import os 
from tqdm import tqdm
from typing import List, Tuple

import pandas as pd
import numpy as np
import parasail

from scipy.sparse import load_npz, csr_matrix, save_npz
from scipy.sparse.csgraph import connected_components

from mrna_bench import get_data_path
from mrna_bench.data_splitter.data_splitter import DataSplitter

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

def compute_kmer_similarity_sparse(
    sequences: List[str], 
    k_small : int = 3,
    k_large : int = 3,
    dtype: np.dtype = np.float16
):
    """Create kmer count encoding for every sequence in the list and then compute 
    pairwise similarity between sequences using cosine similarity.

    Args:
        sequences: List of sequences.
        k_small: Smallest k-mer size.
        k_large: Largest k-mer size.
        dtype: Data type for the similarity matrix.
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

def compute_similarity_sparse(sequences, gap_open=7, gap_extend=2, matrix=parasail.dnafull, dtype=np.float16):
    """
    Compute a symmetric similarity matrix for a list of sequences.
    The matrix is stored in a sparse CSR format.
    
    Parameters:
      sequences: list of sequences (strings)
      gap_open: gap opening penalty
      gap_extend: gap extension penalty
      matrix: substitution matrix to use (e.g., parasail.dnafull)
      dtype: the data type for the stored similarity values (e.g., np.float16)
      
    Returns:
      A scipy.sparse.csr_matrix of shape (n, n)
    """
    n = len(sequences)
    data = []
    rows = []
    cols = []
    
    # Process each pair only once. We compute the diagonal and lower triangle and mirror into the upper triangle.
    for i in tqdm(range(n), desc="Computing similarity matrix"):
        
        diag_val = 1.0
        data.append(diag_val)
        rows.append(i)
        cols.append(i)

        for j in range(i + 1, n):
            sim = calculate_similarity(sequences[i], sequences[j],
                                       gap_open=gap_open,
                                       gap_extend=gap_extend,
                                       matrix=matrix)

            # Store in the lower triangle.
            data.append(sim)
            rows.append(j)  # store at (j,i)
            cols.append(i)

            # # And mirror into the upper triangle.
            # data.append(sim)
            # rows.append(i)  # store at (i,j)
            # cols.append(j)

    sparse_mat = csr_matrix((np.array(data, dtype=dtype), (rows, cols)), shape=(n, n))
    return sparse_mat

def train_test_split_ss(
    sequences: List[str],
    df_seqs: List[str],
    ss_matrix: csr_matrix,
    threshold: float = 0.75,
    test_size: float = 0.3,
    random_state: int | None = None,
    overshoot_to_train: bool = True,
) -> Tuple[List[int], List[int]]:
    """
    Split sequences into two sets such that any sequences that are similar (i.e. with similarity
    >= threshold) end up in the same set.

    The function first maps the dataframe sequences to their corresponding rows/columns in the full
    similarity matrix, thresholds the submatrix, and computes connected components so that sequences
    in the same connected component (i.e. group) are kept together. Then, groups are assigned to test
    or train based on the target test size. If overshoot occurs—that is, if adding an entire group would
    push the test set count above the target—then the behavior depends on the flag:
    
      - If overshoot_to_train is True (default in this version), then the entire group is assigned to
        train (i.e. overshoot happens in train).
      - If overshoot_to_train is False, then the group is assigned to test even if that means overshooting
        the target.

    Args:
        sequences: List of all sequences (the order must match the similarity matrix rows/cols).
        df_seqs: List of sequences (strings) in the dataframe to be split.
        ss_matrix: A square similarity matrix in CSR format.
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
    # 1. Map each sequence in the full list to its index.
    seq_to_index = {seq: i for i, seq in enumerate(sequences)}

    # 2. Get the global indices corresponding to the dataframe sequences.
    global_indices = []
    for seq in df_seqs:
        try:
            global_indices.append(seq_to_index[seq])
        except KeyError:
            raise ValueError(f"Sequence {seq} from dataframe not found in the full sequence list.")

    # 3. Subset the similarity matrix to only include the rows/columns for df_seqs.
    ss_subset = ss_matrix[global_indices, :][:, global_indices]

    # 4. Binarize the matrix at the given threshold.
    binary_ss = (ss_subset >= threshold).astype(int)

    # In case the matrix only contains the upper triangle, symmetrize it.
    ss_subset = ss_subset.maximum(ss_subset.T)

    # 5. Compute connected components so that connected sequences are in the same group.
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

    # 6. Determine the target number of test samples.
    total = len(df_seqs)
    target_test_count = int(round(test_size * total))

    # 7. Shuffle the groups (using the provided random_state).
    np.random.seed(random_state)
    np.random.shuffle(groups)

    # 8. Assign whole groups to test until reaching the target.
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

    SS_LNCRNA_ESS_URL = "/home/dalalt1/Orthrus_eval/essentiality/lncRNA_homology/output/similarity_results_full.npz"

    def __init__(
        self,
        sequences: List[str] | None,
        ss_map_path: str | None,
        threshold: float,
        dataset_name: str | None = None,
        force_redownload: bool = False,
    ):
        """Initialize SSSplitter.

        Sequence similarity splitting requires a similarity matrix. This matrix
        should be a sparse symmetric matrix in CSR format. If the similarity
        matrix is not specified, parasail will be used to calculate the global
        similarity between every pair of sequences in the dataset. However, this 
        is not recommended for datasets with more than 1000 sequences as it is very 
        computationally expensive. Ideally, the similarity matrix should be precomputed 
        and saved in the scipy sparse CSR format. The similarity matrix only contains 
        the upper triangular part of the matrix to save space (since the matrix is 
        symmetric). This code assumes that the order of the sequences in the similarity
        matrix is the same as the order of the sequences in the dataset.

        Args:
            sequences: List of sequences.
            ss_map_path: Path to similarity matrix.
            threshold: Similarity threshold for categorizing sequences as similar.
            force_redownload: Forces redownload of similarity matrix.
        """
        super().__init__()

        self.data_storage_path = get_data_path()
        self.sequences = sequences
        self.threshold = threshold

        if ss_map_path is None and sequences is not None:
            print("No similarity matrix provided. Calculating similarity matrix.")
            
            if dataset_name is None:
                raise ValueError("Dataset name must be provided if no similarity matrix is provided.")
            
            self.ss_map_path = self.data_storage_path + "/ss_maps/" + dataset_name + ".npz"

            # manually compute similarity matrix here
            if len(sequences) > 1000:
                print("Warning: Calculating similarity matrix for large dataset using k-mer frequency + pearson correlation.")
                self.ss_matrix = compute_kmer_similarity_sparse(sequences, k_small=3, k_large=3, dtype=np.float16)
            else:
                self.ss_matrix = compute_similarity_sparse(sequences, gap_open=7, gap_extend=2, matrix=parasail.dnafull, dtype=np.float16)

            print(f"Saving similarity matrix to {self.ss_map_path}.")
            os.makedirs(self.data_storage_path + "/ss_maps", exist_ok=True)

            save_npz(self.ss_map_path, self.ss_matrix)

        elif Path(ss_map_path) != self.data_storage_path + "/ss_maps/" + Path(ss_map_path).name: # if the path is not in the data storage path, copy it there

            self.raw_data_src_path = ss_map_path

            self.ss_map_path = self.data_storage_path + "/ss_maps/" + Path(ss_map_path).name

            if not Path(self.ss_map_path).exists():
                os.makedirs(self.data_storage_path + "/ss_maps", exist_ok=True)
                shutil.copy(self.raw_data_src_path, self.ss_map_path)
            else:
                print(f"Similarity matrix already exists at {self.ss_map_path}.")

            self.ss_matrix = load_npz(Path(self.ss_map_path))
        else:
            self.ss_map_path = ss_map_path
            self.ss_matrix = load_npz(Path(self.ss_map_path))


    def split_df(
        self,
        df: pd.DataFrame,
        test_size: float,
        random_seed: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        df_seqs = df["sequence"].tolist()

        train_indices, test_indices = train_test_split_ss(
            self.sequences,
            df_seqs,
            self.ss_matrix,
            self.threshold,
            test_size,
            random_seed
        )

        assert len(set(train_indices).intersection(set(test_indices))) == 0

        train_df = df.iloc[train_indices].reset_index(drop=True)
        test_df = df.iloc[test_indices].reset_index(drop=True)

        return train_df, test_df
