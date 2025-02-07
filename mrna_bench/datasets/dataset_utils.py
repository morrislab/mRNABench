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
