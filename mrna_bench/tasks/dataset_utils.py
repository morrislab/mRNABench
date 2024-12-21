import numpy as np


def ohe_to_str(
    ohe: np.array,
    nucs: list[str] = ["A", "C", "T", "G", "N"]
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
