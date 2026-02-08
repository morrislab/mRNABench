import numpy as np

from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """PyTorch Dataset for sequence fine-tuning."""

    def __init__(
        self,
        sequences: list[str],
        targets: np.ndarray,
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
    ):
        """Initialize SequenceDataset.

        Args:
            sequences: List of nucleotide sequences.
            targets: Target values array.
            cds: List of CDS tracks (optional).
            splice: List of splice tracks (optional).
        """
        self.sequences = sequences
        self.targets = targets
        self.cds = cds
        self.splice = splice

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        """Get item by index.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with sequence, target, and optional tracks.
        """
        item = {
            "sequence": self.sequences[idx],
            "target": self.targets[idx],
        }

        if self.cds is not None:
            item["cds"] = self.cds[idx]
        if self.splice is not None:
            item["splice"] = self.splice[idx]

        return item
