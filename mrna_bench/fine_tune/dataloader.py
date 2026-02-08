"""PyTorch Dataset and DataLoader utilities for fine-tuning."""

import numpy as np

from torch.utils.data import Dataset, DataLoader

from mrna_bench.datasets.benchmark_dataset import BenchmarkDataset


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


def collate_fn(batch: list[dict]) -> dict:
    """Collate batch keeping variable-length arrays as lists.

    Args:
        batch: List of sample dicts from SequenceDataset.

    Returns:
        Collated dict with sequences as list, targets as array.
    """
    sequences = [item["sequence"] for item in batch]
    targets = np.array([item["target"] for item in batch])

    result: dict = {
        "sequence": sequences,
        "target": targets,
    }

    if "cds" in batch[0]:
        result["cds"] = [item["cds"] for item in batch]
    if "splice" in batch[0]:
        result["splice"] = [item["splice"] for item in batch]

    return result


def create_dataloaders(
    dataset: BenchmarkDataset,
    target_col: str,
    split_type: str,
    random_seed: int,
    batch_size: int,
    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders from a BenchmarkDataset.

    Args:
        dataset: Benchmark dataset instance.
        target_col: Target column name in the dataset dataframe.
        split_type: Type of data split (e.g. "default", "homology").
        random_seed: Random seed for reproducible splits.
        batch_size: Batch size for DataLoaders.
        split_ratios: Train/val/test split ratios.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    splits = dataset.get_splits(
        split_ratios=split_ratios,
        random_seed=random_seed,
        split_type=split_type,
    )

    def df_to_loader(df, shuffle: bool) -> DataLoader:
        sequences = df["sequence"].tolist()

        raw_targets = df[target_col].values
        if hasattr(raw_targets[0], "__len__"):
            targets = np.stack(raw_targets).astype(np.float32)
        else:
            targets = raw_targets.astype(np.float32)

        cds = df["cds"].tolist() if "cds" in df.columns else None
        splice = df["splice"].tolist() if "splice" in df.columns else None

        ds = SequenceDataset(sequences, targets, cds, splice)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
        )

    train_loader = df_to_loader(splits["train_df"], shuffle=True)
    val_loader = df_to_loader(splits["val_df"], shuffle=False)
    test_loader = df_to_loader(splits["test_df"], shuffle=False)

    return train_loader, val_loader, test_loader
