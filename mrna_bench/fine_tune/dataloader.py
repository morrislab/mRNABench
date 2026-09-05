"""PyTorch Dataset and DataLoader utilities for fine-tuning."""

import numpy as np

from torch.utils.data import Dataset, DataLoader

from mrna_bench.datasets.benchmark_dataset import BenchmarkDataset


class SequenceDataset(Dataset):
    """PyTorch Dataset for sequence fine-tuning."""

    is_vep = False

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


class VEPDataset(Dataset):
    """PyTorch Dataset for VEP fine-tuning with paired ref/alt sequences."""

    is_vep = True

    def __init__(
        self,
        ref_sequences: list[str],
        alt_sequences: list[str],
        targets: np.ndarray,
        ref_cds: list[np.ndarray] | None = None,
        alt_cds: list[np.ndarray] | None = None,
        ref_splice: list[np.ndarray] | None = None,
        alt_splice: list[np.ndarray] | None = None,
    ):
        """Initialize VEPDataset.

        Args:
            ref_sequences: List of reference (wild-type) sequences.
            alt_sequences: List of alternate (variant) sequences.
            targets: Target values array.
            ref_cds: List of reference CDS tracks (optional).
            alt_cds: List of alternate CDS tracks (optional).
            ref_splice: List of reference splice tracks (optional).
            alt_splice: List of alternate splice tracks (optional).
        """
        self.ref_sequences = ref_sequences
        self.alt_sequences = alt_sequences
        self.targets = targets
        self.ref_cds = ref_cds
        self.alt_cds = alt_cds
        self.ref_splice = ref_splice
        self.alt_splice = alt_splice

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.ref_sequences)

    def __getitem__(self, idx: int) -> dict:
        """Get item by index.

        Returns:
            Dictionary with ref/alt sequences, target, and optional tracks.
        """
        item: dict = {
            "ref_sequence": self.ref_sequences[idx],
            "alt_sequence": self.alt_sequences[idx],
            "target": self.targets[idx],
        }

        if self.ref_cds is not None:
            item["ref_cds"] = self.ref_cds[idx]
        if self.alt_cds is not None:
            item["alt_cds"] = self.alt_cds[idx]
        if self.ref_splice is not None:
            item["ref_splice"] = self.ref_splice[idx]
        if self.alt_splice is not None:
            item["alt_splice"] = self.alt_splice[idx]

        return item


def collate_fn(batch: list[dict]) -> dict:
    """Collate batch keeping variable-length arrays as lists.

    Args:
        batch: List of sample dicts from SequenceDataset or VEPDataset.

    Returns:
        Collated dict with sequences as list, targets as array.
    """
    targets = np.array([item["target"] for item in batch])

    result: dict = {"target": targets}

    if "sequence" in batch[0]:
        result["sequence"] = [item["sequence"] for item in batch]
    if "ref_sequence" in batch[0]:
        result["ref_sequence"] = [item["ref_sequence"] for item in batch]
        result["alt_sequence"] = [item["alt_sequence"] for item in batch]

    for key in ("cds", "splice", "ref_cds", "alt_cds",
                "ref_splice", "alt_splice"):
        if key in batch[0]:
            result[key] = [item[key] for item in batch]

    return result


def create_dataloaders(
    dataset: BenchmarkDataset,
    target_col: str,
    split_type: str,
    random_seed: int,
    batch_size: int,
    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    num_workers: int = 4,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders from a BenchmarkDataset.

    Args:
        dataset: Benchmark dataset instance.
        target_col: Target column name in the dataset dataframe.
        split_type: Type of data split (e.g. "default", "homology").
        random_seed: Random seed for reproducible splits.
        batch_size: Batch size for DataLoaders.
        split_ratios: Train/val/test split ratios.
        num_workers: Number of DataLoader workers.
        pin_memory: Pin memory for faster GPU transfer.

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
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    train_loader = df_to_loader(splits["train_df"], shuffle=True)
    val_loader = df_to_loader(splits["val_df"], shuffle=False)
    test_loader = df_to_loader(splits["test_df"], shuffle=False)

    return train_loader, val_loader, test_loader


def create_vep_dataloaders(
    dataset: BenchmarkDataset,
    target_col: str,
    split_type: str,
    random_seed: int,
    batch_size: int,
    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    num_workers: int = 4,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders for VEP fine-tuning.

    Splits on variant rows only (excluding wild-type), then pairs each
    variant with its reference via the dataset's get_vep_pairs() method.

    Args:
        dataset: VEP BenchmarkDataset instance (must implement get_vep_pairs).
        target_col: Target column name.
        split_type: Type of data split.
        random_seed: Random seed for reproducible splits.
        batch_size: Batch size for DataLoaders.
        split_ratios: Train/val/test split ratios.
        num_workers: Number of DataLoader workers.
        pin_memory: Pin memory for faster GPU transfer.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    import pandas as pd

    data_df = dataset.data_df
    is_wt = data_df["description"].eq("wild-type")
    wt_df = data_df[is_wt]
    variant_df = data_df[~is_wt].reset_index(drop=True)

    from mrna_bench.data_splitter.split_catalog import SPLIT_CATALOG
    splitter = SPLIT_CATALOG[split_type]()
    train_df, val_df, test_df = splitter.get_all_splits_df(
        df=variant_df,
        split_ratios=split_ratios,
        random_seed=random_seed,
    )

    value_columns = tuple(
        col for col in ("sequence", "cds", "splice")
        if col in data_df.columns
    )

    def df_to_vep_loader(split_df, shuffle: bool) -> DataLoader:
        combined = pd.concat([split_df, wt_df], ignore_index=True)
        paired = dataset.get_vep_pairs(combined, value_columns=value_columns)

        ref_sequences = paired["ref_sequence"].tolist()
        alt_sequences = paired["alt_sequence"].tolist()

        raw_targets = paired[target_col].values
        if hasattr(raw_targets[0], "__len__"):
            targets = np.stack(raw_targets).astype(np.float32)
        else:
            targets = raw_targets.astype(np.float32)

        has_cds = "cds" in value_columns
        has_splice = "splice" in value_columns
        ref_cds = paired["ref_cds"].tolist() if has_cds else None
        alt_cds = paired["alt_cds"].tolist() if has_cds else None
        ref_splice = paired["ref_splice"].tolist() if has_splice else None
        alt_splice = paired["alt_splice"].tolist() if has_splice else None

        ds = VEPDataset(
            ref_sequences, alt_sequences, targets,
            ref_cds, alt_cds, ref_splice, alt_splice,
        )
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    train_loader = df_to_vep_loader(train_df, shuffle=True)
    val_loader = df_to_vep_loader(val_df, shuffle=False)
    test_loader = df_to_vep_loader(test_df, shuffle=False)

    return train_loader, val_loader, test_loader
