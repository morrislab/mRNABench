from hashlib import sha256
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd

from mrna_bench.datasets.benchmark_dataset import (
    BenchmarkDataset,
    DatasetMetadata,
)
from mrna_bench.utils import download_file


IRES_CLASSIFICATION_URL = (
    "https://raw.githubusercontent.com/a96123155/"
    "IRES_Prediction_Design/"
    "780a2ed9241cdc133b1673d6df5e3837c3087a51/"
    "Data/Dataset_Collection_Preparation/v2_dataset.csv"
)
IRES_CLASSIFICATION_SHA256 = (
    "33ddf2bd483b16b5e4dab8de755cbb15"
    "da41430ab2a482d41121c9cb70237f05"
)
HF_URL = (
    "https://huggingface.co/datasets/morrislab/"
    "ires-classification/resolve/main/ires-classification.parquet"
)


class IRESClassification(BenchmarkDataset):
    """Internal ribosome entry site activity classification dataset."""

    METADATA = DatasetMetadata(
        dataset_name="ires-classification",
        species="mixed",
        task=["classification"],
        target_col=["target"],
        default_split_type="default",
        benchmark_set="core",
        evaluations=("linear_probe",),
    )

    SOURCE_COLUMNS: ClassVar[list[str]] = [
        "ID",
        "Sequence",
        "IRES_class_600",
        "Source",
    ]
    EXPECTED_TARGET_COUNTS: ClassVar[dict[int, int]] = {
        0: 37_602,
        1: 9_172,
    }

    def __init__(
        self,
        force_redownload_hf: bool = False,
        force_rebuild_raw: bool = False,
    ):
        """Initialize the IRES Classification dataset.

        Args:
            force_redownload_hf: Force redownload from HuggingFace.
            force_rebuild_raw: Force redownload and rebuild from raw data.
        """
        super().__init__(
            force_redownload_hf=force_redownload_hf,
            force_rebuild_raw=force_rebuild_raw,
            hf_url=HF_URL,
        )

    @staticmethod
    def _validate_checksum(source_path: Path) -> None:
        """Validate the pinned source CSV checksum."""
        digest = sha256()
        with source_path.open("rb") as source_file:
            for chunk in iter(
                lambda: source_file.read(1024 * 1024),
                b"",
            ):
                digest.update(chunk)

        if digest.hexdigest() != IRES_CLASSIFICATION_SHA256:
            raise ValueError(
                f"Unexpected checksum for IRES source file {source_path}."
            )

    @staticmethod
    def _normalize_sequences(source_df: pd.DataFrame) -> pd.Series:
        """Normalize source sequences to mRNABench DNA strings."""
        sequence = (
            source_df["Sequence"]
            .astype(str)
            .str.strip()
            .str.upper()
            .str.replace("U", "T", regex=False)
        )
        if not sequence.str.fullmatch(r"[ACGTN]+").all():
            raise ValueError(
                "IRES source sequences contain characters outside ACGTN."
            )
        if sequence.duplicated().any():
            raise ValueError("IRES source sequences must be unique.")
        return sequence

    def _get_data_from_raw(self) -> pd.DataFrame:
        """Download and process the author-provided IRES source CSV."""
        print("Downloading raw data...")
        source_path = Path(
            download_file(
                IRES_CLASSIFICATION_URL,
                self.raw_data_dir,
                force_redownload=self.force_rebuild_raw,
            )
        )
        self._validate_checksum(source_path)

        try:
            source_df = pd.read_csv(source_path)
        except (OSError, pd.errors.ParserError) as exc:
            raise RuntimeError(
                f"Failed to read IRES source file {source_path}."
            ) from exc

        missing_columns = set(self.SOURCE_COLUMNS) - set(source_df.columns)
        if missing_columns:
            raise ValueError(
                "IRES source file is missing required columns: "
                f"{sorted(missing_columns)}"
            )
        if source_df[self.SOURCE_COLUMNS].isna().any().any():
            raise ValueError("IRES source file contains missing values.")

        transcript_id = source_df["ID"].astype(str)
        if transcript_id.duplicated().any():
            raise ValueError("IRES transcript identifiers must be unique.")

        sequence = self._normalize_sequences(source_df)
        target = pd.to_numeric(
            source_df["IRES_class_600"],
            errors="raise",
        ).astype(np.int8)
        target_counts = target.value_counts().sort_index().to_dict()
        if target_counts != self.EXPECTED_TARGET_COUNTS:
            raise ValueError(
                "Unexpected IRES target distribution: "
                f"{target_counts}."
            )

        sequence_lengths = sequence.str.len()
        cds_tracks = {
            length: np.zeros(length, dtype=np.int8)
            for length in sequence_lengths.unique()
        }
        splice_tracks = {
            length: np.zeros(length, dtype=np.int8)
            for length in sequence_lengths.unique()
        }

        return pd.DataFrame(
            {
                "transcript_id": transcript_id,
                "sequence": sequence,
                "cds": sequence_lengths.map(cds_tracks),
                "splice": sequence_lengths.map(splice_tracks),
                "target": target,
                "description": source_df["Source"].astype(str).map(
                    "source={}".format
                ),
            }
        )
