from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import requests

from mrna_bench.datasets.benchmark_dataset import (
    BenchmarkDataset,
    DatasetMetadata,
)


GOOGLE_DRIVE_URL = (
    "https://drive.usercontent.google.com/download"
    "?id={file_id}&export=download&confirm=t"
)
HF_URL = (
    "https://huggingface.co/datasets/morrislab/"
    "apa-isoform/resolve/main/apa-isoform.parquet"
)


class APAIsoform(BenchmarkDataset):
    """Proximal APA isoform usage in synthetic APARENT 3' UTR windows."""

    METADATA = DatasetMetadata(
        dataset_name="apa-isoform",
        species="synthetic",
        task=["regression"],
        target_col=["target"],
        default_split_type="default",
        benchmark_set="core",
        evaluations=("linear_probe",),
    )

    SOURCE_FILE_IDS: ClassVar[dict[str, str]] = {
        "train": "1iQi8AzEkQk-scxwAc-vQmWiRrnn4Vxkv",
        "val": "1t42XgBp3inxi_9IAtNququjA5IgX48yB",
        "test": "153p_-MLNtlH4T0-aWA6Qg-LCA4aVghFX",
    }
    SOURCE_COLUMNS: ClassVar[list[str]] = [
        "library",
        "seq",
        "proximal_isoform_proportion",
        "library_index",
    ]

    def __init__(
        self,
        force_redownload_hf: bool = False,
        force_rebuild_raw: bool = False,
    ):
        """Initialize the APA Isoform dataset.

        Args:
            force_redownload_hf: Force redownload from HuggingFace.
            force_rebuild_raw: Force redownload and rebuild from raw CSVs.
        """
        super().__init__(
            force_redownload_hf=force_redownload_hf,
            force_rebuild_raw=force_rebuild_raw,
            hf_url=HF_URL,
        )

    def _download_source_file(
        self,
        file_id: str,
        destination: Path,
    ) -> None:
        """Download one BEACON source CSV from Google Drive."""
        url = GOOGLE_DRIVE_URL.format(file_id=file_id)
        partial_path = destination.with_suffix(destination.suffix + ".part")

        try:
            with requests.get(
                url,
                stream=True,
                timeout=(10, 120),
            ) as response:
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").lower()
                if "text/html" in content_type:
                    raise RuntimeError(
                        "Google Drive returned HTML instead of the "
                        f"APA Isoform CSV for {destination.name}."
                    )

                with partial_path.open("wb") as output:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            output.write(chunk)

            partial_path.replace(destination)
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Failed to download APA Isoform source file "
                f"{destination.name} from Google Drive."
            ) from exc
        finally:
            partial_path.unlink(missing_ok=True)

    def _download_raw_data(self) -> None:
        """Download missing BEACON APA Isoform CSV files."""
        raw_data_dir = Path(self.raw_data_dir)
        for split, file_id in self.SOURCE_FILE_IDS.items():
            destination = raw_data_dir / f"{split}.csv"
            if destination.exists() and not self.force_rebuild_raw:
                continue

            print(f"Downloading APA Isoform {split}.csv from Google Drive...")
            self._download_source_file(file_id, destination)

    @staticmethod
    def _normalize_sequences(source_df: pd.DataFrame) -> pd.Series:
        """Normalize source sequences to mRNABench DNA strings."""
        raw_sequence = (
            source_df["seq"]
            .astype(str)
            .str.strip()
            .str.upper()
            .str.replace("U", "T", regex=False)
        )
        if not raw_sequence.str.len().eq(186).all():
            raise ValueError(
                "APA Isoform source sequences must all be 186 characters."
            )

        padded = raw_sequence.str.startswith("X" * 10)
        double_dope = source_df["library"].astype(str).eq("DoubleDope")
        no_other_x = ~raw_sequence.str[10:].str.contains("X", regex=False)
        valid_padding = padded & double_dope & no_other_x
        contains_x = raw_sequence.str.contains("X", regex=False)
        invalid_x = contains_x & ~valid_padding
        if invalid_x.any():
            raise ValueError(
                "APA Isoform X characters must be the ten-character "
                "leading padding used by DoubleDope sequences."
            )

        sequence = raw_sequence.where(
            ~valid_padding,
            raw_sequence.str[10:],
        )
        if not sequence.str.fullmatch(r"[ACGTN]+").all():
            raise ValueError(
                "APA Isoform sequences contain characters outside ACGTN."
            )
        return sequence

    def _get_data_from_raw(self) -> pd.DataFrame:
        """Download and process the BEACON APA Isoform source CSVs."""
        self._download_raw_data()

        raw_frames = []
        for split in self.SOURCE_FILE_IDS:
            source_path = Path(self.raw_data_dir) / f"{split}.csv"
            try:
                frame = pd.read_csv(source_path)
            except (OSError, pd.errors.ParserError) as exc:
                raise RuntimeError(
                    f"Failed to read APA Isoform source file {source_path}."
                ) from exc

            missing_columns = set(self.SOURCE_COLUMNS) - set(frame.columns)
            if missing_columns:
                raise ValueError(
                    f"{source_path.name} is missing required columns: "
                    f"{sorted(missing_columns)}"
                )
            if frame[self.SOURCE_COLUMNS].isna().any().any():
                raise ValueError(
                    f"{source_path.name} contains missing required values."
                )

            raw_frames.append(
                frame[self.SOURCE_COLUMNS].assign(beacon_split=split)
            )

        source_df = pd.concat(raw_frames, ignore_index=True)
        sequence = self._normalize_sequences(source_df)
        if sequence.duplicated().any():
            raise ValueError("APA Isoform source sequences must be unique.")

        target = pd.to_numeric(
            source_df["proximal_isoform_proportion"],
            errors="raise",
        )
        if not target.between(0.0, 1.0, inclusive="both").all():
            raise ValueError("APA Isoform targets must be between 0 and 1.")

        description = source_df["beacon_split"].astype(str).map(
            "beacon_split={}".format
        )
        description += source_df["library"].astype(str).map(
            ";library={}".format
        )
        description += source_df["library_index"].astype(str).map(
            ";library_index={}".format
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
                "sequence": sequence,
                "cds": sequence_lengths.map(cds_tracks),
                "splice": sequence_lengths.map(splice_tracks),
                "target": target,
                "description": description,
            }
        )
