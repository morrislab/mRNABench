from io import BytesIO
from zipfile import ZipFile

import numpy as np
import pandas as pd

from mrna_bench.datasets.benchmark_dataset import (
    BenchmarkDataset,
    DatasetMetadata,
)
from mrna_bench.utils import download_file


HF_URL = (
    "https://huggingface.co/datasets/morrislab/"
    "mrl-hl-lbkwk/resolve/main/mrl-hl-lbkwk.parquet"
)
SOURCE_URL = (
    "https://media.springernature.com/original/springer-static/esm/"
    "art%3A10.1038%2Fs41467-022-28776-w/MediaObjects/"
    "41467_2022_28776_MOESM4_ESM.zip"
)
SOURCE_WORKBOOK = (
    "Supplementary Data 1 - Attributes for pooled 233 sequences.xlsx"
)
SOURCE_SHEET = "Attributes for pooled 233 seque"
COMPONENT_COLUMNS = [
    "Sequence 5 UTR",
    "Sequence CDS",
    "Sequence Constant1",
    "Sequence Barcode",
    "Sequence Constant2",
    "Sequence 3 UTR",
    "Sequence Constant3",
]


class MRLHLLBKWK(BenchmarkDataset):
    """Paired MRL and HL dataset from Leppek et al. 2022."""

    METADATA = DatasetMetadata(
        dataset_name="mrl-hl-lbkwk",
        species="synthetic",
        task=["regression"],
        target_col=["target_in_cell_half_life", "target_ribosome_load"],
        default_split_type="default",
        benchmark_set="core",
        evaluations=("linear_probe",),
    )

    def __init__(
        self,
        force_redownload_hf: bool = False,
        force_rebuild_raw: bool = False,
    ):
        """Initialize MRLHLLBKWK dataset.

        Args:
            force_redownload_hf: Force redownload from HuggingFace.
            force_rebuild_raw: Force rebuild from raw data source.
        """
        super().__init__(
            force_redownload_hf=force_redownload_hf,
            force_rebuild_raw=force_rebuild_raw,
            hf_url=HF_URL,
        )

    def _get_data_from_raw(self) -> pd.DataFrame:
        """Download and process the publication's construct attributes."""
        archive_path = download_file(SOURCE_URL, self.raw_data_dir)
        with ZipFile(archive_path) as archive:
            workbook = BytesIO(archive.read(SOURCE_WORKBOOK))
        source = pd.read_excel(workbook, sheet_name=SOURCE_SHEET)
        has_ribosome_load = source["Ribosome load"].notna()
        has_half_life = source["In-cell half-life"].notna()
        source = source.loc[has_ribosome_load & has_half_life]
        sequences = (
            source["RNA sequence"]
            .str.replace("U", "T", regex=False)
            .str[3:]
            .astype("str")
        )

        cds_tracks = []
        splice_tracks = []
        for (_, row), sequence in zip(
            source.iterrows(),
            sequences,
            strict=True,
        ):
            cds_start = len(row["Sequence 5 UTR"])
            cds_length = len(row["Sequence CDS"])
            component_length = sum(
                len(row[column]) for column in COMPONENT_COLUMNS
            )
            if cds_length % 3 != 0 or len(sequence) != component_length:
                raise ValueError(
                    f"Invalid construct layout for {row['Sequence ID']}."
                )

            cds = np.zeros(len(sequence), dtype=np.int8)
            cds[cds_start : cds_start + cds_length : 3] = 1
            cds_tracks.append(cds)
            splice_tracks.append(
                np.zeros(len(sequence), dtype=np.int8)
            )

        return pd.DataFrame({
            "transcript_id": source["Sequence ID"].astype("str").to_numpy(),
            "sequence": sequences.to_numpy(),
            "cds": cds_tracks,
            "splice": splice_tracks,
            "target_ribosome_load": (
                source["Ribosome load"].astype("float32").to_numpy()
            ),
            "target_in_cell_half_life": (
                source["In-cell half-life"].astype("float32").to_numpy()
            ),
        })
