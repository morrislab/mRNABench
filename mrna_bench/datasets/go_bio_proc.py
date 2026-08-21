import pandas as pd

from mrna_bench.datasets.benchmark_dataset import (
    BenchmarkDataset,
    DatasetMetadata,
)
from mrna_bench.datasets.go_utils import build_go_dataset


HF_URL = (
    "https://huggingface.co/datasets/morrislab/"
    "go-bp/resolve/main/go_dna_dataset_bp.parquet"
)


class GOBiologicalProcess(BenchmarkDataset):
    """GO Biological Process Dataset."""

    METADATA = DatasetMetadata(
        dataset_name="go-bp",
        species="human",
        task=["multilabel"],
        target_col=["target"],
        default_split_type="homology",
        benchmark_set="extended",
        evaluations=("linear_probe",),
    )

    def __init__(
        self,
        force_redownload_hf: bool = False,
        force_rebuild_raw: bool = False,
    ):
        """Initialize GO Biological Process dataset.

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
        """Rebuild GO Biological Process data from source annotations."""
        return build_go_dataset("bp", self.raw_data_dir)
