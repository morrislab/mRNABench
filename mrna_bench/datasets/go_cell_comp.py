import pandas as pd

from mrna_bench.datasets.benchmark_dataset import (
    BenchmarkDataset,
    DatasetMetadata,
)
from mrna_bench.datasets.go_utils import build_go_dataset


HF_URL = (
    "https://huggingface.co/datasets/morrislab/"
    "go-cc/resolve/main/go_dna_dataset_cc.parquet"
)


class GOCellularComponent(BenchmarkDataset):
    """GO Cellular Component Dataset."""

    METADATA = DatasetMetadata(
        dataset_name="go-cc",
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
        """Initialize GO Cellular Component dataset.

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
        """Rebuild GO Cellular Component data from source annotations."""
        return build_go_dataset("cc", self.raw_data_dir)
