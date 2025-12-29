import pandas as pd

from mrna_bench.datasets.benchmark_dataset import (
    BenchmarkDataset,
    DatasetMetadata
)


HF_URL = (
    "https://huggingface.co/datasets/morrislab/"
    "rna-lifecycle-ietswaart/resolve/main/"
    "ietswaart_processed.parquet"
)


class RNALifecycleIetswaart(BenchmarkDataset):
    """RNA Lifecycle Dataset."""

    METADATA = DatasetMetadata(
        dataset_name="rna-lifecycle-ietswaart",
        species="human",
        task=["multilabel"],
        target_col=["target"],
        default_split_type="homology",
        benchmark_set="core"
    )

    def __init__(
        self,
        force_redownload_hf: bool = False,
        force_rebuild_raw: bool = False
    ):
        """Initialize RNALifecycleIetswaart dataset.

        Args:
            force_redownload_hf: Force redownload from HuggingFace.
            force_rebuild_raw: Force rebuild from raw data source.
        """
        super().__init__(
            force_redownload_hf=force_redownload_hf,
            force_rebuild_raw=force_rebuild_raw,
            hf_url=HF_URL
        )

    def _get_data_from_raw(self) -> pd.DataFrame:
        raise NotImplementedError(
            "Code documenting RNA lifecycle data is still in progress."
        )
