import pandas as pd

from mrna_bench.datasets.benchmark_dataset import BenchmarkDataset


class RNALifecycleIetswaart(BenchmarkDataset):
    """RNA Lifecycle Dataset."""

    def __init__(self, force_redownload: bool = False):
        """Initialize RNALifecycleIetswaart dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        super().__init__(
            dataset_name="rna-lifecycle-ietswaart",
            species="human",
            force_redownload=force_redownload,
            hf_url=(
                "https://huggingface.co/datasets/morrislab/"
                "rna-lifecycle-ietswaart/resolve/main/ietswaart_processed.parquet"
            )
        )

    def _get_data_from_raw(self) -> pd.DataFrame:
        raise NotImplementedError(
            "Code documenting RNA lifecycle data is still in progress."
        ) 