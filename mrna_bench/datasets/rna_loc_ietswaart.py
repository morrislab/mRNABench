import pandas as pd

from mrna_bench.datasets.benchmark_dataset import BenchmarkDataset


class RNALocalizationIetswaart(BenchmarkDataset):
    """RNA Subcellular Localization Dataset."""

    def __init__(self, force_redownload: bool = False):
        """Initialize RNALocalization dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        super().__init__(
            dataset_name="rna-loc-ietswaart",
            species="human",
            force_redownload=force_redownload,
            hf_url=(
                "https://huggingface.co/datasets/morrislab/"
                "rna-loc-ietswaart/resolve/main/rna-loc-ietswaart.parquet"
            )
        )

    def _get_data_from_raw(self) -> pd.DataFrame:
        raise NotImplementedError(
            "Code documenting RNA localization data is still in progress."
        )
