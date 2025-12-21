import pandas as pd

from mrna_bench.datasets.benchmark_dataset import BenchmarkDataset


class VEPMaPSy(BenchmarkDataset):
    """MaPSy benchmark for variant effect prediction subsetted for CDS."""

    def __init__(self, force_redownload: bool = False):
        """Initialize MaPSy dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        super().__init__(
            dataset_name="vep-mapsy",
            species="human",
            force_redownload=force_redownload,
            hf_url=(
                "https://huggingface.co/datasets/morrislab/"
                "vep-mapsy/resolve/main/vep-mapsy.parquet"
            )
        )

    def _get_data_from_raw(self) -> pd.DataFrame:
        raise NotImplementedError(
            "Code documenting MaPSy data is still in progress."
        )
