import pandas as pd

from mrna_bench.datasets.benchmark_dataset import BenchmarkDataset


class GOCellularComponent(BenchmarkDataset):
    """GO Cellular Component Dataset."""

    def __init__(self, force_redownload: bool = False):
        """Initialize GO Cellular Component dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        super().__init__(
            dataset_name="go-cc",
            species="human",
            force_redownload=force_redownload,
            hf_url=(
                "https://huggingface.co/datasets/morrislab/"
                "go-cc/resolve/main/go_dna_dataset_cc.parquet"
            )
        )

    def _get_data_from_raw(self) -> pd.DataFrame:
        raise NotImplementedError(
            "Code documenting GO Cellular Component data is still in progress."
        )
