import pandas as pd

from mrna_bench.datasets.benchmark_dataset import BenchmarkDataset


class GOMolecularFunction(BenchmarkDataset):
    """GO Molecular Function Dataset."""

    def __init__(self, force_redownload: bool = False):
        """Initialize GOMolecularFunction dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        super().__init__(
            dataset_name="go-mf",
            species="human",
            force_redownload=force_redownload,
            hf_url=(
                "https://huggingface.co/datasets/morrislab/"
                "go-mf/resolve/main/go_dna_dataset_mf.parquet"
            )
        )

    def _get_data_from_raw(self) -> pd.DataFrame:
        raise NotImplementedError(
            "Code documenting GO Molecular Function data is still in progress."
        )
