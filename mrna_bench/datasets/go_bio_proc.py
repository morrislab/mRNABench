import pandas as pd

from mrna_bench.datasets.benchmark_dataset import BenchmarkDataset


class GOBiologicalProcess(BenchmarkDataset):
    """GO Biological Process Dataset."""

    def __init__(self, force_redownload: bool = False):
        """Initialize GO Biological Process dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        super().__init__(
            dataset_name="go-bp",
            species="human",
            force_redownload=force_redownload,
            hf_url=(
                "https://huggingface.co/datasets/morrislab/"
                "go-bp/resolve/main/go_dna_dataset_bp.parquet"
            )
        )

    def _get_data_from_raw(self) -> pd.DataFrame:
        raise NotImplementedError(
            "Code documenting GO Biological Process data is still in progress."
        )
