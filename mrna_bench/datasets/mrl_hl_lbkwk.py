import pandas as pd

from mrna_bench.datasets.benchmark_dataset import BenchmarkDataset


class MRLHLLBKWK(BenchmarkDataset):
    """Paired MRL and HL dataset from Leppek et al. 2022."""

    def __init__(self, force_redownload: bool = False):
        """Initialize MRLHLLBKWK dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        super().__init__(
            dataset_name="mrl-hl-lbkwk",
            species="synthetic",
            force_redownload=force_redownload,
            hf_url=(
                "https://huggingface.co/datasets/morrislab/"
                "mrl-hl-lbkwk/resolve/main/mrl-hl-lbkwk.parquet"
            )
        )

    def _get_data_from_raw(self) -> pd.DataFrame:
        raise NotImplementedError(
            "Code documenting MRL/HL LBKWK is still in progress."
        )
