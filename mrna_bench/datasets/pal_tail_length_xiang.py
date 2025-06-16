import numpy as np
import pandas as pd

from mrna_bench.datasets.benchmark_dataset import BenchmarkDataset
from mrna_bench.datasets.dataset_utils import ohe_to_str
from mrna_bench.utils import download_file

class PALTailLengthHuman(BenchmarkDataset):
    """PAL Tail Length Human Dataset."""

    def __init__(self, force_redownload: bool = False):
        """Initialize PALTailLengthHuman dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        super().__init__(
            dataset_name="pal-tail-length-xiang",
            species="human",
            force_redownload=force_redownload,
            hf_url=(
                "https://huggingface.co/datasets/morrislab/"
                "pal-tail-length-xiang/resolve/main/"
                "pal-tail-length-xiang.parquet"
            )
        )

    def _get_data_from_raw(self) -> pd.DataFrame:
        raise NotImplementedError(
            "Code documenting PAL tail length data is still in progress."
        )
