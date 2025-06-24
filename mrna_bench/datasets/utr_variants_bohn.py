import numpy as np
import pandas as pd

from mrna_bench.datasets.benchmark_dataset import BenchmarkDataset
from mrna_bench.datasets.dataset_utils import ohe_to_str
from mrna_bench.utils import download_file

class UTRVariantsBohn(BenchmarkDataset):
    """UTR Variants dataset from Bohn et al."""

    def __init__(
        self,
        dataset_name: str,
        force_redownload: bool = False,
        hf_url: str | None = None
    ):
        """Initialize PALTailLengthHuman dataset.

        Args:
            dataset_name: Dataset name formatted utr-variants-bohn-{utr_type}
                where utr_type is in: {
                    "utr5",
                    "utr3",
                }.
            force_redownload: Force raw data download even if pre-existing.
            hf_url: URL to download the dataset from Hugging Face.
        """
        if type(self) is UTRVariantsBohn:
            raise TypeError("UTRVariantsBohn is an abstract class.")

        self.cell_type = dataset_name.split("-")[-1]
        assert self.cell_type in ["utr5", "utr3"]

        super().__init__(dataset_name, "human", force_redownload, hf_url)

    def _get_data_from_raw(self) -> pd.DataFrame:
        raise NotImplementedError(
            "Code documenting PAL tail length data is still in progress."
        )

class UTRVariantsBohnUTR5(UTRVariantsBohn):
    """Concrete class for UTR Variants dataset (5' UTR)."""

    def __init__(self, force_redownload=False):
        """Initialize UTRVariantsBohnUTR5 dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        super().__init__(
            "utr-variants-bohn-utr5",
            force_redownload,
            hf_url=(
                "https://huggingface.co/datasets/morrislab/"
                "utr-variants-bohn/resolve/main/"
                "utr-variants-bohn-utr5.parquet"
            )
        )

class UTRVariantsBohnUTR3(UTRVariantsBohn):
    """Concrete class for UTR Variants dataset (3' UTR)."""

    def __init__(self, force_redownload=False):
        """Initialize UTRVariantsBohnUTR3 dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        super().__init__(
            "utr-variants-bohn-utr3",
            force_redownload,
            hf_url=(
                "https://huggingface.co/datasets/morrislab/"
                "utr-variants-bohn/resolve/main/"
                "utr-variants-bohn-utr3.parquet"
            )
        )