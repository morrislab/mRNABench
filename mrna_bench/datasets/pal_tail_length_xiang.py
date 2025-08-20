import numpy as np
import pandas as pd

from mrna_bench.datasets.benchmark_dataset import BenchmarkDataset
from mrna_bench.datasets.dataset_utils import ohe_to_str
from mrna_bench.utils import download_file

class PALTailLengthHuman(BenchmarkDataset):
    """PAL Tail Length Human Dataset."""

    def __init__(
        self,
        dataset_name: str,
        species: str,
        force_redownload: bool = False,
        hf_url: str | None = None
    ):
        """Initialize PALTailLengthHuman dataset.

        Args:
            dataset_name: Dataset name formatted pal-tail-length-xiang-{subset}
                where subset is in: {
                    "gv",
                    "gvtomii"
                }.
            force_redownload: Force raw data download even if pre-existing.
            hf_url: URL to download the dataset from Hugging Face.
        """
        if type(self) is PALTailLengthHuman:
            raise TypeError("PALTailLengthHuman is an abstract class.")

        self.subset = dataset_name.split("-")[-1]
        assert self.subset in ["gv", "gvtomii", "p4diff", "p4initial"]

        super().__init__(dataset_name, species, force_redownload, hf_url)

    def _get_data_from_raw(self) -> pd.DataFrame:
        raise NotImplementedError(
            "Code documenting PAL tail length data is still in progress."
        )

class PALTailLengthGV(PALTailLengthHuman):
    """Concrete class for PAL Tail Length dataset (GV cell type)."""

    def __init__(self, force_redownload=False):
        """Initialize PALTailLengthGV dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        super().__init__(
            "pal-tail-length-xiang-gv",
            force_redownload,
            hf_url=(
                "https://huggingface.co/datasets/morrislab/"
                "pal-tail-length-xiang/resolve/main/"
                "pal-tail-length-xiang-gv.parquet"
            )
        )


class PALTailLengthGVTomii(PALTailLengthHuman):
    """Concrete class for PAL Tail Length dataset (GVTomii cell type)."""

    def __init__(self, force_redownload=False):
        """Initialize PALTailLengthGVTomii dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        super().__init__(
            "pal-tail-length-xiang-gvtomii",
            force_redownload,
            hf_url=(
                "https://huggingface.co/datasets/morrislab/"
                "pal-tail-length-xiang/resolve/main/"
                "pal-tail-length-xiang-gvtomii.parquet"
            )
        )

class PALTailLengthP4Diff(PALTailLengthHuman):
    """Concrete class for PAL Tail Length dataset (0-7 hour progesterone treatment)."""

    def __init__(self, force_redownload=False):
        """Initialize PALTailLengthP4Diff dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        super().__init__(
            "pal-tail-length-xiang-p4diff",
            "frog",
            force_redownload,
            hf_url=(
                "https://huggingface.co/datasets/morrislab/"
                "pal-tail-length-xiang/resolve/main/"
                "pal-tail-length-xiang-p4diff.parquet"
            )
        )

class PALTailLengthP4Initial(PALTailLengthHuman):
    """Concrete class for PAL Tail Length dataset (Pre-progesterone treatment)."""

    def __init__(self, force_redownload=False):
        """Initialize PALTailLengthP4Initial dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        super().__init__(
            "pal-tail-length-xiang-p4initial",
            "frog",
            force_redownload,
            hf_url=(
                "https://huggingface.co/datasets/morrislab/"
                "pal-tail-length-xiang/resolve/main/"
                "pal-tail-length-xiang-p4initial.parquet"
            )
        )