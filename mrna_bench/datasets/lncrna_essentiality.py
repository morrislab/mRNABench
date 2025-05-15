import pandas as pd

from mrna_bench.datasets.benchmark_dataset import BenchmarkDataset


class LNCRNAEssentiality(BenchmarkDataset):
    """Long Non-Coding RNA Essentiality Dataset."""

    def __init__(
        self,
        dataset_name: str,
        force_redownload: bool = False,
        hf_url: str | None = None
    ):
        """Initialize LNCRNAEssentiality dataset.

        Args:
            dataset_name: Dataset name formatted lncrna-ess-{experiment_name}
                where experiment_name is in: {
                    "hap1",
                    "hek293ft",
                    "k562",
                    "mda-mb-231",
                    "thp1",
                    "shared"
                }.
            force_redownload: Force raw data download even if pre-existing.
            hf_url: URL to download the dataset from Hugging Face.
        """
        if type(self) is LNCRNAEssentiality:
            raise TypeError("LNCRNAEssentiality is an abstract class.")

        valid_cell_lines = [
            "hap1",
            "hek293ft",
            "k562",
            "mda-mb-231",
            "thp1",
            "shared"
        ]

        if "-".join(dataset_name.split("-")[2:]) not in valid_cell_lines:
            raise ValueError(
                "Invalid experiment target in dataset name: {}.".format(
                    dataset_name
                )
            )

        super().__init__(
            dataset_name=dataset_name,
            species="human",
            force_redownload=force_redownload,
            hf_url=hf_url
        )

    def _get_data_from_raw(self) -> pd.DataFrame:
        raise NotImplementedError(
            "Code documenting lncRNA Essentiality is still in progress."
        )


class LNCRNAEssHAP1(LNCRNAEssentiality):
    """Concrete class for HAP1 cell line experiments."""

    def __init__(self, force_redownload=False):
        """Initialize HAP1 dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        super().__init__(
            "lncrna-ess-hap1",
            force_redownload,
            hf_url=(
                "https://huggingface.co/datasets/morrislab/"
                "lncrna-ess/resolve/main/lncrna-ess-hap1.parquet"
            )
        )


class LNCRNAEssHEK293FT(LNCRNAEssentiality):
    """Concrete class for HEK293FT cell line experiments."""

    def __init__(self, force_redownload=False):
        """Initialize HEK293FT dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        super().__init__(
            "lncrna-ess-hek293ft",
            force_redownload,
            hf_url=(
                "https://huggingface.co/datasets/morrislab/"
                "lncrna-ess/resolve/main/lncrna-ess-hek293ft.parquet"
            )
        )


class LNCRNAEssK562(LNCRNAEssentiality):
    """Concrete class for K562 cell line experiments."""

    def __init__(self, force_redownload=False):
        """Initialize K562 dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        super().__init__(
            "lncrna-ess-k562",
            force_redownload,
            hf_url=(
                "https://huggingface.co/datasets/morrislab/"
                "lncrna-ess/resolve/main/lncrna-ess-k562.parquet"
            )
        )


class LNCRNAEssMDA_MB_231(LNCRNAEssentiality):
    """Concrete class for MDA-MB-231 cell line experiments."""

    def __init__(self, force_redownload=False):
        """Initialize MDA-MB-231 dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        super().__init__(
            "lncrna-ess-mda-mb-231",
            force_redownload,
            hf_url=(
                "https://huggingface.co/datasets/morrislab/"
                "lncrna-ess/resolve/main/lncrna-ess-mda-mb-231.parquet"
            )
        )


class LNCRNAEssTHP1(LNCRNAEssentiality):
    """Concrete class for THP1 cell line experiments."""

    def __init__(self, force_redownload=False):
        """Initialize THP1 dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        super().__init__(
            "lncrna-ess-thp1",
            force_redownload,
            hf_url=(
                "https://huggingface.co/datasets/morrislab/"
                "lncrna-ess/resolve/main/lncrna-ess-thp1.parquet"
            )
        )


class LNCRNAEssShared(LNCRNAEssentiality):
    """Concrete class for Shared cell line essentiality."""

    def __init__(self, force_redownload=False):
        """Initialize Shared dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        super().__init__(
            "lncrna-ess-shared",
            force_redownload,
            hf_url=(
                "https://huggingface.co/datasets/morrislab/"
                "lncrna-ess/resolve/main/lncrna-ess-shared.parquet"
            )
        )
