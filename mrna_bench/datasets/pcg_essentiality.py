from mrna_bench.datasets.benchmark_dataset import BenchmarkDataset


class PCGEssentiality(BenchmarkDataset):
    """Protein Coding Gene Essentiality Dataset."""

    def __init__(
        self,
        dataset_name: str,
        force_redownload: bool = False,
        hf_url: str | None = None
    ):
        """Initialize PCGEssentiality dataset.

        Args:
            dataset_name: Dataset name formatted pcg-ess-{experiment_name}
                where experiment_name is in: {
                    "hap1",
                    "hek293ft",
                    "k562",
                    "mda-mb-231",
                    "thp1",
                    "shared"
                }.
            force_redownload: Force raw data download even if pre-existing.
            hf_url: URL to download dataset from Hugging Face.
        """
        if type(self) is PCGEssentiality:
            raise TypeError("PCGEssentiality is an abstract class.")

        valid_cell_lines = [
            "hap1",
            "hek293ft",
            "k562",
            "mda-mb-231",
            "thp1",
            "shared"
        ]

        if dataset_name.split("-")[-1] not in valid_cell_lines:
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


class PCGEssHAP1(PCGEssentiality):
    """Concrete class for HAP1 cell line experiments."""

    def __init__(self, force_redownload=False):
        """Initialize HAP1 dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        super().__init__(
            "pcg-ess-hap1",
            force_redownload,
            hf_url=(
                "https://huggingface.co/datasets/quietflamingo/"
                "pcg-ess/resolve/main/pcg-ess-hap1.parquet"
            )
        )


class PCGEssHEK293FT(PCGEssentiality):
    """Concrete class for HEK293FT cell line experiments."""

    def __init__(self, force_redownload=False):
        """Initialize HEK293FT dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        super().__init__(
            "pcg-ess-hek293ft",
            force_redownload,
            hf_url=(
                "https://huggingface.co/datasets/quietflamingo/"
                "pcg-ess/resolve/main/pcg-ess-hek293ft.parquet"
            )
        )


class PCGEssK562(PCGEssentiality):
    """Concrete class for K562 cell line experiments."""

    def __init__(self, force_redownload=False):
        """Initialize K562 dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        super().__init__(
            "pcg-ess-k562",
            force_redownload,
            hf_url=(
                "https://huggingface.co/datasets/quietflamingo/"
                "pcg-ess/resolve/main/pcg-ess-k562.parquet"
            )
        )


class PCGEssMDA_MB_231(PCGEssentiality):
    """Concrete class for MDA-MB-231 cell line experiments."""

    def __init__(self, force_redownload=False):
        """Initialize MDA-MB-231 dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        super().__init__(
            "pcg-ess-mda-mb-231",
            force_redownload,
            hf_url=(
                "https://huggingface.co/datasets/quietflamingo/"
                "pcg-ess/resolve/main/pcg-ess-mda-mb-231.parquet"
            )
        )


class PCGEssTHP1(PCGEssentiality):
    """Concrete class for THP1 cell line experiments."""

    def __init__(self, force_redownload=False):
        """Initialize THP1 dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        super().__init__(
            "pcg-ess-thp1",
            force_redownload,
            hf_url=(
                "https://huggingface.co/datasets/quietflamingo/"
                "pcg-ess/resolve/main/pcg-ess-thp1.parquet"
            )
        )


class PCGEssShared(PCGEssentiality):
    """Concrete class for Shared cell line essentiality."""

    def __init__(self, force_redownload=False):
        """Initialize Shared dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        super().__init__(
            "pcg-ess-shared",
            force_redownload,
            hf_url=(
                "https://huggingface.co/datasets/quietflamingo/"
                "pcg-ess/resolve/main/pcg-ess-shared.parquet"
            )
        )
