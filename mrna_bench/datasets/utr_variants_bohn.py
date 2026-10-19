import pandas as pd

from mrna_bench.datasets.benchmark_dataset import (
    BenchmarkDataset,
    DatasetMetadata,
)


class UTRVariantsBohn(BenchmarkDataset):
    """UTR Variants dataset from Bohn et al."""

    def __init__(
        self,
        force_redownload_hf: bool = False,
        force_rebuild_raw: bool = False,
        hf_url: str = "",
    ):
        """Initialize UTRVariantsBohn dataset.

        Args:
            force_redownload_hf: Force redownload from HuggingFace.
            force_rebuild_raw: Force rebuild from raw data source.
            hf_url: URL to download the dataset from Hugging Face.
        """
        if type(self) is UTRVariantsBohn:
            raise TypeError("UTRVariantsBohn is an abstract class.")

        self.utr_type = self.METADATA.dataset_name.split("-")[-1]
        assert self.utr_type in ["utr5", "utr3"]

        super().__init__(
            force_redownload_hf=force_redownload_hf,
            force_rebuild_raw=force_rebuild_raw,
            hf_url=hf_url,
        )

    def _get_data_from_raw(self) -> pd.DataFrame:
        raise NotImplementedError(
            "Code documenting UTR variants data is still in progress."
        )


class UTRVariantsBohnUTR5(UTRVariantsBohn):
    """Concrete class for UTR Variants dataset (5' UTR)."""

    METADATA = DatasetMetadata(
        dataset_name="utr-variants-bohn-utr5",
        species="human",
        task=["classification", "zeroshot"],
        target_col=["target"],
        default_split_type="default",
        benchmark_set="core",
        vep=True,
    )

    def __init__(
        self,
        force_redownload_hf: bool = False,
        force_rebuild_raw: bool = False,
    ):
        """Initialize UTRVariantsBohnUTR5 dataset.

        Args:
            force_redownload_hf: Force redownload from HuggingFace.
            force_rebuild_raw: Force rebuild from raw data source.
        """
        super().__init__(
            force_redownload_hf=force_redownload_hf,
            force_rebuild_raw=force_rebuild_raw,
            hf_url=(
                "https://huggingface.co/datasets/morrislab/"
                "utr-variants-bohn/resolve/main/"
                "utr-variants-bohn-utr5.parquet"
            ),
        )


class UTRVariantsBohnUTR3(UTRVariantsBohn):
    """Concrete class for UTR Variants dataset (3' UTR)."""

    METADATA = DatasetMetadata(
        dataset_name="utr-variants-bohn-utr3",
        species="human",
        task=["classification", "zeroshot"],
        target_col=["target"],
        default_split_type="default",
        benchmark_set="core",
        vep=True,
    )

    def __init__(
        self,
        force_redownload_hf: bool = False,
        force_rebuild_raw: bool = False,
    ):
        """Initialize UTRVariantsBohnUTR3 dataset.

        Args:
            force_redownload_hf: Force redownload from HuggingFace.
            force_rebuild_raw: Force rebuild from raw data source.
        """
        super().__init__(
            force_redownload_hf=force_redownload_hf,
            force_rebuild_raw=force_rebuild_raw,
            hf_url=(
                "https://huggingface.co/datasets/morrislab/"
                "utr-variants-bohn/resolve/main/"
                "utr-variants-bohn-utr3.parquet"
            ),
        )
