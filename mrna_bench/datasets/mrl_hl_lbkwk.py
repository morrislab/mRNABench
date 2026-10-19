import pandas as pd

from mrna_bench.datasets.benchmark_dataset import (
    BenchmarkDataset,
    DatasetMetadata,
)


HF_URL = (
    "https://huggingface.co/datasets/morrislab/"
    "mrl-hl-lbkwk/resolve/main/mrl-hl-lbkwk.parquet"
)


class MRLHLLBKWK(BenchmarkDataset):
    """Paired MRL and HL dataset from Leppek et al. 2022."""

    METADATA = DatasetMetadata(
        dataset_name="mrl-hl-lbkwk",
        species="synthetic",
        task=["regression"],
        target_col=["target_in_cell_half_life", "target_ribosome_load"],
        default_split_type="default",
        benchmark_set="core",
        vep=False,
    )

    def __init__(
        self,
        force_redownload_hf: bool = False,
        force_rebuild_raw: bool = False,
    ):
        """Initialize MRLHLLBKWK dataset.

        Args:
            force_redownload_hf: Force redownload from HuggingFace.
            force_rebuild_raw: Force rebuild from raw data source.
        """
        super().__init__(
            force_redownload_hf=force_redownload_hf,
            force_rebuild_raw=force_rebuild_raw,
            hf_url=HF_URL,
        )

    def _get_data_from_raw(self) -> pd.DataFrame:
        raise NotImplementedError(
            "Code documenting MRL/HL LBKWK is still in progress."
        )
