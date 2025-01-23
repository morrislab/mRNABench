import numpy as np
import pandas as pd

from mrna_bench.datasets.benchmark_dataset import BenchmarkDataset
from mrna_bench.datasets.dataset_utils import ohe_to_str


MRLS_URL = "https://zenodo.org/records/14708163/files/mrl_isoform_resolved.npz"


class MRLSugimoto(BenchmarkDataset):
    """Mean Ribosome Load Dataset from Sugimoto et al. 2022."""

    def __init__(self, force_redownload: bool = False):
        """Initialize MRLSugimoto dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        super().__init__(
            dataset_name="mrl-sugimoto",
            species=["human"],
            raw_data_src_url=MRLS_URL,
            force_redownload=force_redownload
        )

    def process_raw_data(self) -> pd.DataFrame:
        """Process raw data into Pandas dataframe.

        Returns:
            Pandas dataframe of processed sequences.
        """
        data = np.load(self.raw_data_path)

        X = data["X"]

        seq_str = ohe_to_str(X[:, :, :4])
        lens = [len(s) for s in seq_str]
        cds = [X[i, :lens[i], 4] for i in range(len(X))]
        splice = [X[i, :lens[i], 5] for i in range(len(X))]

        df = pd.DataFrame({
            "sequence": seq_str,
            "target": [y for y in data["y"]],
            "gene": data["genes"],
            "transcript_length": lens,
            "cds": cds,
            "splice": splice
        })

        return df
