import numpy as np
import pandas as pd

from mrna_bench.tasks.benchmark_dataset import BenchmarkDataset
from mrna_bench.tasks.dataset_utils import ohe_to_str


GOMF_URL = "https://zenodo.org/records/13910050/files/go_dna_dataset.npz"


class GOMolecularFunction(BenchmarkDataset):
    """GO Molecular Function Dataset."""
    def __init__(self):
        super().__init__(
            dataset_name="GO Molecular Function",
            short_name="go-mf",
            description="TODO",
            raw_data_src_url=GOMF_URL,
        )

    def reformat_raw_data(self):
        """Reprocess raw data to reduce storage size."""
        data = np.load(self.raw_data_path)

        np.savez_compressed(
            self.raw_data_path,
            X=data["X"].astype(np.int8),
            y=data["y"].astype(np.int8),
            genes=data["genes"]
        )

    def process_raw_data(self) -> pd.DataFrame:
        """Process raw data into Pandas dataframe.

        Returns:
            Pandas dataframe of processed sequences.
        """
        if self.first_download:
            self.reformat_raw_data()

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
