import pickle
import pandas as pd 
import os 

from pathlib import Path
from mrna_bench.datasets.benchmark_dataset import BenchmarkDataset


LNCRNA_URL = "/data1/morrisq/ian/rna_benchmarks/lncrna-ess/HAP1_lncRNA_ess.pkl"


class LNCRNAEssentiality(BenchmarkDataset):
    """Long Non-Coding RNA Essentiality Dataset."""

    def __init__(self, force_redownload: bool = False):
        super().__init__(
            dataset_name="lncrna-ess",
            species=["human"],
            force_redownload=force_redownload
        )

    def process_raw_data(self) -> pd.DataFrame:
        """Process raw data into Pandas dataframe.

        Returns:
            Pandas dataframe of processed sequences.
        """

        with open(self.raw_data_path, "rb") as f:
            data = pickle.load(f)

        return data

    def get_raw_data(self):
        """Collect the raw data from given local path."""
        raw_file_name = Path(LNCRNA_URL).name
        raw_data_path = self.raw_data_dir + "/" + raw_file_name

        if not os.path.exists(raw_data_path):
            shutil.copy(self.PCG_URL, raw_data_path)
            self.first_download = True

        self.raw_data_path = raw_data_path