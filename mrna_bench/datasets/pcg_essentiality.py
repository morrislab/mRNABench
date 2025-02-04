from pathlib import Path
import shutil

import pandas as pd

from mrna_bench.datasets.benchmark_dataset import BenchmarkDataset


LNCRNA_URL = "/home/shir2/mRNABench/data/HAP1_essentiality_data.tsv"


class PCGEssentiality(BenchmarkDataset):
    """Protein Coding Gene Essentiality Dataset."""

    def __init__(self, force_redownload: bool = False):
        """Initialize PCGEssentiality dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        super().__init__(
            dataset_name="pcg-ess",
            species=["human"],
            force_redownload=force_redownload
        )

    def get_raw_data(self):
        """Download raw data from source."""
        raw_file_name = Path(LNCRNA_URL).name
        raw_data_path = self.raw_data_dir + "/" + raw_file_name

        shutil.copy(LNCRNA_URL, raw_data_path)

        self.raw_data_path = raw_data_path

    def process_raw_data(self) -> pd.DataFrame:
        """Process raw data into Pandas dataframe."""
        data_df = pd.read_csv(self.raw_data_path, delimiter="\t")

        data_df = data_df[data_df["type"] == "pcg"]
        data_df.reset_index(inplace=True, drop=True)
        return data_df
