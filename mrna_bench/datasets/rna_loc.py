import json
import shutil
import numpy as np
import pandas as pd 
import os 

from pathlib import Path
from mrna_bench.datasets.benchmark_dataset import BenchmarkDataset


class RNALocalization(BenchmarkDataset):
    """RNA Subcellular Localization Dataset."""

    RL_URL = "/data1/morrisq/dalalt1/Orthrus/processed_data/rna_localization/rna_subcellular_localization.tsv"

    def __init__(self, 
        force_redownload: bool = False,
        **kwargs # noqa
    ):
        """Initialize RNALocalization dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        super().__init__(
            dataset_name="rna-loc",
            species=["human"],
            force_redownload=force_redownload
        )

    def process_raw_data(self) -> pd.DataFrame:
        """Process raw data into Pandas dataframe.

        Returns:
            Pandas dataframe of processed sequences.
        """
        
        df = pd.read_csv(self.raw_data_path, sep="\t")

        # the cds, and splice == strings of lists, convert -> lists -> numpy arrays
        df["cds"] = df["cds"].apply(lambda x: np.array(json.loads(x)))
        df["splice"] = df["splice"].apply(lambda x: np.array(json.loads(x)))
        df["target"] = df["target"].apply(lambda x: np.array(json.loads(x)))

        return df

    def save_processed_df(self, df: pd.DataFrame):
        """Save dataframe to data storage path.

        Args:
            df: Processed dataframe to save.
        """
        df.to_pickle(self.dataset_path + "/data_df.pkl")

        self.data_df = df


    def load_processed_df(self) -> bool:
        """Load processed dataframe from data storage path.

        Returns:
            Whether dataframe was successfully loaded to class property.
        """
        try:
            df = pd.read_pickle(self.dataset_path + "/data_df.pkl")
            
            self.data_df = df

        except FileNotFoundError:
            print("Processed data frame not found.")
            return False
        return True

    def get_raw_data(self):
        """Collect the raw data from given local path."""
        raw_file_name = Path(self.RL_URL).name
        raw_data_path = self.raw_data_dir + "/" + raw_file_name

        if not os.path.exists(raw_data_path):
            shutil.copy(self.RL_URL, raw_data_path)
            self.first_download = True

        self.raw_data_path = raw_data_path
