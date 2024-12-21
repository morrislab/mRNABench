from abc import ABC, abstractmethod
from pathlib import Path
import shutil
import os

import pandas as pd
import yaml

from mrna_bench.utils import download_file


class BenchmarkDataset(ABC):
    """Abstract class for benchmarking datasets.

    Sequences are internally represented as strings. This is less storage
    efficient, but easier to handle as most parts of the pipeline like to
    use raw text.
    """

    def __init__(
        self,
        dataset_name: str,
        short_name: str,
        description: str,
        data_storage_path: str | None = None,
        raw_data_src_url: str | None = None,
        force_redownload: bool = False,
        raw_data_src_path: str | None = None,

    ):
        """Initialize BenchmarkDataset.

        Args:
            dataset_name: Name of the benchmark dataset.
            short_name: Shortened name of benchmark dataset. Should have no
                spaces, use '-' instead.
            description: Description of the dataset.
            data_storage_path: Path where downloaded data is stored.
            raw_data_src_url: URL where raw data can be downloaded.
            force_redownload: Forces raw data redownload.
            raw_data_src_path: Path where raw data is located.
        """
        if raw_data_src_url is None and raw_data_src_path is None:
            raise ValueError("At least one data source must be defined.")
        elif raw_data_src_path is not None and raw_data_src_url is not None:
            raise ValueError("Only one data source must be defined.")

        self.dataset_name = dataset_name
        self.short_name = short_name

        self.raw_data_src_url = raw_data_src_url
        self.raw_data_src_path = raw_data_src_path

        self.description = description
        self.force_redownload = force_redownload
        self.data_storage_path = data_storage_path
        self.first_download = False

        self.init_folders()

        if not self.load_processed_df():
            if self.raw_data_src_url is None:
                self.collect_raw_data()
            else:
                self.download_raw_data()

            self.data_df = self.process_raw_data()
            self.save_processed_df()

    def init_folders(self):
        """Initialize folders for storing raw data.

        Creates a structure with:

        - data_storage_path
        |    - dataset_name
        |    |    - raw_data
        """
        if self.data_storage_path is None:
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            with open(curr_dir + "/../config.yaml") as stream:
                storage_path = yaml.safe_load(stream)["data_storage_path"]
            self.data_storage_path = storage_path

        self.fs_dataset_name = "_".join(self.dataset_name.lower().split(" "))

        ds_path = Path(self.data_storage_path) / self.fs_dataset_name
        ds_path.mkdir(exist_ok=True)

        raw_data_dir = Path(ds_path) / "raw_data"
        raw_data_dir.mkdir(exist_ok=True)

        self.dataset_path = str(ds_path)
        self.raw_data_dir = str(raw_data_dir)

    def download_raw_data(self):
        """Download the raw data from given web source."""
        raw_data_path, is_dled = download_file(
            self.raw_data_src_url,
            self.raw_data_dir,
            self.force_redownload
        )
        self.raw_data_path = raw_data_path
        self.first_download = is_dled

    def collect_raw_data(self):
        """Collect the raw data from given local path."""
        raw_file_name = Path(self.raw_data_src_path).name
        raw_data_path = self.raw_data_dir + "/" + raw_file_name

        if not os.path.exists(raw_data_path):
            shutil.copy(self.raw_data_src_path, raw_data_path)
            self.first_download = True

        self.raw_data_path = raw_data_path

    def save_processed_df(self, df: pd.DataFrame):
        df.to_pickle(self.raw_data_dir + "/data_df.pkl")

    def load_processed_df(self) -> bool:
        try:
            self.data_df = pd.read_pickle(self.raw_data_dir + "/data_df.pkl")
        except FileNotFoundError:
            print("Processed data frame not found.")
            return False
        return True

    @abstractmethod
    def process_raw_data(self) -> pd.DataFrame:
        """Abstract method to process the dataset for the task."""
        pass
