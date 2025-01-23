from abc import ABC, abstractmethod
from pathlib import Path
import shutil
import os

import pandas as pd

from mrna_bench.utils import download_file, get_data_path


class BenchmarkDataset(ABC):
    """Abstract class for benchmarking datasets.

    Sequences are internally represented as strings. This is less storage
    efficient, but easier to handle as most parts of the pipeline like to
    use raw text.

    TODO: The inheritance pattern here is kind of wonky. Probably best to
    make it so each task has a BenchmarkDataset, but inherits from something
    else.
    """

    def __init__(
        self,
        dataset_name: str,
        species: list[str] = ["human"],
        raw_data_src_url: str | None = None,
        force_redownload: bool = False,
        raw_data_src_path: str | None = None,
    ):
        """Initialize BenchmarkDataset.

        Args:
            dataset_name: Name of the benchmark dataset. Should have no
                spaces, use '-' instead.
            species: Species dataset is collected from.
            raw_data_src_url: URL where raw data can be downloaded.
            force_redownload: Forces raw data redownload.
            raw_data_src_path: Path where raw data is located.
        """
        if raw_data_src_url is None and raw_data_src_path is None:
            raise ValueError("At least one data source must be defined.")
        elif raw_data_src_path is not None and raw_data_src_url is not None:
            raise ValueError("Only one data source must be defined.")

        self.dataset_name = dataset_name

        self.raw_data_src_url = raw_data_src_url
        self.raw_data_src_path = raw_data_src_path

        self.species = species

        self.force_redownload = force_redownload

        self.data_storage_path = get_data_path()
        self.init_folders()

        if force_redownload or not self.load_processed_df():
            print("Downloading raw data.")
            if self.raw_data_src_url is None:
                self.collect_raw_data()
            else:
                self.download_raw_data()

            # TODO: This bugs out due to parallelism I think.
            self.data_df = self.process_raw_data()
            self.save_processed_df(self.data_df)

    def init_folders(self):
        """Initialize folders for storing raw data.

        Creates a structure with:

        - data_path
        |    - dataset_name
        |    |    - raw_data
        |    |    - embeddings
        """
        ds_path = Path(self.data_storage_path) / self.dataset_name
        ds_path.mkdir(exist_ok=True)

        raw_data_dir = Path(ds_path) / "raw_data"
        raw_data_dir.mkdir(exist_ok=True)

        emb_dir = Path(ds_path) / "embeddings"
        emb_dir.mkdir(exist_ok=True)

        self.dataset_path = str(ds_path)
        self.raw_data_dir = str(raw_data_dir)
        self.embedding_dir = str(emb_dir)

    def download_raw_data(self):
        """Download the raw data from given web source."""
        raw_data_path, _ = download_file(
            self.raw_data_src_url,
            self.raw_data_dir,
            self.force_redownload
        )
        self.raw_data_path = raw_data_path

    def collect_raw_data(self):
        """Collect the raw data from given local path."""
        raw_file_name = Path(self.raw_data_src_path).name
        raw_data_path = self.raw_data_dir + "/" + raw_file_name

        if not os.path.exists(raw_data_path):
            shutil.copy(self.raw_data_src_path, raw_data_path)

        self.raw_data_path = raw_data_path

    def save_processed_df(self, df: pd.DataFrame):
        """Save dataframe to data storage path.

        Args:
            df: Processed dataframe to save.
        """
        df.to_pickle(self.dataset_path + "/data_df.pkl")

    def load_processed_df(self) -> bool:
        """Load processed dataframe from data storage path.

        Returns:
            Whether dataframe was successfully loaded to class property.
        """
        try:
            self.data_df = pd.read_pickle(self.dataset_path + "/data_df.pkl")
        except FileNotFoundError:
            print("Processed data frame not found.")
            return False
        return True

    @abstractmethod
    def process_raw_data(self) -> pd.DataFrame:
        """Abstract method to process the dataset for the task."""
        pass
