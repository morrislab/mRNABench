from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from mrna_bench.utils import get_data_path


class BenchmarkDataset(ABC):
    """Abstract class for benchmarking datasets.

    Sequences are internally represented as strings. This is less storage
    efficient, but easier to handle as most parts of the pipeline use raw text.
    """

    def __init__(
        self,
        dataset_name: str,
        species: list[str] = ["human"],
        force_redownload: bool = False,
    ):
        """Initialize BenchmarkDataset.

        Args:
            dataset_name: Name of the benchmark dataset. Should have no
                spaces, use '-' instead.
            species: Species dataset is collected from.
            force_redownload: Forces raw data redownload.
        """
        self.dataset_name = dataset_name
        self.species = species

        self.force_redownload = force_redownload

        self.data_storage_path = get_data_path()
        self.init_folders()

        if force_redownload or not self.load_processed_df():
            self.get_raw_data()
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
    def get_raw_data(self):
        """Abstract method to get the raw data for the task."""
        pass

    @abstractmethod
    def process_raw_data(self) -> pd.DataFrame:
        """Abstract method to process the dataset for the task."""
        pass

    def get_splits(
        self,
        split_ratios: tuple[float, float, float],
        random_seed: int = 2541,
        split_type: str = "homology",
        split_kwargs: dict = {}
    ) -> dict[str, pd.DataFrame]:
        """Get data splits for the dataset.

        Args:
            split_ratios: Ratios for train, val, test splits.
            random_seed: Random seed for reproducibility.
            split_type: Type of split to use.
            split_kwargs: Additional arguments for the split type.

        Returns:
            Dictionary of dataframes containing splits.
        """
        from mrna_bench.data_splitter.split_catalog import SPLIT_CATALOG

        if split_type == "homology" and split_kwargs == {}:
            split_kwargs["species"] = self.species

        splitter = SPLIT_CATALOG[split_type](**split_kwargs)
        splits = splitter.get_all_splits_df(
            self.data_df,
            split_ratios,
            random_seed
        )

        split_df = {
            "train_df": splits[0],
            "val_df": splits[1],
            "test_df": splits[2]
        }

        return split_df
