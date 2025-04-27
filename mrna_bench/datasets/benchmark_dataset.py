from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from mrna_bench.utils import download_file, get_data_path


class BenchmarkDataset(ABC):
    """Abstract class for benchmarking datasets.

    Sequences are internally represented as strings. This is less storage
    efficient, but easier to handle as most parts of the pipeline use raw text.
    """

    def __init__(
        self,
        dataset_name: str,
        species: str = "human",
        force_redownload: bool = False,
        hf_url: str | None = None
    ):
        """Initialize BenchmarkDataset.

        Args:
            dataset_name: Name of the benchmark dataset. Should have no
                spaces, use '-' instead.
            species: Species the dataset is collected from.
            force_redownload: Forces raw data redownload.
            hf_url: URL to download processed data from HuggingFace Hub.
        """
        self.dataset_name = dataset_name
        self.species = species

        self.force_redownload = force_redownload
        self.hf_url = hf_url

        self.data_storage_path = get_data_path()
        self.init_folders()

        if force_redownload or not self.load_processed_df():
            if hf_url is not None:
                print("Downloading data from HuggingFace Hub...")
                try:
                    self.data_df = self.get_data_hf()
                except Exception as e:
                    print(f"Error downloading from HuggingFace: {e}")
                    print("Attempting to process from raw data.")
                    self.data_df = self._get_data_from_raw()
            else:
                print("No HF URL provided. Processing from raw.")
                self.data_df = self._get_data_from_raw()

            self.save_processed_df(self.data_df)

    def get_data_hf(self) -> pd.DataFrame:
        """Download processed data from HuggingFace.

        Returns:
            pd.DataFrame: Processed dataframe.
        """
        if self.hf_url is None:
            raise ValueError("HuggingFace URL not provided.")

        processed_data_path = download_file(self.hf_url, self.raw_data_dir)
        return pd.read_parquet(processed_data_path)

    @abstractmethod
    def _get_data_from_raw(self) -> pd.DataFrame:
        """Abstract method to download and process data from raw source.

        This class can be implemented in the subclass with a passthrough if
        a huggingface url is instead provided.
        """
        pass

    def subset_df(self, target_cols: list[str]) -> pd.DataFrame:
        """Subset dataframe target columns.

        Args:
            df: Dataframe to subset.

        Returns:
            Subsetted dataframe.
        """
        if self.data_df is None:
            raise RuntimeError("Dataframe not loaded.")
        if not isinstance(self.data_df, pd.DataFrame):
            raise TypeError("Dataframe is not a pandas DataFrame.")

        # Set invariant columns
        invariant_col_set = set([
            "sequence",
            "gene",
            "chromosome",
            "cds",
            "splice"
        ])

        keep_col_set = invariant_col_set.union(set(target_cols))
        keep_cols = [c for c in self.data_df.columns if c in keep_col_set]
        return self.data_df[keep_cols]

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
        df.to_parquet(self.dataset_path + "/data_df.parquet")

    def load_processed_df(self) -> bool:
        """Load processed dataframe from data storage path.

        Returns:
            Whether dataframe was successfully loaded to class property.
        """
        try:
            df_path = self.dataset_path + "/data_df.parquet"
            self.data_df = pd.read_parquet(df_path)
        except FileNotFoundError:
            print("Processed data frame not found.")
            return False
        return True

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
