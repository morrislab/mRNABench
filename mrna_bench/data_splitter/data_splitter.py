from abc import ABC, abstractmethod

import pandas as pd


class DataSplitter(ABC):
    """Generates reproducible train test splits."""

    @abstractmethod
    def split_df(
        self,
        df: pd.DataFrame,
        test_size: float,
        random_seed: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split dataframe into train and test dataframes.

        Args:
            df: Dataframe to split.
            test_size: Fraction of dataset to assign to test split.
            random_seed: Random seed used during split sampling.

        Returns:
            Train and test dataframes.
        """
        pass

    def get_all_splits_df(
        self,
        df: pd.DataFrame,
        split_ratios: tuple[float, float, float],
        random_seed: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Get train, validation, test splits.

        Args:
            df: Dataframe to split.
            split_ratios: Ratio of training, validation, test split size.
            random_seed: Random seed used to generate splits.

        Returns:
            Dataframe containing training, validation, test splits.
        """
        if sum(split_ratios) != 1:
            raise ValueError("Split ratios must sum to 1.")

        tv_split_size = split_ratios[1] + split_ratios[2]
        test_split_size = split_ratios[2] / tv_split_size

        train_df, tv_df = self.split_df(df, tv_split_size, random_seed)
        val_df, test_df = self.split_df(tv_df, test_split_size, random_seed)

        return train_df, val_df, test_df
