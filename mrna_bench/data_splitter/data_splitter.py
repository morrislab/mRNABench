from abc import ABC, abstractmethod

import pandas as pd


class DataSplitter(ABC):
    """Generates reproducible train test splits."""

    def __init__(
        self,
        default_split_ratio: tuple[float, float, float] = (0.7, 0.15, 0.15),
        **kwargs
    ):
        """Initialize DataSplitter.

        Args:
            default_split_ratio: Ratio of training, validation, test splits.
            **kwargs: Additional arguments.
        """
        self.default_split_ratio = default_split_ratio

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
        split_ratios: tuple[float, float, float] | None = None,
        random_seed: int = 2541,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Get train, validation, test splits from dataframe.

        Args:
            df: Dataframe to split.
            random_seed: Random seed used to generate splits.
            split_ratios: Optional override of split ratios provided during
                initialization. Ratio of training, validation, test split size.

        Returns:
            Dataframe containing training, validation, test splits.
        """
        if split_ratios is None:
            split_ratios = self.default_split_ratio

        if sum(split_ratios) != 1:
            raise ValueError("Split ratios must sum to 1.")

        tv_split_size = split_ratios[1] + split_ratios[2]

        if tv_split_size == 0:
            return df, pd.DataFrame(), pd.DataFrame()

        test_split_size = split_ratios[2] / tv_split_size

        if split_ratios[0] == 0:
            train_df = pd.DataFrame()
            tv_df = df
        else:
            train_df, tv_df = self.split_df(df, tv_split_size, random_seed)

        if split_ratios[1] == 0:
            return train_df, pd.DataFrame(), tv_df
        elif split_ratios[2] == 0:
            return train_df, tv_df, pd.DataFrame()

        val_df, test_df = self.split_df(tv_df, test_split_size, random_seed)

        return train_df, val_df, test_df
