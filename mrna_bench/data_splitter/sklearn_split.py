from sklearn.model_selection import train_test_split

import pandas as pd

from mrna_bench.data_splitter.data_splitter import DataSplitter


class SklearnSplitter(DataSplitter):
    """Wrapper for sklearn train_test_split."""

    def split_df(
        self,
        df: pd.DataFrame,
        test_size: float,
        random_seed: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split dataframe into train and test split using sklearn.

        Args:
            df: Dataframe to split.
            test_size: Fraction of dataset to place in test set.
            random_seed: Random seed used during sampling.

        Return:
            Train and test dataframes.
        """
        return train_test_split(
            df,
            test_size=test_size,
            random_state=random_seed
        )
