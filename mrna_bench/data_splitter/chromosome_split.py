import pandas as pd
import numpy as np

from mrna_bench.data_splitter.data_splitter import DataSplitter


class ChromosomeSplitter(DataSplitter):
    """Hold-out chromosome(s) for train-test split."""

    def split_df(
        self,
        df: pd.DataFrame,
        test_size: float,
        random_seed: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split dataframe by holding out chromosome(s).

        Args:
            df: Dataframe to split.
            test_size: Fraction of dataset to place in test set.
            random_seed: Random seed used during sampling.

        Return:
            Train and test dataframes.
        """
        self._validate_input_df(df)

        # Get unique chromosomes and their sizes
        chr_counts = df["chromosome"].value_counts()

        # Randomly shuffle chromosomes
        np.random.seed(random_seed)
        chromosomes = chr_counts.index.tolist()
        np.random.shuffle(chromosomes)

        curr_test_size = 0
        target_test_size = int(test_size * len(df))

        test_chroms, train_chroms = [], []
        for chr in chromosomes:
            # If adding this chromosome won't exceed target test size, add it
            space = target_test_size - curr_test_size
            if curr_test_size < target_test_size and chr_counts[chr] <= space:
                test_chroms.append(chr)
                curr_test_size += chr_counts[chr]
            # Otherwise, add it to train set
            else:
                train_chroms.append(chr)

        # If we couldn't get any chromosomes (test_size too small),
        # take smallest chromosome
        chr_counts_sorted = chr_counts.sort_values(ascending=True)
        if not test_chroms:
            test_chroms = [chr_counts_sorted.index[0]]
            train_chroms.remove(test_chroms[0])

        # Create train/test split
        test_df = df[df["chromosome"].isin(test_chroms)]
        train_df = df[~df["chromosome"].isin(test_chroms)]

        chr_out = "Train chromosomes: {}  Test chromosomes: {}"
        print(chr_out.format(train_chroms, test_chroms))

        # Check if test size is correct
        actual_ratio = len(test_df) / (len(test_df) + len(train_df))
        split_size_out = "Train size: {}  Test size: {}"
        print(split_size_out.format(test_size, actual_ratio))

        return train_df, test_df

    def _validate_input_df(self, df: pd.DataFrame) -> None:
        """Validate input dataframe for chromosome splitting.

        Args:
            df: Input dataframe to validate

        Raises:
            ValueError: If dataframe is empty or contains only one unique
                chromosome.
        """
        if len(df) == 0:
            raise ValueError("Passed in empty dataframe.")

        if len(df["chromosome"].unique()) == 1:
            raise ValueError(
                "Cannot split data with only one unique chromosome - "
                "chromosome-based splitting requires at least 2 "
                "different chromosomes"
            )
