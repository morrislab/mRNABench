import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer

from mrna_bench.data_splitter.data_splitter import DataSplitter


class KMerSplitter(DataSplitter):
    """Splits dataset after clustering sequences with similar k-mer counts."""

    def __init__(
        self,
        default_split_ratio: tuple[float, float, float] = (0.7, 0.15, 0.15),
        **kwargs
    ):
        """Initialize splitter with k-mer length.

        Args:
            default_split_ratio: Default ratio of split sizes.
            **kwargs: KMer specific arguments:
                - k: Length of k-mer.
                - n_clusters: Number of clusters for KMeans. If not set,
                    defaults to using 1 cluster per 100 sequences.
        """
        super().__init__(default_split_ratio)

        self.k = kwargs.get("k", 3)
        self.n_clusters = kwargs.get("n_clusters", None)

        self.vectorizer = CountVectorizer(
            analyzer="char",
            ngram_range=(self.k, self.k)
        )

    def vectorize_sequences(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorize sequences using k-mer counts.

        Args:
            df: Dataframe containing sequences.

        Returns:
            Dataframe with k-mer counts.
        """
        kmer_counts = self.vectorizer.fit_transform(df["sequence"]).toarray()
        df["kmer"] = kmer_counts.tolist()

        return df

    def cluster_sequences(
        self,
        df: pd.DataFrame,
        random_state: int
    ) -> pd.DataFrame:
        """Cluster sequences based on k-mer counts.

        Args:
            df: Dataframe containing sequences and k-mer counts.
            random_state: Random seed for KMeans.

        Returns:
            Dataframe with cluster id for each sequence.
        """
        if self.n_clusters:
            n_clusters = self.n_clusters
        else:
            n_clusters = max(2, len(df) // 100)

        self.cluster = KMeans(n_clusters=n_clusters, random_state=random_state)
        df["cluster"] = self.cluster.fit_predict(df["kmer"].tolist())

        return df

    def split_df(
        self,
        df: pd.DataFrame,
        test_size: float,
        random_seed: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split dataframe into train and test split while clustering by kmer.

        Args:
            df: Dataframe to split.
            test_size: Fraction of dataset to place in test set.
            random_seed: Random seed used during sampling.

        Return:
            Train and test dataframes.
        """
        np.random.seed(random_seed)

        print("Running vectorization...")
        df = self.vectorize_sequences(df)
        print("Running cluster...")
        clust_df = self.cluster_sequences(df, random_seed)
        print("Finished.")
        clusters = clust_df["cluster"].to_list()
        cluster_list = list(set(clusters))

        np.random.shuffle(cluster_list)

        curr_test_size = 0
        target_test_size = int(test_size * len(df))

        for cluster_ind in cluster_list:
            cluster_inds = clust_df[clust_df["cluster"] == cluster_ind].index
            cluster_size = len(cluster_inds)

            if curr_test_size < target_test_size:
                clust_df.loc[cluster_inds, "split"] = "test"
            else:
                clust_df.loc[cluster_inds, "split"] = "train"
            curr_test_size += cluster_size

        train_df = clust_df[clust_df["split"] == "train"].copy()
        test_df = clust_df[clust_df["split"] == "test"].copy()

        train_df.drop(columns=["kmer", "cluster", "split"], inplace=True)
        test_df.drop(columns=["kmer", "cluster", "split"], inplace=True)

        return train_df, test_df
