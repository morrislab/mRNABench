from unittest.mock import patch

import pandas as pd

from mrna_bench.data_splitter.kmer_split import KMerSplitter


def test_kmer_splitter_vectorize():
    """Test KMerSplitter vectorization."""
    splitter = KMerSplitter(k=3)
    assert splitter.vectorizer.ngram_range == (3, 3)

    # Test vectorize_sequences
    df = pd.DataFrame({"sequence": ["ACGTAGTA", "ACGTAGAT"]})

    splitter.vectorize_sequences(df)
    assert "kmer" in df.columns
    assert df["kmer"].isna().sum() == 0
    assert (df["kmer"].apply(type) == list).all()


def test_kmer_splitter_cluster():
    """Test KMerSplitter clustering."""
    splitter = KMerSplitter(k=3)

    # Test cluster_sequences
    df = pd.DataFrame({"sequence": [
        "GGGGGGGA",
        "GGGGGGGC",
        "GGGGGGGT",
        "GGGGGGGG",
        "AAAAAAAA",
        "AAAAAAAT",
        "AAAAAAAC",
        "AAAAAAAG",
    ]})

    vectorized_df = splitter.vectorize_sequences(df)
    clustered_df = splitter.cluster_sequences(vectorized_df, 2541)

    assert "cluster" in clustered_df.columns
    assert clustered_df["cluster"].isna().sum() == 0
    assert (clustered_df["cluster"].apply(type) == int).all()


def test_kmer_splitter_split():
    """Test KMerSplitter split."""
    test_ratio = 0.3

    splitter = KMerSplitter(k=3, n_cluster=2)

    df = pd.DataFrame({"sequence": [
        "GGGGGGGA",
        "GGGGGGGC",
        "GGGGGGGT",
        "GGGGGGGG",
        "AAAAAAAA",
        "AAAAAAAT",
        "AAAAAAAC",
        "AAAAAAAG",
    ]})

    with patch.object(splitter, "cluster_sequences") as mock:
        mock.return_value = pd.DataFrame({
            "cluster": [0, 0, 0, 0, 1, 1, 1, 1],
            "kmer": [1, 2, 3, 4, 5, 6, 7, 8],
            "split": ["tr", "tr", "tr", "tr", "te", "te", "te", "te"]
        })

        train_df, test_df = splitter.split_df(
            df,
            test_size=test_ratio,
            random_seed=2547
        )

        assert len(train_df) + len(test_df) == len(df)
        assert len(train_df) > 0
        assert len(test_df) > 0
        assert "cluster" not in train_df.columns
        assert "cluster" not in test_df.columns
        assert "kmer" not in train_df.columns
        assert "kmer" not in test_df.columns
        assert "split" not in train_df.columns
        assert "split" not in test_df.columns
