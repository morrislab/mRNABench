from unittest.mock import patch
import pytest
import pandas as pd

from mrna_bench.data_splitter.chromosome_split import ChromosomeSplitter
import mrna_bench as mb

def test_chromosome_splitter_basic():
    """Test ChromosomeSplitter basic split."""
    splitter = ChromosomeSplitter()
    df = pd.DataFrame({"chromosome": [
        "8",
        "8",
        "8",
        "8",
        "8",
        "9",
        "9",
        "9",
    ]})

    for test_size in [0.4, 0.3, 0.2, 0.1]:
        train_df, test_df = splitter.split_df(df, test_size, 38)
        assert len(train_df) == 5
        assert len(test_df) == 3
        assert train_df["chromosome"].unique().item() == "8"
        assert test_df["chromosome"].unique().item() == "9"
    
    # Weird but possible
    train_df, test_df = splitter.split_df(df, 0.8, 38)
    assert len(train_df) == 3
    assert len(test_df) == 5
    assert train_df["chromosome"].unique().item() == "9"
    assert test_df["chromosome"].unique().item() == "8"

def test_chromosome_splitter_exact():
    """Test ChromosomeSplitter exact split."""
    splitter = ChromosomeSplitter()
    df = pd.DataFrame({"chromosome": [
        "8",
        "8",
        "8",
        "8",
        "8",
        "9",
        "9",
        "9",
        "X",
        "X",
    ]})

    train_df, test_df = splitter.split_df(df, 0.2, 38)
    assert len(train_df) == 8
    assert len(test_df) == 2
    assert set(train_df["chromosome"]) == {"8", "9"}
    assert test_df["chromosome"].unique().item() == "X"

def test_chromosome_splitter_single_chromosome():
    """Test ChromosomeSplitter with only one chromosome."""
    splitter = ChromosomeSplitter()
    df = pd.DataFrame({"chromosome": ["8"] * 10})

    with pytest.raises(ValueError):
        splitter.split_df(df, 0.3, 38)

def test_chromosome_splitter_tiny_test_size():
    """Test ChromosomeSplitter with very small test_size."""
    splitter = ChromosomeSplitter()
    df = pd.DataFrame({"chromosome": [
        "1", "1", "1", "1", "1",  # 5 entries
        "2", "2", "2", "2", "2",  # 5 entries
    ]})

    # Even though test_size is 0.01 (0.1 entries), should still take smallest chromosome
    train_df, test_df = splitter.split_df(df, 0.01, 38)
    assert len(test_df) == 5
    assert len(train_df) == 5
    assert len(set(train_df["chromosome"])) == 1
    assert len(set(test_df["chromosome"])) == 1

def test_chromosome_splitter_empty_df():
    """Test ChromosomeSplitter with empty dataframe."""
    splitter = ChromosomeSplitter()
    df = pd.DataFrame({"chromosome": []})

    with pytest.raises(ValueError):
        splitter.split_df(df, 0.3, 38)

def test_real_data():
    """Test ChromosomeSplitter with real data."""
    # Smallest dataset
    dataset = mb.load_dataset("go-mf")
    data_df = dataset.data_df

    splitter = ChromosomeSplitter()

    for test_prop in [0.4,0.3,0.2,0.1]: 
        train_df, test_df = splitter.split_df(data_df, test_prop, 32)
        train_df_prop = len(train_df)/(len(train_df) + len(test_df))
        test_df_prop = len(test_df)/(len(train_df) + len(test_df))

        # Check rough proportions are correct
        assert abs(train_df_prop - (1-test_prop)) <= 0.05
        assert abs(test_df_prop - test_prop) <= 0.05

        # Check that chromosomes are unique to each split
        train_df_chroms = set(train_df['chromosome'])
        test_df_chroms = set(test_df['chromosome'])
        assert len(train_df_chroms.intersection(test_df_chroms)) == 0

        # Check that full dataset is split/preserved
        assert len(train_df) + len(test_df) == len(data_df)
