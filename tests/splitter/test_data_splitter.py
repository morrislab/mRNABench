import pytest

import pandas as pd

from mrna_bench.data_splitter.data_splitter import DataSplitter


class MockSplitter(DataSplitter):
    """Mock data splitter which returns dummy dataframes."""

    def split_df(
        self,
        df: pd.DataFrame,
        test_size: float,
        random_seed: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return dummy dataframes."""
        return pd.DataFrame({"mock": [1]}), pd.DataFrame({"mock": [2]})


@pytest.fixture
def mock_splitter() -> MockSplitter:
    """Create MockSplitter."""
    return MockSplitter()


def test_data_splitter_invalid_ratio():
    """Test DataSplitter get_all_splits_df with invalid ratio."""
    with pytest.raises(ValueError):
        mock_splitter = MockSplitter()
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        mock_splitter.get_all_splits_df(
            df,
            split_ratios=(0, 0.5, 0.1)
        )


def test_data_splitter_get_all_splits(mock_splitter: MockSplitter):
    """Test DataSplitter get_all_splits_df."""
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    train_df, val_df, test_df = mock_splitter.get_all_splits_df(
        df,
        split_ratios=(0.8, 0.1, 0.1)
    )

    assert train_df.shape[0] == 1
    assert train_df["mock"].sum() == 1
    assert val_df.shape[0] == 1
    assert val_df["mock"].sum() == 1
    assert test_df.shape[0] == 1
    assert test_df["mock"].sum() == 2


def test_data_splitter_no_val():
    """Test DataSplitter get_all_splits_df with no validation set."""
    mock_splitter = MockSplitter()
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

    train_df, val_df, test_df = mock_splitter.get_all_splits_df(
        df,
        split_ratios=(0.8, 0, 0.2)
    )

    assert train_df.shape[0] == 1
    assert not train_df.empty
    assert val_df.empty
    assert test_df.shape[0] == 1
    assert not test_df.empty


def test_data_splitter_no_test():
    """Test DataSplitter get_all_splits_df with no test set."""
    mock_splitter = MockSplitter()
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

    train_df, val_df, test_df = mock_splitter.get_all_splits_df(
        df,
        split_ratios=(0.8, 0.2, 0)
    )

    assert train_df.shape[0] == 1
    assert not train_df.empty
    assert val_df.shape[0] == 1
    assert not val_df.empty
    assert test_df.empty


def test_data_splitter_only_train():
    """Test DataSplitter get_all_splits_df with only train set.

    Unclear why this would be needed but worth checking this edge case.
    """
    mock_splitter = MockSplitter()
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    train_df, val_df, test_df = mock_splitter.get_all_splits_df(
        df,
        split_ratios=(1, 0, 0)
    )

    assert train_df.shape[0] == 3
    assert not train_df.empty
    assert val_df.empty
    assert test_df.empty


def test_data_splitter_only_test_val():
    """Test DataSplitter get_all_splits_df with no train set."""
    mock_splitter = MockSplitter()
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    train_df, val_df, test_df = mock_splitter.get_all_splits_df(
        df,
        split_ratios=(0, 0.5, 0.5)
    )

    assert train_df.empty
    assert val_df["mock"].sum() == 1
    assert not val_df.empty
    assert test_df["mock"].sum() == 2
    assert not test_df.empty
