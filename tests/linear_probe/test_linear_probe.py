import pytest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV

from mrna_bench.linear_probe.linear_probe import LinearProbe
from mrna_bench.linear_probe.persister import LinearProbePersister


def test_regression_model_names_are_distinct():
    """Use OLS for regression and cross-validated Ridge when requested."""
    assert isinstance(
        LinearProbe.regression_models["ols"](),
        LinearRegression,
    )
    assert isinstance(
        LinearProbe.regression_models["ridge"](),
        RidgeCV,
    )


def test_regression_results_use_explicit_persistence_identity(tmp_path):
    """New OLS results do not overwrite historical RidgeCV rows."""
    dataset = Mock(dataset_path=str(tmp_path))
    persister = LinearProbePersister(
        dataset,
        "model",
        "regression",
        "target",
        "default",
    )

    assert persister.task == "regression_ols"

    ridge = LinearProbePersister(
        dataset,
        "model",
        "regression",
        "target",
        "default",
        regressor="ridge",
    )
    assert ridge.task == "regression_ridge"


@pytest.fixture
def linear_probe() -> LinearProbe:
    """Return a LinearProbe instance."""
    data_df = pd.DataFrame({"target": [0, 1, 2, 3, 4, 5]})
    embeddings = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
        [13, 14, 15],
        [16, 17, 18]
    ])
    target_col = "target"
    task = "regression"
    splitter = Mock()
    evaluator = Mock()
    evaluator.evaluate_linear_probe = Mock()

    eval_all_splits = True

    return LinearProbe(
        data_df=data_df,
        embeddings=embeddings,
        target_col=target_col,
        task=task,
        splitter=splitter,
        evaluator=evaluator,
        eval_all_splits=eval_all_splits
    )


@pytest.fixture
def linear_probe_persister(linear_probe: LinearProbe) -> LinearProbe:
    """Return a LinearProbe instance with a persister."""
    persister = Mock()
    persister.persist_run_results = Mock()
    linear_probe.persister = persister
    return linear_probe


@pytest.fixture
def split_dataframes():
    """Return train, validation, and test dataframes."""
    return (
        pd.DataFrame({
            "embeddings": [[1, 2, 3], [4, 5, 6]],
            "target": [0, 1],
        }),
        pd.DataFrame({
            "embeddings": [[7, 8, 9], [10, 11, 12]],
            "target": [2, 3],
        }),
        pd.DataFrame({
            "embeddings": [[13, 14, 15], [16, 17, 18]],
            "target": [4, 5],
        }),
    )


def test_get_df_splits_reg_class(
    linear_probe: LinearProbe,
    split_dataframes,
):
    """Test get_df_splits method."""
    random_seed = 42
    dropna = True

    with patch.object(linear_probe, "splitter") as mock_splitter:
        mock_splitter.get_all_splits_df.return_value = split_dataframes

        splits = linear_probe.get_df_splits(random_seed, dropna)

        assert "train_X" in splits
        assert "val_X" in splits
        assert "test_X" in splits
        assert "train_y" in splits
        assert "val_y" in splits
        assert "test_y" in splits


def test_linear_probe(linear_probe: LinearProbe, split_dataframes):
    """Test run_linear_probe method."""
    with patch.object(linear_probe, "splitter") as mock_splitter:
        mock_splitter.get_all_splits_df.return_value = split_dataframes

        linear_probe.eval_all_splits = False
        linear_probe.run_linear_probe(random_seed=42)

        assert len(linear_probe.models) == 1

        linear_probe.evaluator.evaluate_linear_probe.assert_called_once()
        c_args, _ = linear_probe.evaluator.evaluate_linear_probe.call_args

        assert c_args[0] == linear_probe.models[42]
        assert sorted(list(c_args[1].keys())) == ["val_X", "val_y"]


def test_linear_probe_persister(
    linear_probe_persister: LinearProbe,
    split_dataframes,
):
    """Test run_linear_probe method with persister."""
    with patch.object(linear_probe_persister, "splitter") as mock_splitter:
        mock_splitter.get_all_splits_df.return_value = split_dataframes

        linear_probe_persister.eval_all_splits = False
        linear_probe_persister.run_linear_probe(random_seed=42, persist=True)

        linear_probe_persister.persister.persist_run_results.assert_called()
        linear_probe_persister.persister.persist_run_results.reset_mock()
        linear_probe_persister.run_linear_probe(random_seed=42, persist=False)
        linear_probe_persister.persister.persist_run_results.assert_not_called()


def test_linear_probe_no_persister_persist(
    linear_probe: LinearProbe,
    split_dataframes,
):
    """Test run_linear_probe method with persist flag but no persister."""
    with patch.object(linear_probe, "splitter") as mock_splitter:
        mock_splitter.get_all_splits_df.return_value = split_dataframes

        linear_probe.eval_all_splits = False

        with pytest.raises(RuntimeError):
            linear_probe.run_linear_probe(random_seed=42, persist=True)
