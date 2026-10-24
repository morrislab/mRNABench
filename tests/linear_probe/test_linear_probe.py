import pytest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Ridge,
    RidgeCV,
)
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

from mrna_bench.linear_probe.linear_probe import LinearProbe, to_torch


def test_regression_model_names_are_distinct():
    """Regressor names map to OLS and standardized RidgeCV."""
    assert isinstance(
        LinearProbe.regression_models["ols"](),
        LinearRegression,
    )
    assert isinstance(
        LinearProbe.regression_models["ridge"]()[-1],
        RidgeCV,
    )


def test_probe_config_names_the_estimator():
    """Ridge and OLS probes persist under different probe configs."""
    ridge = LinearProbe.get_probe_config("regression", "ridge")
    ols = LinearProbe.get_probe_config("regression", "ols")

    assert ridge != ols
    assert "RidgeCV" in ridge["estimator"]
    assert "LinearRegression" in ols["estimator"]


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
    evaluator.evaluate_linear_probe = Mock(return_value={"val_r": 0.1})

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
def linear_probe_persister(linear_probe: LinearProbe, tmp_path) -> LinearProbe:
    """Return a LinearProbe instance with a persister."""
    persister = Mock()
    persister.persist_run_results = Mock()
    persister.get_checkpoint_path = Mock(return_value=tmp_path / "probe.pt")
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
        checkpoint = linear_probe_persister.persister.get_checkpoint_path(42)
        linear_probe_persister.run_linear_probe(random_seed=42, persist=True)

        linear_probe_persister.persister.persist_run_results.assert_called()
        assert not checkpoint.exists()
        linear_probe_persister.run_linear_probe(
            random_seed=42, persist=True, save_checkpoint=True
        )
        assert checkpoint.exists()
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


@pytest.mark.parametrize("model, y", [
    (LinearRegression(), np.arange(20.0)),
    (RidgeCV(alphas=[1e-3, 1]), np.arange(40.0).reshape(20, 2)),
    (Ridge(solver="sag", alpha=1e-3), np.arange(20.0)),
    (LogisticRegression(), np.arange(20) % 2),
    (LogisticRegression(), np.arange(20) % 3),
    (MultiOutputClassifier(LogisticRegression()),
     np.stack([np.arange(20) % 2, np.arange(20) // 10], axis=1)),
    (LinearProbe.regression_models["ridge"](), np.arange(20.0)),
    (LinearProbe.linear_models["classification"](), np.arange(20) % 2),
    (LinearProbe.linear_models["multilabel"](),
     np.stack([np.arange(20) % 2, np.arange(20) // 10], axis=1)),
])
def test_torch_checkpoint_matches_sklearn(model, y, tmp_path):
    """The saved torch module reproduces the fitted sklearn model."""
    X = np.random.default_rng(0).normal(size=(20, 5))
    model.fit(X, y)
    torch.save(to_torch(model), tmp_path / "probe.pt")

    module = LinearProbe.load_checkpoint(tmp_path / "probe.pt")
    out = module(torch.tensor(X, dtype=torch.float32)).detach().numpy()

    final = model[-1] if isinstance(model, Pipeline) else model
    if isinstance(final, MultiOutputClassifier):
        expected = np.stack(
            [p[:, 1] for p in model.predict_proba(X)], axis=1
        )
    elif isinstance(final, LogisticRegression):
        expected = model.predict_proba(X)
        if expected.shape[1] == 2:
            expected = expected[:, 1:]
    else:
        expected = model.predict(X).reshape(len(X), -1)
    np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-5)


def test_probe_config_records_view_and_splitter():
    """Feature views and splitter settings give distinct probe configs."""
    mean = LinearProbe.get_probe_config("regression", feature_view="mean")
    max_ = LinearProbe.get_probe_config("regression", feature_view="max")
    homology = LinearProbe.get_probe_config(
        "regression",
        feature_view="mean",
        splitter_config={"similarity_threshold": 20.0},
    )

    assert mean["features"] == "mean"
    assert len({str(mean), str(max_), str(homology)}) == 3
    assert "splitter" not in LinearProbe.get_probe_config(
        "regression", splitter_config={}
    )
