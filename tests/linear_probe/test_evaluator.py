import pytest
from unittest.mock import patch

import numpy as np

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

from mrna_bench.linear_probe.evaluator import (
    LinearProbeEvaluator,
    eval_regression,
    eval_classification,
    eval_multilabel
)


@pytest.fixture
def mock_regression():
    """Mock regression model."""
    return LinearRegression().fit(np.random.rand(2, 10), np.random.rand(2))


@pytest.fixture
def mock_classifier():
    """Mock classifier model."""
    return LogisticRegression().fit(
        np.random.rand(2, 10),
        [0, 1]
    )


@pytest.fixture
def mock_multioutput_classifier():
    """Mock regression model."""
    return MultiOutputClassifier(LogisticRegression()).fit(
        np.random.rand(2, 10),
        [[0, 1], [1, 0]]
    )


def test_eval_regression(mock_regression):
    """Test eval_regression function."""
    X = np.random.rand(2, 10)
    y = np.random.rand(2)
    metrics = eval_regression(mock_regression, X, y)
    assert isinstance(metrics, dict)
    assert "mse" in metrics
    assert "r" in metrics
    assert "p" in metrics


def test_classification(mock_classifier):
    """Test eval_classification function."""
    X = np.random.rand(2, 10)
    y = [0, 1]
    metrics = eval_classification(mock_classifier, X, y)
    assert isinstance(metrics, dict)
    assert "auroc" in metrics
    assert "auprc" in metrics


def test_multilabel(mock_multioutput_classifier):
    """Test eval_multilabel function."""
    X = np.random.rand(2, 10)
    y = [[0, 1], [1, 0]]

    metrics = eval_multilabel(mock_multioutput_classifier, X, y)
    assert isinstance(metrics, dict)
    assert "auroc" in metrics
    assert "auprc" in metrics


def test_linear_probe_evaluator_task_check():
    """Test LinearProbeEvaluator class checks task validity."""
    evaluator = LinearProbeEvaluator("regression")
    assert evaluator.task == "regression"
    with pytest.raises(ValueError):
        LinearProbeEvaluator("invalid_task")


def test_linear_probe_evaluator_validate_input():
    """Test LinearProbeEvaluator split input validity."""
    def mock_dict(keys):
        return {key: np.random.rand(2, 10) for key in keys}

    valid_splits = ["train_X", "train_y", "val_X", "val_y"]
    invalid_splits_missing = ["train_X", "train_y", "val_X"]
    invalid_split_empty = []
    invalid_split_mislabel = ["train_X", "train_labels"]

    LinearProbeEvaluator.validate_input(mock_dict(valid_splits))

    with pytest.raises(ValueError):
        LinearProbeEvaluator.validate_input(mock_dict(invalid_splits_missing))
    with pytest.raises(ValueError):
        LinearProbeEvaluator.validate_input(mock_dict(invalid_split_empty))
    with pytest.raises(ValueError):
        LinearProbeEvaluator.validate_input(mock_dict(invalid_split_mislabel))


def test_linear_probe_evaluator_evaluate(mock_regression):
    """Test LinearProbeEvaluator evaluate method."""
    evaluator = LinearProbeEvaluator("regression")

    with patch(
        "mrna_bench.linear_probe.evaluator.eval_regression",
        side_effect=eval_regression
    ) as reg_mock:
        splits = {
            "train_X": np.random.rand(2, 10),
            "train_y": np.random.rand(2),
            "val_X": np.random.rand(2, 10),
            "val_y": np.random.rand(2)
        }
        metrics = evaluator.evaluate_linear_probe(mock_regression, splits)

        assert reg_mock.call_count == 2
        assert isinstance(metrics, dict)
        assert "train_mse" in metrics
        assert "train_p" in metrics
        assert "val_mse" in metrics
        assert "train_p" in metrics
