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
from mrna_bench.metrics import classification_metrics, multilabel_metrics


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
    assert "mcc" in metrics
    assert "balanced_accuracy" in metrics


def test_multiclass_classification_metrics():
    """Test multiclass classification reports macro/micro metrics."""
    X = np.eye(6)
    y = np.array([0, 0, 1, 1, 2, 2])
    model = LogisticRegression().fit(X, y)

    metrics = eval_classification(model, X, y)

    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert "auroc_macro" in metrics
    assert "auroc_micro" in metrics
    assert "auprc_macro" in metrics
    assert "auprc_micro" in metrics


def test_linear_probe_metrics_mark_missing_classes_nan():
    """Probe evaluators mark undefined held-out metrics as NaN."""
    X = np.eye(6)
    model = LogisticRegression().fit(
        X,
        np.array([0, 0, 1, 1, 2, 2]),
    )

    classification = eval_classification(
        model,
        X[:4],
        np.array([0, 0, 1, 1]),
    )
    assert np.isnan(classification["auroc_macro"])
    assert np.isnan(classification["auprc_macro"])

    multilabel_model = MultiOutputClassifier(LogisticRegression()).fit(
        X[:4],
        np.array([[0, 0], [0, 0], [1, 1], [1, 1]]),
    )
    multilabel = eval_multilabel(
        multilabel_model,
        X[:2],
        np.zeros((2, 2), dtype=int),
    )
    assert np.isnan(multilabel["auroc_macro"])
    assert np.isnan(multilabel["auprc_macro"])


def test_classification_metrics_handle_missing_and_nonstandard_classes():
    """Missing classes return NaN and binary labels need not be zero/one."""
    binary = classification_metrics(
        np.array([2, 3]),
        np.array([[0.9, 0.1], [0.1, 0.9]]),
        np.array([2, 3]),
    )
    assert binary["auprc"] == 1.0

    missing = classification_metrics(
        np.array([0, 0]),
        np.array([[0.9, 0.1], [0.8, 0.2]]),
        np.array([0, 1]),
        missing_class_nan=True,
    )
    assert np.isnan(missing["auroc"])
    assert np.isnan(missing["auprc"])

    multiclass = classification_metrics(
        np.array([0, 1]),
        np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]]),
        np.array([0, 1, 2]),
        missing_class_nan=True,
    )
    assert np.isnan(multiclass["auroc_macro"])
    assert np.isnan(multiclass["auprc_macro"])
    assert not np.isnan(multiclass["auroc_micro"])


def test_multilabel_metrics_mark_only_undefined_averages_nan():
    """A missing label class invalidates macro but not micro metrics."""
    metrics = multilabel_metrics(
        np.array([[0, 0], [1, 0]]),
        np.array([[0.1, 0.2], [0.9, 0.3]]),
        missing_class_nan=True,
    )

    assert np.isnan(metrics["auroc_macro"])
    assert np.isnan(metrics["auprc_macro"])
    assert not np.isnan(metrics["auroc_micro"])
    assert not np.isnan(metrics["auprc_micro"])


def test_multilabel(mock_multioutput_classifier):
    """Test eval_multilabel function."""
    X = np.random.rand(2, 10)
    y = [[0, 1], [1, 0]]

    metrics = eval_multilabel(mock_multioutput_classifier, X, y)
    assert isinstance(metrics, dict)
    assert "auroc_micro" in metrics
    assert "auroc_macro" in metrics
    assert "auprc_micro" in metrics
    assert "auprc_macro" in metrics
    assert "mcc_micro" in metrics


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
        assert "val_p" in metrics
