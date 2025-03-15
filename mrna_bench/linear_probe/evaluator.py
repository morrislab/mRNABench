import numpy as np
from scipy.stats import pearsonr, spearmanr

from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.multioutput import MultiOutputClassifier


def eval_regression(
    model: RegressorMixin,
    X: np.ndarray,
    y: np.ndarray
) -> dict[str, float]:
    """Perform linear probing on regression task.

    Args:
        model: Scikit-learn regression model.
        X: Data to be probed.
        y: Ground truth labels of data.

    Returns:
        Dictionary of regression metrics from linear probe.
    """
    y_pred = model.predict(X)

    metrics = {
        "mse": np.mean((y_pred - y) ** 2),
        "r": pearsonr(y_pred, y).statistic,
        "p": spearmanr(y_pred, y).statistic
    }

    return metrics


def eval_classification(
    model: ClassifierMixin,
    X: np.ndarray,
    y: np.ndarray
) -> dict[str, float]:
    """Perform linear probing on classification task.

    Args:
        model: Scikit-learn classification model.
        X: Data to be probed.
        y: Ground truth labels of data.

    Returns:
        Dictionary of classification metrics from linear probe.
    """
    y_pred = model.predict_proba(X)[:, 1]

    metrics = {
        "auroc": roc_auc_score(y, y_pred),
        "auprc": average_precision_score(y, y_pred)
    }
    return metrics


def eval_multilabel(
    model: MultiOutputClassifier,
    X: np.ndarray,
    y: np.ndarray,
    average: str = "micro"
) -> dict[str, float]:
    """Perform linear probing on multilabel classification task.

    Args:
        model: Scikit-learn classification model supporting multi-output.
        X: Data to be probed.
        y: Ground truth labels of data.
        average: Type of averaging to use for multilabel metrics.

    Returns:
        Dictionary of multilabel metrics from linear probe.
    """
    y_pred = model.predict_proba(X)
    y_pred = np.swapaxes(np.array(y_pred), 0, 1)[:, :, 1]

    metrics = {
        "auroc": roc_auc_score(y, y_pred, average=average),
        "auprc": average_precision_score(y, y_pred, average=average)
    }

    return metrics


class LinearProbeEvaluator:
    """Evaluator for linear probing tasks."""

    valid_tasks = [
        "regression",
        "classification",
        "multilabel",
        "reg_lin",
        "reg_ridge"
    ]

    def __init__(self, task: str):
        """Initialize linear probing evaluator.

        Args:
            eval_all_splits: Whether to evaluate on all splits.
        """
        if task not in self.valid_tasks:
            raise ValueError("Invalid task: {}".format(task))

        self.task = task

    @staticmethod
    def validate_input(splits: dict[str, np.ndarray]):
        """Validate data splits for linear probing input.

        For each data split, the splits dictionary must contain the data and
        labels, denoted as "{split_name}_X" and "{split_name}_y" respectively
        for each evaluated split.

        Args:
            splits: Data splits of embeddings to be validated.

        Raises:
            ValueError: If splits are not named correctly or are empty.
        """
        if len(splits) == 0:
            raise ValueError("No splits provided.")

        split_names = set([k[:-2] for k in splits.keys()])

        for split_name in split_names:
            split_data_name = split_name + "_X"
            split_label_name = split_name + "_y"
            if split_data_name not in splits or split_label_name not in splits:
                raise ValueError("Both _X and _y splits must be provided.")

    def evaluate_linear_probe(
        self,
        model: RegressorMixin | ClassifierMixin | MultiOutputClassifier,
        data_splits: dict[str, np.ndarray]
    ) -> dict[str, float]:
        """Evaluate linear probing model on splits.

        Args:
            model: Scikit-learn model to evaluate.
            data_splits: Data splits of embeddings to be probed. Must contain
                keys "{split_name}_X" and "{split_name}_y" for each evaluated
                split. Dict entries are name of split and corresponding data /
                labels.

        Returns:
            Dictionary of linear probing metrics per split.

        Raises:
            ValueError: If splits are not named correctly or are empty.
        """
        self.validate_input(data_splits)

        split_names = set([k[:-2] for k in data_splits.keys()])

        metrics = {}
        for split_name in split_names:
            split_X = data_splits[split_name + "_X"]
            split_y = data_splits[split_name + "_y"]

            if self.task in ["regression", "reg_lin", "reg_ridge"]:
                split_metrics = eval_regression(model, split_X, split_y)
            elif self.task == "classification":
                split_metrics = eval_classification(model, split_X, split_y)
            elif self.task == "multilabel":
                split_metrics = eval_multilabel(model, split_X, split_y)
            else:
                raise ValueError("Invalid task: {}".format(self.task))

            for metric_name, metric_val in split_metrics.items():
                metrics[split_name + "_" + metric_name] = metric_val

        return metrics
