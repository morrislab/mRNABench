import numpy as np

from mrna_bench.metrics import ranking_metrics, regression_metrics


class ZeroShotEvaluator:
    """Evaluator for zero-shot scores."""

    valid_tasks = [
        "classification",
        "regression",
    ]

    def __init__(self, task: str):
        """Initialize zero-shot evaluator.

        Args:
            task: Classification of labels or regression of values.
        """
        if task not in self.valid_tasks:
            raise ValueError("Invalid zero-shot task: {}".format(task))

        self.task = task

    def evaluate_zero_shot(
        self,
        labels: np.ndarray,
        scores: np.ndarray,
    ) -> dict[str, float]:
        """Evaluate scores against labels.

        Args:
            labels: Labels or target values.
            scores: Scores, higher for the positive class or a larger value.

        Returns:
            Ranking metrics for classification, or regression metrics.
        """
        if self.task == "regression":
            return regression_metrics(labels, scores)
        else:
            return ranking_metrics(labels, scores)
