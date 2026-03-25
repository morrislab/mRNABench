from typing import Protocol, runtime_checkable

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score


@runtime_checkable
class TaskHeadProtocol(Protocol):
    """Interface that any fine-tuning head must implement.

    Custom heads (CNN, attention, etc.) should satisfy this protocol
    so the trainer can obtain the loss function and task type.
    """

    task_type: str

    def get_loss_fn(self) -> nn.Module:
        """Return appropriate loss function for the task."""
        ...

    def prepare_targets(self, targets: np.ndarray) -> torch.Tensor:
        """Convert numpy targets to correctly-typed tensor."""
        ...

    def score(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to probability scores for metric computation."""
        ...

    def compute_metrics(
        self, logits: torch.Tensor, targets: torch.Tensor,
    ) -> dict[str, float]:
        """Compute task-appropriate metrics from logits and targets."""
        ...


class TaskHead(nn.Module):
    """MLP prediction head for fine-tuning.

    Supports regression, classification, and multilabel tasks with
    optional hidden layers.
    """

    VALID_TASKS = ("regression", "classification", "multilabel")

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        task_type: str,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ):
        """Initialize TaskHead.

        Args:
            input_dim: Embedding dimension from backbone model.
            output_dim: Number of output units (classes or targets).
            task_type: One of regression, classification, multilabel.
            hidden_dims: Hidden layer dimensions. None for linear head.
            dropout: Dropout probability between layers.
        """
        super().__init__()

        if task_type not in self.VALID_TASKS:
            raise ValueError(
                "task_type must be one of {}".format(self.VALID_TASKS)
            )

        self.task_type = task_type
        self.input_dim = input_dim
        self.output_dim = output_dim

        hidden_dims = hidden_dims or []
        layers = []
        prev_dim = input_dim

        for hdim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hdim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hdim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through head.

        For regression, squeezes the last dimension so the output shape
        matches the target shape (batch,) rather than (batch, 1).

        Args:
            x: Input embedding tensor of shape (batch, input_dim).

        Returns:
            Output tensor.
        """
        out = self.mlp(x)
        if self.task_type == "regression":
            out = out.squeeze(-1)
        return out

    def get_loss_fn(self) -> nn.Module:
        """Return appropriate loss function for task type.

        Returns:
            Loss function module.
        """
        if self.task_type == "regression":
            return nn.MSELoss()
        elif self.task_type == "classification":
            return nn.CrossEntropyLoss()
        elif self.task_type == "multilabel":
            return nn.BCEWithLogitsLoss()
        else:
            raise ValueError("Unknown task type: {}".format(self.task_type))

    def prepare_targets(self, targets: np.ndarray) -> torch.Tensor:
        """Convert numpy targets to tensor for loss computation.

        Args:
            targets: Raw numpy targets from dataloader.

        Returns:
            Tensor with correct dtype for the task's loss function.
        """
        if self.task_type == "classification":
            return torch.from_numpy(targets).long()
        return torch.from_numpy(targets).float()

    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to discrete predictions.

        Args:
            logits: Raw output from forward pass.

        Returns:
            Predictions appropriate for task type.
        """
        if self.task_type == "regression":
            return logits
        elif self.task_type == "classification":
            return torch.argmax(logits, dim=-1)
        elif self.task_type == "multilabel":
            return torch.sigmoid(logits)
        else:
            raise ValueError("Unknown task type: {}".format(self.task_type))

    def score(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to probability scores for metric computation.

        Args:
            logits: Raw output from forward pass.

        Returns:
            Scores suitable for sklearn metrics (e.g. roc_auc_score).
        """
        if self.task_type == "regression":
            return logits
        elif self.task_type == "classification":
            return torch.softmax(logits, dim=-1)
        elif self.task_type == "multilabel":
            return torch.sigmoid(logits)
        else:
            raise ValueError("Unknown task type: {}".format(self.task_type))

    def compute_metrics(
        self, logits: torch.Tensor, targets: torch.Tensor,
    ) -> dict[str, float]:
        """Compute task-appropriate evaluation metrics.

        Args:
            logits: Raw model output.
            targets: Ground truth targets (prepared via prepare_targets).

        Returns:
            Dictionary of metric name to value.
        """
        if self.task_type == "regression":
            preds_np = logits.numpy().flatten()
            targets_np = targets.numpy().flatten()
            return {
                "pearson_r": float(pearsonr(preds_np, targets_np)[0]),
                "spearman_r": float(spearmanr(preds_np, targets_np)[0]),
            }

        scores_np = self.score(logits).numpy()
        targets_np = targets.numpy()
        try:
            if self.task_type == "multilabel":
                auroc = float(roc_auc_score(
                    targets_np, scores_np, average="micro",
                ))
            else:
                auroc = float(roc_auc_score(targets_np, scores_np))
        except ValueError as e:
            if "Only one class" in str(e):
                auroc = float("nan")
            else:
                raise
        return {"auroc": auroc}
