import torch
import torch.nn as nn


class TaskHead(nn.Module):
    """Task-specific prediction head for fine-tuning.

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

        Args:
            x: Input embedding tensor of shape (batch, input_dim).

        Returns:
            Output tensor of shape (batch, output_dim).
        """
        return self.mlp(x)

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

    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to predictions.

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
