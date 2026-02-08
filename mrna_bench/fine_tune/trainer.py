from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class TrainerConfig:
    """Configuration for fine-tuning trainer."""

    learning_rate: float = 1e-4
    epochs: int = 10
    warmup_steps: int = 100
    early_stopping_patience: int = 3
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0


class FineTuneTrainer:
    """Trainer for fine-tuning nucleotide foundation models."""

    def __init__(self, model, config: TrainerConfig | None = None):
        """Initialize FineTuneTrainer.

        Args:
            model: Fine-tunable model with attached head.
            config: Training configuration. Uses defaults if not provided.
        """
        if model._task_head is None:
            raise ValueError("Model must have task head attached.")

        self.model = model
        self.device = model.device
        self.config = config or TrainerConfig()
        self.loss_fn = model._task_head.get_loss_fn()
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
        }

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create AdamW optimizer for trainable parameters."""
        params = self.model.get_trainable_parameters()
        return torch.optim.AdamW(
            params,
            lr=self.config.learning_rate,
            weight_decay=0,
        )

    def _create_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """Create linear warmup scheduler (constant after warmup)."""
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=self.config.warmup_steps,
        )

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch.

        Args:
            dataloader: Training data loader.

        Returns:
            Average training loss.
        """
        self.model.set_train_mode()
        total_loss = 0.0
        num_batches = 0

        progress = tqdm(dataloader, desc="Training")
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(progress):
            target = batch["target"]
            if isinstance(target, np.ndarray):
                target = torch.from_numpy(target)
            target = target.to(self.device).float()

            sequences = batch["sequence"]
            cds = batch.get("cds")
            splice = batch.get("splice")

            output = self.model.forward_with_head(sequences, cds, splice)
            # Only squeeze last dim for regression (output_dim=1)
            # Multilabel needs [batch, num_classes] shape preserved
            if output.shape[-1] == 1:
                output = output.squeeze(-1)
            loss = self.loss_fn(output, target)

            batch_loss = loss.item()
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.get_trainable_parameters(),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()

                self.optimizer.zero_grad()

            total_loss += batch_loss
            num_batches += 1
            progress.set_postfix({"loss": batch_loss})

        return total_loss / num_batches

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        """Evaluate model on validation data.

        Args:
            dataloader: Validation data loader.

        Returns:
            Dictionary of evaluation metrics.
        """
        self.model.set_inference_mode()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        for batch in tqdm(dataloader, desc="Evaluating"):
            target = batch["target"]
            if isinstance(target, np.ndarray):
                target = torch.from_numpy(target)
            target = target.to(self.device).float()

            sequences = batch["sequence"]
            cds = batch.get("cds")
            splice = batch.get("splice")

            output = self.model.forward_with_head(sequences, cds, splice)
            # Only squeeze last dim for regression (output_dim=1)
            # Multilabel needs [batch, num_classes] shape preserved
            if output.shape[-1] == 1:
                output = output.squeeze(-1)
            loss = self.loss_fn(output, target)
            total_loss += loss.item() * len(sequences)

            all_preds.append(output.cpu())
            all_targets.append(target.cpu())

        avg_loss = total_loss / len(dataloader.dataset)
        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)

        metrics = {"loss": avg_loss}

        task_type = self.model._task_head.task_type
        if task_type == "regression":
            from scipy.stats import pearsonr, spearmanr
            preds_np = preds.numpy().flatten()
            targets_np = targets.numpy().flatten()
            metrics["pearson_r"] = float(pearsonr(preds_np, targets_np)[0])
            metrics["spearman_r"] = float(spearmanr(preds_np, targets_np)[0])
        elif task_type in ("classification", "multilabel"):
            from sklearn.metrics import roc_auc_score
            preds_np = torch.sigmoid(preds).numpy()
            targets_np = targets.numpy()
            try:
                if task_type == "multilabel":
                    metrics["auroc"] = float(roc_auc_score(
                        targets_np, preds_np, average="micro"
                    ))
                else:
                    metrics["auroc"] = float(roc_auc_score(targets_np, preds_np))
            except ValueError:
                metrics["auroc"] = float("nan")

        return metrics

    def fit(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None = None,
    ) -> dict[str, list[float]]:
        """Full training loop.

        Args:
            train_dataloader: Training data loader.
            val_dataloader: Validation data loader (optional).

        Returns:
            Training history dictionary.
        """
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler(self.optimizer)

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config.epochs):
            print("Epoch {}/{}".format(epoch + 1, self.config.epochs))

            train_loss = self.train_epoch(train_dataloader)
            self.history["train_loss"].append(train_loss)
            print("Train loss: {:.4f}".format(train_loss))

            if val_dataloader is not None:
                val_metrics = self.evaluate(val_dataloader)
                val_loss = val_metrics["loss"]
                self.history["val_loss"].append(val_loss)
                print("Val loss: {:.4f}".format(val_loss))

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.config.early_stopping_patience:
                    print("Early stopping at epoch {}".format(epoch + 1))
                    break

        return self.history
