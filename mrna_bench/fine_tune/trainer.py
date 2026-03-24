"""Trainer for fine-tuning genomic foundation models."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mrna_bench.fine_tune.fine_tune_wrapper import FineTuneWrapper


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

    def __init__(self, wrapper: FineTuneWrapper, config: TrainerConfig | None = None):
        """Initialize FineTuneTrainer.

        Args:
            wrapper: FineTuneWrapper with backbone and task head.
            config: Training configuration. Uses defaults if not provided.
        """
        self.wrapper = wrapper
        self.device = wrapper.device
        self.config = config or TrainerConfig()
        self.loss_fn = wrapper.task_head.get_loss_fn()
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
        }

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create AdamW optimizer for trainable parameters."""
        params = self.wrapper.get_trainable_parameters()
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

    def _save_trainable_state(self) -> dict:
        """Save only trainable (LoRA + head) state dicts.

        Returns:
            Dictionary with LoRA and head state dicts.
        """
        try:
            from peft import get_peft_model_state_dict
            lora_state = deepcopy(
                get_peft_model_state_dict(self.wrapper.backbone.model)
            )
        except ImportError:
            lora_state = deepcopy({
                k: v for k, v in self.wrapper.backbone.model.state_dict().items()
                if v.requires_grad
            })

        head_state = deepcopy(self.wrapper.task_head.state_dict())
        return {"lora": lora_state, "head": head_state}

    def _restore_trainable_state(self, state: dict):
        """Restore trainable state dicts.

        Args:
            state: Dictionary with LoRA and head state dicts.
        """
        try:
            from peft import set_peft_model_state_dict
            set_peft_model_state_dict(
                self.wrapper.backbone.model, state["lora"]
            )
        except ImportError:
            self.wrapper.backbone.model.load_state_dict(
                state["lora"], strict=False
            )

        self.wrapper.task_head.load_state_dict(state["head"])

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch.

        Args:
            dataloader: Training data loader.

        Returns:
            Average training loss.
        """
        self.wrapper.set_train_mode()
        total_loss = 0.0
        num_batches = 0

        progress = tqdm(dataloader, desc="Training")
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(progress):
            target = batch["target"]
            if not isinstance(target, np.ndarray):
                target = np.array(target)
            target = self.wrapper.task_head.prepare_targets(target)
            target = target.to(self.device)

            sequences = batch["sequence"]
            cds = batch.get("cds")
            splice = batch.get("splice")

            output = self.wrapper.forward(sequences, cds, splice)
            loss = self.loss_fn(output, target)

            batch_loss = loss.item()
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.wrapper.get_trainable_parameters(),
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
        self.wrapper.set_inference_mode()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        for batch in tqdm(dataloader, desc="Evaluating"):
            target = batch["target"]
            if not isinstance(target, np.ndarray):
                target = np.array(target)
            target = self.wrapper.task_head.prepare_targets(target)
            target = target.to(self.device)

            sequences = batch["sequence"]
            cds = batch.get("cds")
            splice = batch.get("splice")

            output = self.wrapper.forward(sequences, cds, splice)
            loss = self.loss_fn(output, target)
            total_loss += loss.item() * len(sequences)

            all_preds.append(output.cpu())
            all_targets.append(target.cpu())

        avg_loss = total_loss / len(dataloader.dataset)
        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)

        metrics = {"loss": avg_loss}
        metrics.update(
            self.wrapper.task_head.compute_metrics(preds, targets)
        )
        return metrics

    def fit(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None = None,
    ) -> dict[str, list[float]]:
        """Full training loop with early stopping and best-model restore.

        Args:
            train_dataloader: Training data loader.
            val_dataloader: Validation data loader (optional).

        Returns:
            Training history dictionary.
        """
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler(self.optimizer)

        best_val_loss = float("inf")
        best_state = None
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
                    best_state = self._save_trainable_state()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.config.early_stopping_patience:
                    print("Early stopping at epoch {}".format(epoch + 1))
                    break

        if best_state is not None:
            self._restore_trainable_state(best_state)

        return self.history
