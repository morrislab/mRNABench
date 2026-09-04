"""Trainer for fine-tuning genomic foundation models."""

import gc
from dataclasses import dataclass

from typing import Any, Callable

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
    # Decay schedule applied after warmup: "none", "linear", or "cosine".
    # "none" keeps LR constant after warmup.
    lr_schedule: str = "none"
    # Total optimizer steps for decay schedules. If None, computed from the
    # dataloader length at the start of fit().
    total_steps: int | None = None
    use_amp: bool = True
    random_seed: int | None = None


class FineTuneTrainer:
    """Trainer for fine-tuning nucleotide foundation models."""

    def __init__(
            self,
            wrapper: FineTuneWrapper,
            config: TrainerConfig | None = None
    ):
        """Initialize FineTuneTrainer.

        Args:
            wrapper: FineTuneWrapper with backbone and task head.
            config: Training configuration. Uses defaults if not provided.
        """
        self.wrapper = wrapper
        self.device = wrapper.device
        self.config = config or TrainerConfig()
        _head: Any = wrapper.task_head
        self.loss_fn: Callable[..., torch.Tensor] = _head.get_loss_fn()
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.best_val_metrics: dict[str, float] = {}

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
        total_steps: int,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """Create scheduler: linear warmup followed by optional decay.

        Args:
            optimizer: Optimizer to schedule.
            total_steps: Total optimizer steps (used for decay schedules).
        """
        valid = ("none", "linear", "cosine")
        if self.config.lr_schedule not in valid:
            raise ValueError(
                "lr_schedule must be one of {}; got {!r}".format(
                    valid, self.config.lr_schedule
                )
            )

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=self.config.warmup_steps,
        )

        decay_steps = max(total_steps - self.config.warmup_steps, 1)

        if self.config.lr_schedule == "none":
            decay: torch.optim.lr_scheduler.LRScheduler = (
                torch.optim.lr_scheduler.ConstantLR(
                    optimizer, factor=1.0, total_iters=decay_steps,
                )
            )
        elif self.config.lr_schedule == "linear":
            decay = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=decay_steps,
            )
        else:  # cosine
            decay = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=decay_steps, eta_min=0.0,
            )

        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, decay],
            milestones=[self.config.warmup_steps],
        )

    def _save_trainable_state(self) -> dict:
        """Save only trainable (LoRA + head) state dicts.

        Returns:
            Dictionary with LoRA and head state dicts.
        """
        model = self.wrapper.backbone.get_peft_target()
        trainable = {
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        lora_state = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
            if name in trainable
        }
        head_state = {
            name: value.detach().clone()
            for name, value in self.wrapper.task_head.state_dict().items()
        }
        return {"lora": lora_state, "head": head_state}

    def _restore_trainable_state(self, state: dict):
        """Restore trainable state dicts.

        Args:
            state: Dictionary with LoRA and head state dicts.
        """
        self.wrapper.backbone.get_peft_target().load_state_dict(
            state["lora"], strict=False
        )

        self.wrapper.task_head.load_state_dict(state["head"])

    def _forward_batch(
        self, batch: dict, is_vep: bool
    ) -> torch.Tensor:
        """Dispatch a batch through the wrapper's standard or VEP forward.

        Args:
            batch: Collated batch dict from the dataloader.
            is_vep: Whether this is a VEP paired batch.

        Returns:
            Model output tensor.
        """
        if is_vep:
            return self.wrapper.forward_vep(
                ref_sequences=batch["ref_sequence"],
                alt_sequences=batch["alt_sequence"],
                ref_cds=batch.get("ref_cds"),
                alt_cds=batch.get("alt_cds"),
                ref_splice=batch.get("ref_splice"),
                alt_splice=batch.get("alt_splice"),
            )
        return self.wrapper.forward(
            sequences=batch["sequence"],
            cds=batch.get("cds"),
            splice=batch.get("splice"),
        )

    def _get_amp_dtype(self) -> torch.dtype | None:
        """Return the autocast dtype if AMP is enabled, else None."""
        if not self.config.use_amp:
            return None
        if self.device.type == "cuda":
            return torch.bfloat16
        return None

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch.

        Args:
            dataloader: Training data loader.

        Returns:
            Average training loss.
        """
        assert self.optimizer is not None, "Call fit() before train_epoch()"
        self.wrapper.set_train_mode()
        total_loss = 0.0
        num_batches = 0
        accum = self.config.gradient_accumulation_steps
        amp_dtype = self._get_amp_dtype()

        progress = tqdm(dataloader, desc="Training")
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(progress):
            target = batch["target"]
            if not isinstance(target, np.ndarray):
                target = np.array(target)
            target = self.wrapper.task_head.prepare_targets(
                target
            )  # type: ignore[operator]
            target = target.to(self.device)

            is_vep = "ref_sequence" in batch

            if amp_dtype is not None:
                with torch.amp.autocast(
                    self.device.type, dtype=amp_dtype
                ):
                    output = self._forward_batch(batch, is_vep)
                    loss = self.loss_fn(output.float(), target)
            else:
                output = self._forward_batch(batch, is_vep)
                loss = self.loss_fn(output, target)

            batch_loss = loss.item()

            # Scale by the actual number of batches in this accumulation
            # group — the final group may be smaller than `accum`.
            remaining = len(dataloader) - (batch_idx // accum) * accum
            group_size = min(accum, remaining)
            loss = loss / group_size
            loss.backward()

            if (batch_idx + 1) % accum == 0 or (
                batch_idx + 1
            ) == len(dataloader):
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
        amp_dtype = self._get_amp_dtype()

        for batch in tqdm(dataloader, desc="Evaluating"):
            target = batch["target"]
            if not isinstance(target, np.ndarray):
                target = np.array(target)
            target = self.wrapper.task_head.prepare_targets(
                target
            )  # type: ignore[operator]
            target = target.to(self.device)

            is_vep = "ref_sequence" in batch
            batch_size = len(
                batch["ref_sequence"] if is_vep else batch["sequence"]
            )

            if amp_dtype is not None:
                with torch.amp.autocast(
                    self.device.type, dtype=amp_dtype
                ):
                    output = self._forward_batch(batch, is_vep)
                    loss = self.loss_fn(output.float(), target)
            else:
                output = self._forward_batch(batch, is_vep)
                loss = self.loss_fn(output, target)
            total_loss += loss.item() * batch_size

            all_preds.append(output.float().cpu())
            all_targets.append(target.cpu())

        n = len(dataloader.dataset)  # type: ignore[arg-type]
        avg_loss = total_loss / n
        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)

        metrics = {"loss": avg_loss}
        metrics.update(
            self.wrapper.task_head.compute_metrics(
                preds,
                targets
            )  # type: ignore[operator]
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
        if self.config.random_seed is not None:
            torch.manual_seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.config.random_seed)

        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
        }

        self.optimizer = self._create_optimizer()
        steps_per_epoch = max(
            len(train_dataloader) // self.config.gradient_accumulation_steps, 1
        )
        total_steps = self.config.total_steps or (
            steps_per_epoch * self.config.epochs
        )
        self.scheduler = self._create_scheduler(self.optimizer, total_steps)

        best_val_loss = float("inf")
        best_state = None
        best_val_metrics: dict[str, float] = {}
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
                    best_val_metrics = val_metrics
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.config.early_stopping_patience:
                    print("Early stopping at epoch {}".format(epoch + 1))
                    break

        if best_state is not None:
            self._restore_trainable_state(best_state)

        self.best_val_metrics = best_val_metrics

        return self.history

    @staticmethod
    def cleanup():
        """Free GPU memory between runs."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
