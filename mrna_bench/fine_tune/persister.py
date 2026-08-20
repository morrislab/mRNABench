"""Persister for fine-tuning results."""

import json
from pathlib import Path

from mrna_bench.datasets.benchmark_dataset import BenchmarkDataset


class FineTunePersister:
    """Persists and loads fine-tuning results to and from disk."""

    def __init__(
        self,
        dataset: BenchmarkDataset,
        model_short_name: str,
        task: str,
        target_col: str,
        split_type: str,
        learning_rate: float,
        lora_rank: int | None = None,
        lora_alpha: int | None = None,
        head_type: str = "mlp",
    ):
        """Initialize FineTunePersister.

        Args:
            dataset: Dataset being evaluated.
            model_short_name: Name of model evaluated.
            task: Task evaluated (regression, classification, multilabel).
            target_col: Target column evaluated.
            split_type: Type of data split used.
            learning_rate: Learning rate used for fine-tuning.
            lora_rank: LoRA rank used (None if full fine-tune).
            lora_alpha: LoRA scaling factor.
            head_type: Type of task head used (e.g. "mlp", "cnn").
        """
        self.dataset = dataset
        self.model_short_name = model_short_name
        self.task = task
        self.target_col = target_col
        self.split_type = split_type
        self.learning_rate = learning_rate
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.head_type = head_type

        self._result_dir = Path(dataset.dataset_path) / "ft_results"

    def _get_path(self, random_seed: int | str, suffix: str = "") -> Path:
        """Get full path for a result file.

        Args:
            random_seed: Random seed used for data split.
            suffix: File suffix (e.g., ".json", "_model.pt").
        """
        parts = [
            "result_ft",
            self.dataset.dataset_name,
            self.model_short_name,
            self.task,
            "tcol-{}".format(self.target_col),
            "split-{}".format(self.split_type),
            "lr-{}".format(self.learning_rate),
        ]

        if self.lora_rank is not None:
            parts.append("lora-{}".format(self.lora_rank))
        if self.lora_alpha is not None:
            parts.append("alpha-{}".format(self.lora_alpha))

        parts.append("head-{}".format(self.head_type))
        parts.append("rs-{}".format(random_seed))

        return self._result_dir / ("_".join(parts) + suffix)

    def _build_config(self, random_seed: int | str) -> dict:
        """Build config dict for self-describing results.

        Args:
            random_seed: Random seed used for data split.

        Returns:
            Configuration dictionary.
        """
        return {
            "model": self.model_short_name,
            "dataset": self.dataset.dataset_name,
            "task": self.task,
            "target_col": self.target_col,
            "split_type": self.split_type,
            "learning_rate": self.learning_rate,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "head_type": self.head_type,
            "seed": random_seed,
        }

    def persist_run_results(
        self,
        metrics: dict,
        random_seed: int | str,
        history: dict[str, list[float]] | None = None,
    ):
        """Persist fine-tuning results.

        Args:
            metrics: Fine-tuning metrics with "val" and optionally "test" keys.
            random_seed: Random seed used for data split.
            history: Training history (train_loss, val_loss per epoch).
        """
        self._result_dir.mkdir(exist_ok=True)

        results = {
            "config": self._build_config(random_seed),
            "metrics": metrics,
        }
        if history is not None:
            results["history"] = history

        with open(self._get_path(random_seed, ".json"), "w") as f:
            json.dump(results, f, indent=2)

    def load_run_results(self, random_seed: int | str) -> dict:
        """Load fine-tuning results from persisted file.

        Args:
            random_seed: Random seed used for data split.

        Returns:
            Dictionary containing config, metrics, and optionally history.
        """
        with open(self._get_path(random_seed, ".json"), "r") as f:
            return json.load(f)
