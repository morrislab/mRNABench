import json
from pathlib import Path

from mrna_bench.datasets.benchmark_dataset import BenchmarkDataset


class LinearProbePersister:
    """Persists and loads linear probe results to and from disk."""

    def __init__(
        self,
        dataset: BenchmarkDataset,
        model_short_name: str,
        task: str,
        target_col: str,
        split_type: str
    ):
        """Initialize LinearProbePersister.

        Args:
            dataset_name: Name of the dataset evaluated.
            model_short_name: Name of model evaluated.
            task: Task evaluated.
            target_col: Target column evaluated.
            split_type: Type of data split used.
        """
        self.dataset = dataset
        self.model_short_name = model_short_name
        self.task = task
        self.target_col = target_col
        self.split_type = split_type

    def get_output_filename(self, random_seed: str | int) -> str:
        """Generate output filename for linear probing results.

        Args:
            random_seed: Random seed used for data split, or 'all' if getting
                file name for multi-run results.

        Returns:
            File name used for storing linear probing results.
        """
        out_fn = "result_lp_{}_{}_{}_tcol-{}_split-{}".format(
            self.dataset.dataset_name,
            self.model_short_name,
            self.task,
            self.target_col,
            self.split_type
        )

        if random_seed == "all":
            out_fn += "_rs-all"
        else:
            out_fn += "_rs-{}".format(random_seed)

        out_fn += ".json"

        return out_fn

    def persist_run_results(
        self,
        metrics: dict[str, float] | dict[str, str],
        random_seed: int | str
    ):
        """Persist linear probe results.

        Args:
            metrics: Linear probing metrics.
            random_seed: Random seed used for data split, or 'all'.
        """
        dataset_root = Path(self.dataset.dataset_path)

        result_dir = dataset_root / "lp_results"
        result_dir.mkdir(exist_ok=True)

        result_fn = self.get_output_filename(random_seed)

        with open(result_dir / result_fn, "w") as f:
            json.dump(metrics, f)

    def load_multirun_results(
        self,
        random_seeds: list[int]
    ) -> dict[int, dict[str, float]]:
        """Load multi-run linear probing results from persisted files.

        Args:
            random_seeds: Random seeds used for data splits.

        Returns:
            Dictionary of metrics per random seed used to generate data splits
            for each individual linear probing run.
        """
        metrics = {}
        dataset_root = Path(self.dataset.dataset_path)

        result_dir = dataset_root / "lp_results"

        for random_seed in random_seeds:
            result_fn = self.get_output_filename(random_seed)

            with open(result_dir / result_fn, "r") as f:
                metrics[random_seed] = json.load(f)

        return metrics
