"""Persister for evaluation results using SQLite."""

import contextlib
import fcntl
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from mrna_bench.datasets.benchmark_dataset import BenchmarkDataset
from mrna_bench.embedder.embedder_utils import get_config_hash

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS runs (
    run_id           INTEGER PRIMARY KEY,
    model            TEXT NOT NULL,
    evaluation       TEXT NOT NULL,
    task             TEXT NOT NULL,
    target_col       TEXT NOT NULL,
    split_type       TEXT NOT NULL,
    seed             TEXT NOT NULL,
    embedding_hash   TEXT NOT NULL,
    probe_hash       TEXT NOT NULL,
    embedding_config TEXT NOT NULL,
    probe_config     TEXT NOT NULL,
    fit_info         TEXT,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (model, evaluation, task, target_col, split_type, seed,
            embedding_hash, probe_hash)
);
CREATE TABLE IF NOT EXISTS metrics (
    run_id INTEGER NOT NULL REFERENCES runs (run_id),
    split  TEXT NOT NULL,
    metric TEXT NOT NULL,
    value  REAL,
    PRIMARY KEY (run_id, split, metric)
);
"""


def metrics_by_split(metrics: dict[str, float]) -> dict[str, dict]:
    """Group metrics named ``{split}_{metric}`` by split.

    Args:
        metrics: Metrics such as {"test_r": 0.5, "val_r": 0.4}.

    Returns:
        Metrics per split, such as {"test": {"r": 0.5}, "val": {"r": 0.4}}.
    """
    by_split: dict[str, dict] = {}
    for name, value in metrics.items():
        split, metric = name.split("_", 1)
        by_split.setdefault(split, {})[metric] = value
    return by_split


_KEY = (
    "model=? AND evaluation=? AND task=? AND target_col=? AND split_type=?"
    " AND seed=? AND embedding_hash=? AND probe_hash=?"
)


class ResultsPersister:
    """Persists and loads evaluation results using SQLite.

    Results are stored in ``{dataset_path}/results.db``. Each run is a row
    of ``runs`` identified by the model, evaluation, task, target, split,
    seed and the hashes of the embedding and probe configs; its metrics are
    rows of ``metrics`` keyed by run, split and metric name. Re-running a
    configuration overwrites its previous result.

    Connections acquire an exclusive fcntl.flock on ``results.db.lock``
    first. This serializes concurrent SLURM jobs on NFS without relying on
    SQLite's WAL locking, which is unreliable across NFS nodes.
    """

    def __init__(
        self,
        dataset: BenchmarkDataset,
        model_short_name: str,
        evaluation: str,
        task: str,
        target_col: str,
        split_type: str,
        embedding_config: dict,
        probe_config: dict,
    ):
        """Initialize ResultsPersister.

        Args:
            dataset: Dataset being evaluated.
            model_short_name: Name of model evaluated.
            evaluation: Evaluation method, e.g. "linear_probe".
            task: Task evaluated.
            target_col: Target column evaluated.
            split_type: Type of data split used.
            embedding_config: Settings the model was run with, as stored
                with its embeddings.
            probe_config: Settings of the probe or training run.
        """
        self.dataset = dataset
        self.model_short_name = model_short_name
        self.evaluation = evaluation
        self.task = task
        self.target_col = target_col
        self.split_type = split_type
        self.embedding_config = embedding_config
        self.probe_config = probe_config

        self._db_path = Path(dataset.dataset_path) / "results.db"
        self._lock_path = Path(dataset.dataset_path) / "results.db.lock"

    @contextmanager
    def _locked_connect(self) -> Generator[sqlite3.Connection, None, None]:
        """Acquire an exclusive file lock, then yield an open connection."""
        with open(self._lock_path, "w") as lock_fh:
            fcntl.flock(lock_fh, fcntl.LOCK_EX)
            conn = sqlite3.connect(str(self._db_path), timeout=60)
            conn.row_factory = sqlite3.Row
            try:
                conn.executescript(_SCHEMA)
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                with contextlib.suppress(Exception):
                    conn.close()

    def _key_tuple(self, seed: int | str) -> tuple:
        return (
            self.model_short_name, self.evaluation, self.task,
            self.target_col, self.split_type, str(seed),
            get_config_hash(self.embedding_config),
            get_config_hash(self.probe_config),
        )

    def get_checkpoint_path(self, seed: int | str) -> Path:
        """Get path of the trained weights saved with a result.

        Args:
            seed: Random seed used for data split.

        Returns:
            Path in the dataset checkpoint directory named by the result key.
        """
        key = (self.dataset.dataset_name, *self._key_tuple(seed))
        return Path(self.dataset.checkpoint_dir) / ("_".join(key) + ".pt")

    def result_exists(self, seed: int | str) -> bool:
        """Check whether a result exists for the given seed.

        Args:
            seed: Random seed used for data split.

        Returns:
            True if a matching run exists in the database.
        """
        if not self._db_path.exists():
            return False

        with self._locked_connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM runs WHERE " + _KEY, self._key_tuple(seed)
            ).fetchone()
        return row is not None

    def persist_run_results(
        self,
        metrics: dict[str, dict[str, float]],
        random_seed: int | str,
        fit_info: dict | None = None,
    ) -> None:
        """Persist results of one run, replacing any earlier result.

        Args:
            metrics: Metric values per split, e.g. {"test": {"r": 0.5}}.
            random_seed: Random seed used for data split.
            fit_info: Values produced by fitting, e.g. the chosen
                regularization or the training history.
        """
        self.persist_multirun_results(
            {random_seed: metrics},
            {random_seed: fit_info} if fit_info else None,
        )

    def persist_multirun_results(
        self,
        metrics: dict[int | str, dict[str, dict[str, float]]],
        fit_info: dict[int | str, dict] | None = None,
    ) -> None:
        """Persist results of several seeds, replacing earlier results.

        Args:
            metrics: Metric values per split for each random seed.
            fit_info: Values produced by fitting for each random seed.
        """
        fit_info = fit_info or {}
        with self._locked_connect() as conn:
            for seed, seed_metrics in metrics.items():
                info = fit_info.get(seed)
                run_id = conn.execute(
                    "INSERT INTO runs"
                    " (model, evaluation, task, target_col, split_type, seed,"
                    "  embedding_hash, probe_hash, embedding_config,"
                    "  probe_config, fit_info)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                    " ON CONFLICT DO UPDATE SET"
                    "  embedding_config=excluded.embedding_config,"
                    "  probe_config=excluded.probe_config,"
                    "  fit_info=excluded.fit_info,"
                    "  created_at=CURRENT_TIMESTAMP"
                    " RETURNING run_id",
                    self._key_tuple(seed) + (
                        json.dumps(self.embedding_config, sort_keys=True),
                        json.dumps(self.probe_config, sort_keys=True),
                        json.dumps(info, default=float) if info else None,
                    ),
                ).fetchone()[0]
                conn.execute("DELETE FROM metrics WHERE run_id=?", (run_id,))
                conn.executemany(
                    "INSERT INTO metrics (run_id, split, metric, value)"
                    " VALUES (?, ?, ?, ?)",
                    [
                        (run_id, split, metric, float(value))
                        for split, split_metrics in seed_metrics.items()
                        for metric, value in split_metrics.items()
                    ],
                )

    def load_run_results(self, random_seed: int | str) -> dict:
        """Load the results of one run.

        Args:
            random_seed: Random seed used for data split.

        Returns:
            Dictionary with "metrics" per split and "fit_info".

        Raises:
            FileNotFoundError: If no result exists for this configuration.
        """
        if not self._db_path.exists():
            raise FileNotFoundError(
                "No results.db found at {}".format(self._db_path)
            )

        with self._locked_connect() as conn:
            run = conn.execute(
                "SELECT run_id, fit_info FROM runs WHERE " + _KEY,
                self._key_tuple(random_seed),
            ).fetchone()
            if run is None:
                raise FileNotFoundError(
                    "No results for seed {}".format(random_seed)
                )
            rows = conn.execute(
                "SELECT split, metric, value FROM metrics WHERE run_id=?",
                (run["run_id"],),
            ).fetchall()

        metrics: dict[str, dict[str, float]] = {}
        for row in rows:
            metrics.setdefault(row["split"], {})[row["metric"]] = row["value"]
        fit_info = json.loads(run["fit_info"]) if run["fit_info"] else None
        return {"metrics": metrics, "fit_info": fit_info}

    def load_multirun_results(
        self,
        random_seeds: list[int]
    ) -> dict[int, dict[str, dict[str, float]]]:
        """Load the metrics of several runs.

        Args:
            random_seeds: Random seeds used for data splits.

        Returns:
            Metrics per split for each random seed.
        """
        return {
            seed: self.load_run_results(seed)["metrics"]
            for seed in random_seeds
        }

    @staticmethod
    def load_all_results(dataset_path: str | Path) -> list[dict]:
        """Load every run and its metrics for a dataset.

        Args:
            dataset_path: Dataset directory containing results.db.

        Returns:
            One dictionary per run with its key, configs, fit info and
            metrics per split.
        """
        db_path = Path(dataset_path) / "results.db"
        if not db_path.exists():
            return []

        conn = sqlite3.connect(str(db_path), timeout=60)
        conn.row_factory = sqlite3.Row
        try:
            conn.executescript(_SCHEMA)
            runs = conn.execute("SELECT * FROM runs").fetchall()
            metric_rows = conn.execute("SELECT * FROM metrics").fetchall()
        finally:
            conn.close()

        metrics: dict[int, dict[str, dict[str, float]]] = {}
        for row in metric_rows:
            metrics.setdefault(row["run_id"], {}).setdefault(
                row["split"], {}
            )[row["metric"]] = row["value"]

        results = []
        for run in runs:
            result = dict(run)
            for col in ("embedding_config", "probe_config", "fit_info"):
                if result[col] is not None:
                    result[col] = json.loads(result[col])
            result["metrics"] = metrics.get(run["run_id"], {})
            results.append(result)
        return results
