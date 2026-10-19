"""Persister for linear probe results using SQLite."""

import json
import sqlite3
from pathlib import Path

from mrna_bench.datasets.benchmark_dataset import BenchmarkDataset

_LP_SCHEMA = """\
CREATE TABLE IF NOT EXISTS lp_results (
    model      TEXT NOT NULL,
    task       TEXT NOT NULL,
    target_col TEXT NOT NULL,
    split_type TEXT NOT NULL,
    seed       TEXT NOT NULL,
    metrics    TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (model, task, target_col, split_type, seed)
)
"""


class LinearProbePersister:
    """Persists and loads linear probe results using SQLite.

    Results are stored in ``{dataset_path}/results.db`` in the
    ``lp_results`` table. Re-running with the same configuration
    overwrites the previous result (upsert).
    """

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
            dataset: Dataset being evaluated.
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

        self._db_path = Path(dataset.dataset_path) / "results.db"

    def _connect(self) -> sqlite3.Connection:
        """Return a connection with WAL mode and Row factory."""
        conn = sqlite3.connect(str(self._db_path), timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_table(self, conn: sqlite3.Connection):
        conn.execute(_LP_SCHEMA)

    def _key_tuple(self, seed: int | str) -> tuple:
        return (
            self.model_short_name, self.task, self.target_col,
            self.split_type, str(seed),
        )

    def result_exists(self, seed: int | str) -> bool:
        """Check whether a result row exists for the given seed.

        Args:
            seed: Random seed used for data split, or "all".

        Returns:
            True if a matching row exists in the database.
        """
        with self._connect() as conn:
            self._ensure_table(conn)
            row = conn.execute(
                "SELECT 1 FROM lp_results"
                " WHERE model=? AND task=? AND target_col=?"
                "   AND split_type=? AND seed=?",
                self._key_tuple(seed),
            ).fetchone()
        return row is not None

    def persist_run_results(
        self,
        metrics: dict[str, float] | dict[str, str],
        random_seed: int | str
    ):
        """Persist linear probe results (upsert).

        Args:
            metrics: Linear probing metrics.
            random_seed: Random seed used for data split, or 'all'.
        """
        with self._connect() as conn:
            self._ensure_table(conn)
            conn.execute(
                "INSERT OR REPLACE INTO lp_results"
                " (model, task, target_col, split_type, seed, metrics)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                self._key_tuple(random_seed) + (
                    json.dumps(metrics, default=float),
                ),
            )

    def load_multirun_results(
        self,
        random_seeds: list[int]
    ) -> dict[int, dict[str, float]]:
        """Load multi-run linear probing results from the database.

        Args:
            random_seeds: Random seeds used for data splits.

        Returns:
            Dictionary of metrics per random seed.

        Raises:
            FileNotFoundError: If results for a seed are missing.
        """
        metrics = {}
        with self._connect() as conn:
            self._ensure_table(conn)
            for seed in random_seeds:
                row = conn.execute(
                    "SELECT metrics FROM lp_results"
                    " WHERE model=? AND task=? AND target_col=?"
                    "   AND split_type=? AND seed=?",
                    self._key_tuple(seed),
                ).fetchone()
                if row is None:
                    raise FileNotFoundError(
                        "No LP results for seed {}".format(seed)
                    )
                metrics[seed] = json.loads(row["metrics"])
        return metrics

    @staticmethod
    def load_all_results(dataset_path: str | Path) -> list[dict]:
        """Load every LP result row for a dataset.

        Args:
            dataset_path: Root path of the dataset.

        Returns:
            List of dicts with model, task, target_col, split_type,
            seed, and metrics keys.
        """
        db_path = Path(dataset_path) / "results.db"
        if not db_path.exists():
            return []

        conn = sqlite3.connect(str(db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")

        rows = conn.execute(
            "SELECT model, task, target_col, split_type,"
            "       seed, metrics FROM lp_results"
        ).fetchall()
        conn.close()

        return [
            {
                "model": r["model"],
                "task": r["task"],
                "target_col": r["target_col"],
                "split_type": r["split_type"],
                "seed": r["seed"],
                "metrics": json.loads(r["metrics"]),
            }
            for r in rows
        ]
