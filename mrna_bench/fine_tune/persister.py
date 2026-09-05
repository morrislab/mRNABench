"""Persister for fine-tuning results using SQLite."""

import contextlib
import fcntl
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from mrna_bench.datasets.benchmark_dataset import BenchmarkDataset

_FT_SCHEMA = """\
CREATE TABLE IF NOT EXISTS ft_results (
    model      TEXT NOT NULL,
    task       TEXT NOT NULL,
    target_col TEXT NOT NULL,
    split_type TEXT NOT NULL,
    lr         TEXT NOT NULL,
    lora_rank  TEXT NOT NULL,
    lora_alpha TEXT NOT NULL,
    head_type  TEXT NOT NULL,
    seed       TEXT NOT NULL,
    metrics    TEXT NOT NULL,
    history    TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (model, task, target_col, split_type,
                 lr, lora_rank, lora_alpha, head_type, seed)
)
"""


class FineTunePersister:
    """Persists and loads fine-tuning results using SQLite.

    Results are stored in ``{dataset_path}/results.db`` in the
    ``ft_results`` table (alongside any ``lp_results`` rows).

    Write operations acquire an exclusive fcntl.flock on
    ``results.db.lock`` to serialize concurrent SLURM jobs on NFS.
    """

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
            lora_rank: LoRA rank used (None if head-only).
            lora_alpha: LoRA scaling factor (None if head-only).
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
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                with contextlib.suppress(Exception):
                    conn.close()

    def _ensure_table(self, conn: sqlite3.Connection) -> None:
        conn.execute(_FT_SCHEMA)

    def _key_tuple(self, seed: int | str) -> tuple:
        return (
            self.model_short_name,
            self.task,
            self.target_col,
            self.split_type,
            str(self.learning_rate),
            str(self.lora_rank) if self.lora_rank is not None else "none",
            str(self.lora_alpha) if self.lora_alpha is not None else "none",
            self.head_type,
            str(seed),
        )

    def result_exists(self, seed: int | str) -> bool:
        """Check whether a result row exists for the given seed.

        Args:
            seed: Random seed used for data split.

        Returns:
            True if a matching row exists in the database.
        """
        if not self._db_path.exists():
            return False

        try:
            with self._locked_connect() as conn:
                self._ensure_table(conn)
                row = conn.execute(
                    "SELECT 1 FROM ft_results"
                    " WHERE model=? AND task=? AND target_col=?"
                    "   AND split_type=? AND lr=? AND lora_rank=?"
                    "   AND lora_alpha=? AND head_type=? AND seed=?",
                    self._key_tuple(seed),
                ).fetchone()
        except sqlite3.DatabaseError as exc:
            raise sqlite3.DatabaseError(
                f"results.db is corrupted at {self._db_path}. "
                "Delete it and re-run to recover."
            ) from exc
        return row is not None

    def persist_run_results(
        self,
        metrics: dict,
        random_seed: int | str,
        history: dict[str, list[float]] | None = None,
    ):
        """Persist fine-tuning results (upsert).

        Args:
            metrics: Fine-tuning metrics with "val" and optionally "test" keys.
            random_seed: Random seed used for data split.
            history: Training history (train_loss, val_loss per epoch).
        """
        with self._locked_connect() as conn:
            self._ensure_table(conn)
            conn.execute(
                "INSERT OR REPLACE INTO ft_results"
                " (model, task, target_col, split_type,"
                "  lr, lora_rank, lora_alpha, head_type,"
                "  seed, metrics, history)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                self._key_tuple(random_seed) + (
                    json.dumps(metrics, default=float),
                    json.dumps(history, default=float) if history else None,
                ),
            )

    def load_run_results(self, random_seed: int | str) -> dict:
        """Load fine-tuning results for a single seed.

        Args:
            random_seed: Random seed used for data split.

        Returns:
            Dictionary containing metrics and optionally history.

        Raises:
            FileNotFoundError: If no result exists for this configuration.
        """
        if not self._db_path.exists():
            raise FileNotFoundError(
                "No results.db found at {}".format(self._db_path)
            )

        with self._locked_connect() as conn:
            self._ensure_table(conn)
            row = conn.execute(
                "SELECT metrics, history FROM ft_results"
                " WHERE model=? AND task=? AND target_col=?"
                "   AND split_type=? AND lr=? AND lora_rank=?"
                "   AND lora_alpha=? AND head_type=? AND seed=?",
                self._key_tuple(random_seed),
            ).fetchone()

        if row is None:
            raise FileNotFoundError(
                "No FT results for seed {}".format(random_seed)
            )

        result = {"metrics": json.loads(row["metrics"])}
        if row["history"]:
            result["history"] = json.loads(row["history"])
        return result

    @staticmethod
    def load_all_results(dataset_path: str | Path) -> list[dict]:
        """Load every FT result row for a dataset."""
        db_path = Path(dataset_path) / "results.db"
        if not db_path.exists():
            return []

        conn = sqlite3.connect(str(db_path), timeout=60)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute(
                "SELECT name FROM sqlite_master"
                " WHERE type='table' AND name='ft_results'"
            )
            if not conn.execute(
                "SELECT name FROM sqlite_master"
                " WHERE type='table' AND name='ft_results'"
            ).fetchone():
                return []

            rows = conn.execute(
                "SELECT model, task, target_col, split_type,"
                "       lr, lora_rank, lora_alpha, head_type,"
                "       seed, metrics, history FROM ft_results"
            ).fetchall()
        finally:
            conn.close()

        return [
            {
                "model": row["model"],
                "task": row["task"],
                "target_col": row["target_col"],
                "split_type": row["split_type"],
                "lr": row["lr"],
                "lora_rank": row["lora_rank"],
                "lora_alpha": row["lora_alpha"],
                "head_type": row["head_type"],
                "seed": row["seed"],
                "metrics": json.loads(row["metrics"]),
                "history": (
                    json.loads(row["history"])
                    if row["history"] else None
                ),
            }
            for row in rows
        ]
