"""Zero-shot VEP scoring using embedding delta norms."""

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from mrna_bench.linear_probe.persister import LinearProbePersister
from mrna_bench.linear_probe.vep import compute_vep_deltas

_ZEROSHOT_SEED = "zeroshot"


class ZeroShotVEP:
    """Zero-shot variant effect prediction via embedding delta scoring.

    No training is performed. Each variant's embedding delta (variant minus
    wildtype) is mapped to a scalar score with a user-supplied function
    (default: L2 norm), then evaluated against binary labels using AUROC
    and AUPRC across all non-wildtype rows.
    """

    def __init__(
        self,
        data_df: pd.DataFrame,
        embeddings: np.ndarray,
        target_col: str,
        persister: LinearProbePersister | None = None,
        scoring_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        """Initialize ZeroShotVEP.

        Args:
            data_df: Dataset dataframe; must contain ``transcript_id``,
                ``description``, and ``target_col`` columns.
            embeddings: Pre-computed embeddings aligned with ``data_df`` rows.
            target_col: Binary label column for AUROC/AUPRC evaluation.
            persister: Optional persister for writing results to results.db.
            scoring_fn: Callable mapping an (N, D) delta matrix to (N,)
                scalar scores. Defaults to row-wise L2 norm.
        """
        if target_col not in data_df.columns:
            raise ValueError(f"Target column '{target_col}' not in dataframe.")

        self.data_df = data_df.copy()
        self.data_df["embeddings"] = list(embeddings)
        self.target_col = target_col
        self.persister = persister
        self.scoring_fn: Callable[[np.ndarray], np.ndarray] = (
            scoring_fn if scoring_fn is not None
            else lambda X: np.linalg.norm(X, axis=1)
        )

    def result_exists(self) -> bool:
        """Return True if a zeroshot result is already stored."""
        if self.persister is None:
            return False
        return self.persister.result_exists(_ZEROSHOT_SEED)

    def run(self, persist: bool = False) -> dict[str, float]:
        """Compute zero-shot VEP scores and evaluate.

        Args:
            persist: Write results to results.db when True.

        Returns:
            Dict with ``auroc`` and ``auprc`` keys.
        """
        delta_df = compute_vep_deltas(self.data_df)
        delta_df = delta_df.dropna(subset=[self.target_col])

        delta_matrix = np.stack(delta_df["embeddings"].tolist())
        scores = self.scoring_fn(delta_matrix)
        labels = delta_df[self.target_col].to_numpy()

        metrics: dict[str, float] = {
            "auroc": float(roc_auc_score(labels, scores)),
            "auprc": float(average_precision_score(labels, scores)),
        }

        if persist:
            if self.persister is None:
                raise RuntimeError(
                    "Must provide persister to persist results."
                )
            self.persister.persist_run_results(metrics, _ZEROSHOT_SEED)

        return metrics
