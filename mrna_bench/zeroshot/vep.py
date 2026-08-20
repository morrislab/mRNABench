from typing import TYPE_CHECKING, Callable
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from mrna_bench.linear_probe.persister import LinearProbePersister

if TYPE_CHECKING:
    from mrna_bench.datasets import BenchmarkDataset
    from mrna_bench.models import EmbeddingModel, ModelBehavior

_ZEROSHOT_SEED = "embedding_vep"
_MASKED_MARGINAL = "masked_marginal"


class ZeroShotVEP:
    """Evaluate embedding- or likelihood-based zero-shot variant scores."""

    def __init__(
        self,
        data_df: pd.DataFrame,
        target_col: str,
        persister: LinearProbePersister | None = None,
        scoring_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        model: "EmbeddingModel | None" = None,
        score_method: "ModelBehavior | str | None" = None,
        normalization: str = "mean",
    ):
        """Initialize ZeroShotVEP.

        Args:
            data_df: Dataset-specific VEP rows normalized to ref/alt columns.
            target_col: Binary label column for AUROC/AUPRC evaluation.
            persister: Optional persister for writing results to results.db.
            scoring_fn: Callable mapping an (N, D) delta matrix to (N,)
                scalar scores. Defaults to row-wise L2 norm.
            model: Optional likelihood-capable model instead of embeddings.
            score_method: Causal likelihood, pseudo-likelihood, or masked
                marginal scoring.
            normalization: Sum for a likelihood ratio or mean for a
                length-normalized score.
        """
        if target_col not in data_df.columns:
            raise ValueError(f"Target column '{target_col}' not in dataframe.")

        self.data_df = data_df.copy()
        self.target_col = target_col
        self.persister = persister
        self.model = model
        self.score_method = score_method
        self.normalization = normalization
        self.result_key = (
            _ZEROSHOT_SEED
            if model is None
            else "{}-{}-attn-{}".format(
                score_method,
                normalization,
                getattr(model, "attn_implementation", None) or "none",
            )
        )
        self.scoring_fn: Callable[[np.ndarray], np.ndarray] = (
            scoring_fn if scoring_fn is not None
            else lambda X: np.linalg.norm(X, axis=1)
        )

    @classmethod
    def from_embeddings(
        cls,
        dataset: "BenchmarkDataset",
        embeddings: np.ndarray,
        target_col: str | None = None,
        **kwargs,
    ) -> "ZeroShotVEP":
        """Build embedding VEP directly from a dataset and embeddings."""
        from mrna_bench.datasets import EvaluationMethod

        if EvaluationMethod.EMBEDDING_VEP not in dataset.metadata.evaluations:
            raise ValueError(
                f"{dataset.dataset_name} does not support embedding VEP."
            )
        data_df = dataset.data_df.copy()
        data_df["embeddings"] = list(embeddings)
        return cls(
            dataset.get_vep_pairs(data_df, ("embeddings",)),
            target_col or dataset.metadata.target_col[0],
            **kwargs,
        )

    @classmethod
    def from_model(
        cls,
        dataset: "BenchmarkDataset",
        model: "EmbeddingModel",
        score_method: "ModelBehavior | str | None" = None,
        target_col: str | None = None,
        normalization: str = "sum",
        **kwargs,
    ) -> "ZeroShotVEP":
        """Build likelihood VEP directly from a dataset and model."""
        from mrna_bench.datasets import EvaluationMethod

        if EvaluationMethod.LIKELIHOOD_VEP not in (
            dataset.metadata.compatible_evaluations(model)
        ):
            raise ValueError(
                f"{model.__class__.__name__} cannot run likelihood VEP on "
                f"{dataset.dataset_name}."
            )
        data_df = dataset.get_vep_pairs(dataset.data_df)
        from mrna_bench.models import ModelBehavior

        supported = model.behaviors.intersection({
            ModelBehavior.CAUSAL_LIKELIHOOD,
            ModelBehavior.PSEUDO_LIKELIHOOD,
        })
        if score_method is None:
            if len(supported) != 1:
                raise ValueError(
                    "score_method is required when the model does not expose "
                    "exactly one likelihood method."
                )
            score_method = next(iter(supported))
        elif score_method == _MASKED_MARGINAL:
            if ModelBehavior.PSEUDO_LIKELIHOOD not in supported:
                raise ValueError(
                    "masked_marginal requires pseudo_likelihood support."
                )
        else:
            score_method = ModelBehavior(score_method)
            if score_method not in supported:
                raise ValueError(
                    "{} does not support {}.".format(
                        model.__class__.__name__,
                        score_method.value,
                    )
                )
        return cls(
            data_df,
            target_col or dataset.metadata.target_col[0],
            model=model,
            score_method=score_method,
            normalization=normalization,
            **kwargs,
        )

    def result_exists(self) -> bool:
        """Return True if a zeroshot result is already stored."""
        if self.persister is None:
            return False
        return self.persister.result_exists(self.result_key)

    def run(self, persist: bool = False) -> dict[str, float]:
        """Compute zero-shot VEP scores and evaluate.

        Args:
            persist: Write results to results.db when True.

        Returns:
            Dict with ``auroc`` and ``auprc`` keys.
        """
        scored_df = self.data_df.dropna(subset=[self.target_col])
        if self.model is None:
            delta_matrix = np.subtract(
                np.stack(scored_df["alt_embeddings"]),
                np.stack(scored_df["ref_embeddings"]),
            )
            scores = self.scoring_fn(delta_matrix)
        else:
            if self.score_method == _MASKED_MARGINAL:
                substitutions = scored_df["ref_sequence"].str.len().eq(
                    scored_df["alt_sequence"].str.len()
                )
                if not substitutions.all():
                    dropped = int((~substitutions).sum())
                    warnings.warn(
                        "Masked-marginal VEP excludes {} indel variants."
                        .format(dropped),
                        RuntimeWarning,
                    )
                    scored_df = scored_df[substitutions]
                if scored_df.empty:
                    raise ValueError(
                        "Masked-marginal VEP requires substitution variants."
                    )
            refs = scored_df["ref_sequence"].tolist()
            alts = scored_df["alt_sequence"].tolist()
            ref_cds = self._track_values(scored_df, "ref_cds")
            alt_cds = self._track_values(scored_df, "alt_cds")
            ref_splice = self._track_values(scored_df, "ref_splice")
            alt_splice = self._track_values(scored_df, "alt_splice")
            if self.score_method == _MASKED_MARGINAL:
                scores = np.array(
                    self.model.masked_marginal_llr(
                        refs,
                        alts,
                        normalization=self.normalization,
                        cds=ref_cds,
                        splice=ref_splice,
                    ),
                    dtype=float,
                )
            else:
                ref_scores: np.ndarray = np.array(
                    self._sequence_scores(
                        refs,
                        ref_cds,
                        ref_splice,
                    ),
                    dtype=float,
                )
                alt_scores: np.ndarray = np.array(
                    self._sequence_scores(
                        alts,
                        alt_cds,
                        alt_splice,
                    ),
                    dtype=float,
                )
                scores = ref_scores - alt_scores

        labels = scored_df[self.target_col].to_numpy()

        metrics: dict[str, float] = {
            "auroc": float(roc_auc_score(labels, scores)),
            "auprc": float(average_precision_score(labels, scores)),
        }

        if persist:
            if self.persister is None:
                raise RuntimeError(
                    "Must provide persister to persist results."
                )
            self.persister.persist_run_results(metrics, self.result_key)

        return metrics

    def _sequence_scores(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None,
        splice: list[np.ndarray] | None,
    ) -> list[float]:
        """Score each unique sequence/track combination once."""
        assert self.model is not None
        unique_sequences: list[str] = []
        unique_cds: list[np.ndarray] | None = (
            [] if cds is not None else None
        )
        unique_splice: list[np.ndarray] | None = (
            [] if splice is not None else None
        )
        key_to_index: dict[
            tuple[str, bytes | None, bytes | None], int
        ] = {}
        sequence_indices: list[int] = []

        for index, sequence in enumerate(sequences):
            cds_track = None if cds is None else cds[index]
            splice_track = None if splice is None else splice[index]
            key = (
                sequence,
                None if cds_track is None else cds_track.tobytes(),
                None if splice_track is None else splice_track.tobytes(),
            )
            if key not in key_to_index:
                key_to_index[key] = len(unique_sequences)
                unique_sequences.append(sequence)
                if unique_cds is not None:
                    assert cds_track is not None
                    unique_cds.append(cds_track)
                if unique_splice is not None:
                    assert splice_track is not None
                    unique_splice.append(splice_track)
            sequence_indices.append(key_to_index[key])

        unique_scores = self.model.sequence_score(
            unique_sequences,
            self.score_method,
            self.normalization,
            cds=unique_cds,
            splice=unique_splice,
        )
        return [unique_scores[index] for index in sequence_indices]

    @staticmethod
    def _track_values(
        dataframe: pd.DataFrame,
        column: str,
    ) -> list[np.ndarray] | None:
        """Return an optional paired sequence track."""
        if column not in dataframe:
            return None
        return dataframe[column].tolist()
