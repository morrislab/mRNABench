from typing import TYPE_CHECKING, Callable
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from mrna_bench.linear_probe.persister import LinearProbePersister
from mrna_bench.metrics import regression_metrics

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
        task: str = "classification",
        likelihood_direction: str | None = None,
    ):
        """Initialize ZeroShotVEP.

        Args:
            data_df: Dataset-specific VEP rows normalized to ref/alt columns.
            target_col: Classification or regression label column.
            persister: Optional persister for writing results to results.db.
            scoring_fn: Callable mapping an (N, D) delta matrix to (N,)
                scalar scores. Defaults to row-wise L2 norm.
            model: Optional likelihood-capable model instead of embeddings.
            score_method: Causal likelihood, pseudo-likelihood, or masked
                marginal scoring.
            normalization: Sum for a likelihood ratio or mean for a
                length-normalized score.
            task: Classification or regression.
            likelihood_direction: Order of the likelihood score difference.
        """
        if target_col not in data_df.columns:
            raise ValueError(f"Target column '{target_col}' not in dataframe.")
        if task not in {"classification", "regression"}:
            raise ValueError("VEP task must be classification or regression.")
        if likelihood_direction is None:
            likelihood_direction = self.default_likelihood_direction(task)
        if likelihood_direction not in {"ref-alt", "alt-ref"}:
            raise ValueError(
                "likelihood_direction must be ref-alt or alt-ref."
            )
        if model is None and task == "regression" and scoring_fn is None:
            raise ValueError(
                "Embedding VEP regression requires a signed scoring_fn; "
                "the default L2 norm discards effect direction."
            )

        self.data_df = data_df.copy()
        self.target_col = target_col
        self.persister = persister
        self.model = model
        self.score_method = score_method
        self.normalization = normalization
        self.task = task
        self.likelihood_direction = likelihood_direction
        if model is None:
            self.result_key = _ZEROSHOT_SEED
        else:
            self.result_key = self.likelihood_result_key(
                score_method,
                normalization,
                getattr(model, "attn_implementation", None) or "none",
                likelihood_direction,
            )
        self.scoring_fn: Callable[[np.ndarray], np.ndarray] = (
            scoring_fn if scoring_fn is not None
            else lambda X: np.linalg.norm(X, axis=1)
        )

    @staticmethod
    def likelihood_result_key(
        score_method: object,
        normalization: str,
        attention: str,
        direction: str,
    ) -> str:
        """Return the persisted key for a likelihood VEP configuration."""
        result_key = "{}-{}-attn-{}".format(
            score_method,
            normalization,
            attention,
        )
        if direction != "ref-alt":
            result_key += "-{}".format(direction)
        return result_key

    @staticmethod
    def default_likelihood_direction(task: str) -> str:
        """Return the score direction matching a VEP task's target."""
        return "alt-ref" if task == "regression" else "ref-alt"

    @staticmethod
    def _resolve_target_and_task(
        metadata: object,
        target_col: str | None,
        task: str | None,
    ) -> tuple[str, str]:
        """Resolve a VEP target and task from dataset metadata."""
        spec = getattr(metadata, "vep_task_spec", None)
        if spec is not None:
            selected_target = target_col or spec.target_col
            selected_task = task or spec.task
            return selected_target, selected_task

        targets = getattr(metadata, "target_col", ["target"])
        tasks = getattr(metadata, "task", ["classification"])
        return target_col or targets[0], task or tasks[0]

    @classmethod
    def from_embeddings(
        cls,
        dataset: "BenchmarkDataset",
        embeddings: np.ndarray,
        target_col: str | None = None,
        task: str | None = None,
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
        target_col, task = cls._resolve_target_and_task(
            dataset.metadata,
            target_col,
            task,
        )
        return cls(
            dataset.get_vep_pairs(data_df, ("embeddings",)),
            target_col,
            task=task,
            **kwargs,
        )

    @classmethod
    def from_model(
        cls,
        dataset: "BenchmarkDataset",
        model: "EmbeddingModel",
        score_method: "ModelBehavior | str | None" = None,
        target_col: str | None = None,
        task: str | None = None,
        normalization: str = "sum",
        likelihood_direction: str | None = None,
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
        target_col, task = cls._resolve_target_and_task(
            dataset.metadata,
            target_col,
            task,
        )
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
            target_col,
            model=model,
            score_method=score_method,
            normalization=normalization,
            task=task,
            likelihood_direction=likelihood_direction,
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
            Classification or regression metrics.
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
            if self.likelihood_direction == "alt-ref":
                scores = -scores

        labels = scored_df[self.target_col].to_numpy()
        if self.task == "regression":
            metrics = regression_metrics(labels, scores)
        else:
            metrics = {
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
