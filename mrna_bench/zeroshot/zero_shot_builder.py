from collections.abc import Callable
from typing import Self

import numpy as np

from mrna_bench import load_dataset
from mrna_bench.datasets import BenchmarkDataset, EvaluationMethod
from mrna_bench.embedder import EmbeddingFetcher
from mrna_bench.models import EmbeddingModel, ModelBehavior
from mrna_bench.persister import ResultsPersister
from mrna_bench.zeroshot.evaluator import ZeroShotEvaluator
from mrna_bench.zeroshot.scorers import (
    Direction,
    EmbeddingDeltaScorer,
    LikelihoodRatioScorer,
    ScoreMethod,
)
from mrna_bench.zeroshot.zero_shot import NO_SPLIT, ZeroShot


class ZeroShotBuilder(EmbeddingFetcher):
    """Factory class for ZeroShot.

    Builds variant effect prediction from either the embeddings of each
    variant's alleles or a model's likelihood of them.
    """

    def __init__(
        self,
        dataset: BenchmarkDataset | None = None,
        dataset_name: str | None = None,
    ):
        """Initialize ZeroShotBuilder.

        Can be initialized with BenchmarkDataset instance or name. Only one
        of dataset or dataset_name should be provided. The target and task
        default to the dataset's VEP target.

        Recommended build order:
            1. Fetch embeddings, or set a likelihood model
            2. (Optional) set target column and evaluator task
            3. (Optional) specify if persister should be used
            4. Build ZeroShot instance.

        Args:
            dataset: BenchmarkDataset to evaluate.
            dataset_name: Name of dataset to evaluate.
        """
        if dataset is None and dataset_name is None:
            raise ValueError("Must provide dataset or dataset name.")
        elif dataset is not None and dataset_name is not None:
            raise ValueError("Provide only one of dataset or dataset name.")

        if dataset is None and dataset_name is not None:
            dataset = load_dataset(dataset_name)

        assert dataset is not None

        vep_spec = dataset.metadata.vep_task_spec
        if vep_spec is None:
            raise ValueError(
                "{} has no zero-shot evaluation.".format(dataset.dataset_name)
            )

        self.dataset = dataset
        self.target_col = vep_spec.target_col
        self.task = vep_spec.task

        self.model_short_name: str | None = None
        self.embeddings: np.ndarray | None = None
        self.embedding_config: dict | None = None
        self.scoring_fn: Callable[[np.ndarray], np.ndarray] | None = None

        self.model: EmbeddingModel | None = None
        self.score_method: ScoreMethod | None = None
        self.normalization = "sum"
        self.direction: Direction | None = None

        self.persister_flag = False

    def fetch_embedding_by_embedding_instance(
        self,
        model_short_name: str,
        embeddings: np.ndarray,
        embedding_config: dict | None = None,
    ) -> Self:
        """Store embeddings for ZeroShot using an embedding instance.

        Args:
            model_short_name: Short name of model used to generate embeddings.
            embeddings: Embeddings of the dataset rows.
            embedding_config: Settings the embeddings were computed with,
                see get_embedding_config. Required to persist results.

        Returns:
            ZeroShotBuilder with set embeddings.
        """
        evaluations = self.dataset.metadata.evaluations
        if EvaluationMethod.EMBEDDING_VEP not in evaluations:
            raise ValueError(
                "{} does not support embedding VEP.".format(
                    self.dataset.dataset_name
                )
            )
        return super().fetch_embedding_by_embedding_instance(
            model_short_name, embeddings, embedding_config
        )

    def set_scoring_fn(
        self,
        scoring_fn: Callable[[np.ndarray], np.ndarray],
    ) -> "ZeroShotBuilder":
        """Set the score of the difference of alt and ref embeddings.

        Args:
            scoring_fn: Function mapping an (N, D) matrix of alt minus ref
                embeddings to (N,) scores. Regression targets need a signed
                score.

        Returns:
            ZeroShotBuilder with set scoring function.
        """
        self.scoring_fn = scoring_fn
        return self

    def use_likelihood(
        self,
        model: EmbeddingModel,
        score_method: ScoreMethod | str | None = None,
        normalization: str = "sum",
        direction: Direction | str | None = None,
    ) -> "ZeroShotBuilder":
        """Score variants with a model's likelihood of their alleles.

        Args:
            model: Model with a causal or pseudo-likelihood.
            score_method: Likelihood score. Defaults to the model's only
                likelihood.
            normalization: Sum for a likelihood ratio or mean for a
                length-normalized score.
            direction: Order of the likelihood difference. Defaults to
                alt minus ref for regression and ref minus alt otherwise.

        Returns:
            ZeroShotBuilder set to score with the model.
        """
        compatible = self.dataset.metadata.compatible_evaluations(model)
        if EvaluationMethod.LIKELIHOOD_VEP not in compatible:
            raise ValueError(
                "{} cannot run likelihood VEP on {}.".format(
                    model.__class__.__name__, self.dataset.dataset_name
                )
            )

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
            score_method = next(iter(supported)).value
        score_method = ScoreMethod(score_method)

        if score_method == ScoreMethod.MASKED_MARGINAL:
            if ModelBehavior.PSEUDO_LIKELIHOOD not in supported:
                raise ValueError(
                    "masked_marginal requires pseudo_likelihood support."
                )
        elif ModelBehavior(score_method) not in supported:
            raise ValueError("{} does not support {}.".format(
                model.__class__.__name__, score_method
            ))

        self.model = model
        self.model_short_name = model.short_name
        self.score_method = score_method
        self.normalization = normalization
        self.direction = None if direction is None else Direction(direction)
        return self

    def set_target(self, target_col: str) -> "ZeroShotBuilder":
        """Set the target column.

        Args:
            target_col: Column from dataframe to use as labels.

        Returns:
            ZeroShotBuilder with set target column.
        """
        self.target_col = target_col
        return self

    def build_evaluator(self, task: str) -> "ZeroShotBuilder":
        """Set the evaluator task.

        Args:
            task: Classification or regression.

        Returns:
            ZeroShotBuilder with set task.
        """
        self.task = task
        return self

    def use_persister(self) -> "ZeroShotBuilder":
        """Indicate that persister for ZeroShot should be built.

        Returns:
            ZeroShotBuilder with persister flag set.
        """
        self.persister_flag = True
        return self

    def validate(self) -> list[str]:
        """Return a list of missing required fields before build."""
        missing = []
        if self.embeddings is None and self.model is None:
            missing.append(
                "embeddings or model (call fetch_embedding_by_* or "
                "use_likelihood)"
            )
        if self.target_col not in self.dataset.data_df.columns:
            missing.append(
                "target_col '{}' (not found in dataset columns)".format(
                    self.target_col
                )
            )
        needs_config = self.model is None and self.embedding_config is None
        if self.persister_flag and needs_config:
            missing.append(
                "embedding_config (required when persister is enabled)"
            )
        return missing

    def build(self) -> ZeroShot:
        """Build a ZeroShot instance.

        Returns:
            Configured zero-shot evaluation ready to run.
        """
        missing = self.validate()
        if len(missing) > 0:
            raise ValueError(
                "Cannot build ZeroShot; missing configuration: {}".format(
                    "; ".join(missing)
                )
            )

        scorer: EmbeddingDeltaScorer | LikelihoodRatioScorer
        embedding_config: dict | None
        if self.model is not None:
            assert self.score_method is not None
            default = LikelihoodRatioScorer.default_direction(self.task)
            scorer = LikelihoodRatioScorer(
                self.model,
                self.score_method,
                self.normalization,
                self.direction or default,
            )
            pairs_df = self.dataset.get_vep_pairs(self.dataset.data_df)
            attn = self.model.attn_implementation
            embedding_config = {
                "attn_implementation": (
                    attn or self.model.default_attn_implementation
                ),
            }
        else:
            assert self.embeddings is not None
            scorer = EmbeddingDeltaScorer(self.scoring_fn)
            data_df = self.dataset.data_df.copy()
            data_df["embeddings"] = list(self.embeddings)
            pairs_df = self.dataset.get_vep_pairs(data_df, ("embeddings",))
            embedding_config = self.embedding_config

        persister = None
        if self.persister_flag:
            assert self.model_short_name is not None
            assert embedding_config is not None
            persister = ResultsPersister(
                self.dataset,
                self.model_short_name,
                scorer.evaluation,
                self.task,
                self.target_col,
                NO_SPLIT,
                embedding_config,
                scorer.probe_config,
            )

        return ZeroShot(
            pairs_df,
            self.target_col,
            scorer,
            ZeroShotEvaluator(self.task),
            persister,
        )
