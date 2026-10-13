import pandas as pd

from mrna_bench.persister import ResultsPersister
from mrna_bench.zeroshot.evaluator import ZeroShotEvaluator
from mrna_bench.zeroshot.scorers import (
    EmbeddingDeltaScorer,
    LikelihoodRatioScorer,
)

# zero-shot evaluation has no data split, so results use this seed and split
# type
NO_SPLIT = "none"


class ZeroShot:
    """Zero-shot evaluation.

    Scores every row with a scorer that is not fit to the data, e.g. from
    the embeddings or model likelihood of a variant's reference and
    alternate allele, and evaluates the scores against the targets. With
    nothing fit, all rows are evaluated as one split.
    """

    def __init__(
        self,
        data_df: pd.DataFrame,
        target_col: str,
        scorer: EmbeddingDeltaScorer | LikelihoodRatioScorer,
        evaluator: ZeroShotEvaluator,
        persister: ResultsPersister | None = None,
    ):
        """Initialize ZeroShot.

        Args:
            data_df: Rows to score, with the columns the scorer reads, e.g.
                ref- and alt-prefixed columns from
                BenchmarkDataset.get_vep_pairs.
            target_col: Column with labels or target values.
            scorer: Scorer for the rows.
            evaluator: Evaluator for the scores.
            persister: Persister for results. If not provided, results will
                not be saved.
        """
        if target_col not in data_df.columns:
            raise ValueError(
                "Target column '{}' not in dataframe.".format(target_col)
            )
        if evaluator.task == "regression" and not scorer.signed:
            raise ValueError(
                "Embedding VEP regression requires a signed scoring_fn; "
                "the default L2 norm discards effect direction. Use "
                "LinearProbeBuilder with the dataset's VEP target for a "
                "supervised signed delta-embedding probe."
            )

        self.data_df = data_df
        self.target_col = target_col
        self.scorer = scorer
        self.evaluator = evaluator
        self.persister = persister

    def result_exists(self) -> bool:
        """Check whether a result is already persisted.

        Returns:
            True if the persister has a result for this configuration.
        """
        if self.persister is None:
            return False
        else:
            return self.persister.result_exists(NO_SPLIT)

    def run_zero_shot(self, persist: bool = False) -> dict[str, float]:
        """Score rows and evaluate the scores.

        Args:
            persist: Save results to disk.

        Returns:
            Classification or regression metrics.
        """
        data_df = self.data_df.dropna(subset=[self.target_col])
        data_df = self.scorer.select_rows(data_df)
        scores = self.scorer.score(data_df)

        metrics = self.evaluator.evaluate_zero_shot(
            data_df[self.target_col].to_numpy(), scores
        )

        if self.persister and persist:
            self.persister.persist_run_results({"all": metrics}, NO_SPLIT)
        elif persist and not self.persister:
            raise RuntimeError("Must provide persister to persist results.")

        return metrics
