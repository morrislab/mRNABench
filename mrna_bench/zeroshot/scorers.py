from collections.abc import Callable
from enum import StrEnum
import warnings

import numpy as np
import pandas as pd

from mrna_bench.models import EmbeddingModel, ModelBehavior


class ScoreMethod(StrEnum):
    """Likelihood scores of a variant."""

    CAUSAL_LIKELIHOOD = "causal_likelihood"
    PSEUDO_LIKELIHOOD = "pseudo_likelihood"
    MASKED_MARGINAL = "masked_marginal"


class Direction(StrEnum):
    """Order of the likelihood difference of a variant."""

    REF_ALT = "ref-alt"
    ALT_REF = "alt-ref"


class EmbeddingDeltaScorer:
    """Scores variants from the difference of alt and ref embeddings."""

    evaluation = "embedding_vep"

    def __init__(
        self,
        scoring_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        """Initialize EmbeddingDeltaScorer.

        Args:
            scoring_fn: Function mapping an (N, D) matrix of alt minus ref
                embeddings to (N,) scores. Defaults to the row-wise L2 norm,
                which has no sign.
        """
        self.scoring_fn = scoring_fn

    @staticmethod
    def get_probe_config(
        scoring_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> dict:
        """Get the probe config persisted with results.

        Args:
            scoring_fn: Scoring function, None for the L2 norm.

        Returns:
            Config naming the score.
        """
        if scoring_fn is None:
            return {"score": "l2_norm"}
        else:
            return {"score": getattr(scoring_fn, "__name__", repr(scoring_fn))}

    @property
    def probe_config(self) -> dict:
        """Probe config of this scorer."""
        return self.get_probe_config(self.scoring_fn)

    @property
    def signed(self) -> bool:
        """Whether scores have a sign, as regression targets need."""
        return self.scoring_fn is not None

    def select_rows(self, pairs_df: pd.DataFrame) -> pd.DataFrame:
        """Return the rows this scorer can score (all of them)."""
        return pairs_df

    def score(self, pairs_df: pd.DataFrame) -> np.ndarray:
        """Score variants.

        Args:
            pairs_df: Variants with "ref_embeddings" and "alt_embeddings".

        Returns:
            One score per variant.
        """
        delta = np.subtract(
            np.stack(pairs_df["alt_embeddings"]),
            np.stack(pairs_df["ref_embeddings"]),
        )
        if self.scoring_fn is None:
            return np.linalg.norm(delta, axis=1)
        else:
            return self.scoring_fn(delta)


class LikelihoodRatioScorer:
    """Scores variants from the model likelihood of ref and alt sequences."""

    evaluation = "likelihood_vep"

    def __init__(
        self,
        model: EmbeddingModel,
        score_method: ScoreMethod,
        normalization: str = "sum",
        direction: Direction = Direction.REF_ALT,
    ):
        """Initialize LikelihoodRatioScorer.

        Args:
            model: Model with a causal or pseudo-likelihood.
            score_method: Likelihood score to use.
            normalization: Sum for a likelihood ratio or mean for a
                length-normalized score.
            direction: Ref minus alt, which is high for damaging variants,
                or alt minus ref, to match signed effects.
        """
        self.model = model
        self.score_method = score_method
        self.normalization = normalization
        self.direction = direction

    @staticmethod
    def default_direction(task: str) -> Direction:
        """Get the direction that matches the labels of a VEP task.

        Args:
            task: Classification or regression.

        Returns:
            Alt minus ref for signed effects, ref minus alt otherwise.
        """
        if task == "regression":
            return Direction.ALT_REF
        else:
            return Direction.REF_ALT

    @staticmethod
    def get_probe_config(
        score_method: ScoreMethod,
        normalization: str,
        direction: Direction,
    ) -> dict:
        """Get the probe config persisted with results.

        Args:
            score_method: Likelihood score.
            normalization: Sum or mean of token log-probabilities.
            direction: Order of the likelihood difference.

        Returns:
            Config of the likelihood score.
        """
        return {
            "score_method": str(score_method),
            "normalization": normalization,
            "direction": str(direction),
        }

    @property
    def probe_config(self) -> dict:
        """Probe config of this scorer."""
        return self.get_probe_config(
            self.score_method, self.normalization, self.direction
        )

    # likelihood differences always have a sign
    signed = True

    def select_rows(self, pairs_df: pd.DataFrame) -> pd.DataFrame:
        """Return the rows this scorer can score.

        Masked marginal scores compare aligned tokens, so they score only
        substitutions.

        Args:
            pairs_df: Variants with "ref_sequence" and "alt_sequence".

        Returns:
            Variants that can be scored.
        """
        if self.score_method != ScoreMethod.MASKED_MARGINAL:
            return pairs_df

        substitutions = pairs_df["ref_sequence"].str.len().eq(
            pairs_df["alt_sequence"].str.len()
        )
        if not substitutions.all():
            warnings.warn(
                "Masked-marginal VEP excludes {} indel variants.".format(
                    int((~substitutions).sum())
                ),
                RuntimeWarning,
            )
        if not substitutions.any():
            raise ValueError(
                "Masked-marginal VEP requires substitution variants."
            )
        return pairs_df[substitutions]

    def score(self, pairs_df: pd.DataFrame) -> np.ndarray:
        """Score variants.

        Args:
            pairs_df: Variants with "ref_sequence" and "alt_sequence" and
                optional "ref_cds", "alt_cds", "ref_splice", "alt_splice".

        Returns:
            One score per variant.
        """
        refs = pairs_df["ref_sequence"].tolist()
        alts = pairs_df["alt_sequence"].tolist()
        ref_cds = self._track_values(pairs_df, "ref_cds")
        ref_splice = self._track_values(pairs_df, "ref_splice")

        if self.score_method == ScoreMethod.MASKED_MARGINAL:
            scores = np.array(self.model.masked_marginal_llr(
                refs,
                alts,
                normalization=self.normalization,
                cds=ref_cds,
                splice=ref_splice,
            ), dtype=float)
        else:
            ref_scores = self._sequence_scores(refs, ref_cds, ref_splice)
            alt_scores = self._sequence_scores(
                alts,
                self._track_values(pairs_df, "alt_cds"),
                self._track_values(pairs_df, "alt_splice"),
            )
            scores = ref_scores - alt_scores

        if self.direction == Direction.ALT_REF:
            return -scores
        else:
            return scores

    def _sequence_scores(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None,
        splice: list[np.ndarray] | None,
    ) -> np.ndarray:
        """Score each unique sequence and track combination once.

        Args:
            sequences: Sequences to score.
            cds: Optional CDS tracks aligned to sequences.
            splice: Optional splice tracks aligned to sequences.

        Returns:
            One likelihood score per sequence.
        """
        unique_index: dict[tuple, int] = {}
        first: list[int] = []
        indices = []
        for i, sequence in enumerate(sequences):
            key = (
                sequence,
                None if cds is None else cds[i].tobytes(),
                None if splice is None else splice[i].tobytes(),
            )
            if key not in unique_index:
                unique_index[key] = len(first)
                first.append(i)
            indices.append(unique_index[key])

        unique_scores = self.model.sequence_score(
            [sequences[i] for i in first],
            ModelBehavior(self.score_method),
            self.normalization,
            cds=None if cds is None else [cds[i] for i in first],
            splice=None if splice is None else [splice[i] for i in first],
        )
        return np.array(unique_scores, dtype=float)[indices]

    @staticmethod
    def _track_values(
        pairs_df: pd.DataFrame,
        column: str,
    ) -> list[np.ndarray] | None:
        """Get an optional track column as a list."""
        if column not in pairs_df:
            return None
        else:
            return pairs_df[column].tolist()
