import pytest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from mrna_bench.datasets import EvaluationMethod
from mrna_bench.datasets.utr_variants_bohn import UTRVariantsBohn
from mrna_bench.datasets.vep_traitgym import VEPTraitGym
from mrna_bench.models import ModelBehavior
from mrna_bench.persister import ResultsPersister
from mrna_bench.zeroshot import (
    EmbeddingDeltaScorer,
    ZeroShot,
    ZeroShotBuilder,
    ZeroShotEvaluator,
)


class RowDataset(SimpleNamespace):
    get_vep_pairs = UTRVariantsBohn.get_vep_pairs


class PairedDataset(SimpleNamespace):
    def get_vep_pairs(self, dataframe, value_columns=None):
        return dataframe.copy()


class ScoringModel:
    """Model stub scoring sequences by their G count."""

    short_name = "scorer"
    attn_implementation = None
    default_attn_implementation = None
    behaviors = {ModelBehavior.PSEUDO_LIKELIHOOD}

    def __init__(self):
        self.calls = []

    def sequence_score(self, sequences, method, normalization, cds=None,
                       splice=None):
        assert method == ModelBehavior.PSEUDO_LIKELIHOOD
        self.calls.append((normalization, cds, splice))
        return [sequence.count("G") for sequence in sequences]

    def masked_marginal_llr(self, references, alternates, **kwargs):
        self.calls.append(kwargs)
        return [
            reference.count("G") - alternate.count("G")
            for reference, alternate in zip(references, alternates)
        ]


def vep_metadata(task="classification", target_col="target"):
    """Metadata stub of a dataset with a VEP evaluation."""
    return SimpleNamespace(
        vep_task_spec=SimpleNamespace(task=task, target_col=target_col),
        evaluations=(EvaluationMethod.EMBEDDING_VEP,),
        compatible_evaluations=lambda model: (
            EvaluationMethod.LIKELIHOOD_VEP,
        ),
    )


@pytest.fixture
def row_dataset():
    """Dataset with wild-type and variant rows of two transcripts."""
    dataframe = pd.DataFrame({
        "transcript_id": ["tx1", "tx1", "tx2", "tx2"],
        "description": [
            "wild-type",
            "chr1:1 A:C",
            "wild-type",
            "chr2:1 A:C",
        ],
        "sequence": ["AAA", "CAA", "GGG", "GGC"],
        "cds": [np.zeros(3)] * 4,
        "splice": [np.zeros(3)] * 4,
        "target": [0, 1, 0, 0],
    })
    return RowDataset(
        data_df=dataframe, dataset_name="vep", metadata=vep_metadata()
    )


@pytest.mark.parametrize(
    "get_pairs",
    [UTRVariantsBohn.get_vep_pairs, VEPTraitGym.get_vep_pairs],
)
def test_get_vep_pairs(get_pairs):
    """Pair row-wise sequence and track columns with each wild type."""
    ref_cds = np.array([1, 0, 0])
    alt_cds = np.array([1, 0, 0])
    dataframe = pd.DataFrame({
        "transcript_id": ["tx1", "tx1"],
        "description": ["wild-type", "chr1:1 A:C"],
        "sequence": ["AAA", "CAA"],
        "cds": [ref_cds, alt_cds],
        "target": [0, 1],
    })

    dataset = SimpleNamespace(dataset_name="vep")
    paired = get_pairs(dataset, dataframe, ("sequence", "cds"))

    assert paired["ref_sequence"].tolist() == ["AAA"]
    assert paired["alt_sequence"].tolist() == ["CAA"]
    np.testing.assert_array_equal(paired.iloc[0]["ref_cds"], ref_cds)
    np.testing.assert_array_equal(paired.iloc[0]["alt_cds"], alt_cds)

    with pytest.raises(
        ValueError,
        match="Missing wild-type",
    ):
        get_pairs(dataset, dataframe.iloc[1:], ("sequence", "cds"))


def test_embedding_delta_scores_variants(row_dataset):
    """Embedding VEP scores variants by the norm of the allele difference."""
    embeddings = np.array([
        [0.0, 0.0],
        [3.0, 4.0],
        [0.0, 0.0],
        [0.0, 1.0],
    ])

    metrics = (
        ZeroShotBuilder(row_dataset)
        .fetch_embedding_by_embedding_instance("model", embeddings)
        .build()
        .run_zero_shot()
    )

    assert metrics == {"auroc": 1.0, "auprc": 1.0, "prevalence": 0.5}


def test_regression_uses_custom_signed_score():
    """Continuous effects use a caller-selected signed embedding score."""
    dataframe = pd.DataFrame({
        "ref_embeddings": [np.array([0.0, 0.0])] * 3,
        "alt_embeddings": [
            np.array([-1.0, 4.0]),
            np.array([0.0, 3.0]),
            np.array([2.0, 2.0]),
        ],
        "effect": [-1.0, 0.0, 2.0],
    })

    zero_shot = ZeroShot(
        dataframe,
        "effect",
        EmbeddingDeltaScorer(lambda delta: delta[:, 0]),
        ZeroShotEvaluator("regression"),
    )

    assert zero_shot.run_zero_shot() == {"mse": 0.0, "r": 1.0, "p": 1.0}


def test_regression_rejects_unsigned_default():
    """Regression requires an explicit signed embedding score."""
    dataframe = pd.DataFrame({
        "ref_embeddings": [np.array([0.0])],
        "alt_embeddings": [np.array([1.0])],
        "effect": [1.0],
    })

    with pytest.raises(ValueError, match="signed scoring_fn"):
        ZeroShot(
            dataframe,
            "effect",
            EmbeddingDeltaScorer(),
            ZeroShotEvaluator("regression"),
        )


def test_builder_uses_dataset_vep_target():
    """The builder takes its target and task from the VEP task spec."""
    dataset = PairedDataset(
        data_df=pd.DataFrame({"absolute": [0.1, 0.2], "effect": [-0.4, 0.4]}),
        dataset_name="vep-regression",
        metadata=vep_metadata(task="regression", target_col="effect"),
    )

    zero_shot = (
        ZeroShotBuilder(dataset)
        .fetch_embedding_by_embedding_instance("model", np.zeros((2, 1)))
        .set_scoring_fn(lambda delta: delta[:, 0])
        .build()
    )

    assert zero_shot.target_col == "effect"
    assert zero_shot.evaluator.task == "regression"


def test_likelihood_scores_each_allele_once():
    """Likelihood VEP scores unique sequences with their own tracks."""
    df = pd.DataFrame({
        "ref_sequence": ["GGGG", "AAAA", "GGGA", "AAAG"],
        "alt_sequence": ["AAAA", "GGGG", "AAAA", "GGGG"],
        "ref_cds": [np.zeros(4)] * 4,
        "alt_cds": [np.ones(4)] * 4,
        "ref_splice": [np.arange(4)] * 4,
        "alt_splice": [np.arange(4)] * 4,
        "target": [1, 0, 1, 0],
    })
    dataset = PairedDataset(
        data_df=df, dataset_name="vep", metadata=vep_metadata()
    )
    model = ScoringModel()

    zero_shot = ZeroShotBuilder(dataset).use_likelihood(model).build()
    metrics = zero_shot.run_zero_shot()

    assert metrics == {"auroc": 1.0, "auprc": 1.0, "prevalence": 0.5}
    (ref_norm, ref_cds, ref_splice), (_, alt_cds, _) = model.calls
    assert ref_norm == "sum"
    assert len(ref_cds) == 4 and len(alt_cds) == 2
    assert ref_cds[0].sum() == 0 and alt_cds[0].sum() == 4
    assert ref_splice[0].tolist() == [0, 1, 2, 3]
    assert zero_shot.scorer.probe_config == {
        "score_method": "pseudo_likelihood",
        "normalization": "sum",
        "direction": "ref-alt",
    }


def test_likelihood_regression_scores_alt_minus_ref():
    """Regression orients likelihood scores to match signed effects."""
    dataset = PairedDataset(
        data_df=pd.DataFrame({
            "ref_sequence": ["AAAA", "AAAA", "AAAA"],
            "alt_sequence": ["AAAG", "AAGG", "AGGG"],
            "effect": [1.0, 2.0, 3.0],
        }),
        dataset_name="vep-regression",
        metadata=vep_metadata(task="regression", target_col="effect"),
    )

    zero_shot = ZeroShotBuilder(dataset).use_likelihood(
        ScoringModel(), score_method="pseudo_likelihood"
    ).build()

    assert zero_shot.run_zero_shot()["mse"] == 0.0
    assert zero_shot.scorer.direction == "alt-ref"


def test_likelihood_pairs_row_schema(row_dataset):
    """Datasets with one row per allele are paired before scoring."""
    zero_shot = (
        ZeroShotBuilder(row_dataset).use_likelihood(ScoringModel()).build()
    )

    assert {"ref_sequence", "alt_sequence"} <= set(zero_shot.data_df)


def test_masked_marginal_scores_substitutions():
    """Masked-marginal VEP passes reference tracks to the model."""
    df = pd.DataFrame({
        "ref_sequence": ["GGGG", "AAAA", "GGGA", "AAAG"],
        "alt_sequence": ["AAAA", "GGGG", "AAAA", "GGGG"],
        "ref_cds": [np.zeros(4)] * 4,
        "target": [1, 0, 1, 0],
    })
    dataset = PairedDataset(
        data_df=df, dataset_name="vep", metadata=vep_metadata()
    )
    model = ScoringModel()

    metrics = ZeroShotBuilder(dataset).use_likelihood(
        model, score_method="masked_marginal"
    ).build().run_zero_shot()

    assert metrics == {"auroc": 1.0, "auprc": 1.0, "prevalence": 0.5}
    assert model.calls[0]["cds"][0].sum() == 0


def test_masked_marginal_requires_pseudo_likelihood(row_dataset):
    """Masked-marginal scores need a pseudo-likelihood model."""
    model = ScoringModel()
    model.behaviors = {ModelBehavior.CAUSAL_LIKELIHOOD}

    with pytest.raises(ValueError, match="requires pseudo_likelihood"):
        ZeroShotBuilder(row_dataset).use_likelihood(
            model, score_method="masked_marginal"
        )


def test_persisted_run_records_scorer(row_dataset, tmp_path):
    """Persisted runs name the evaluation and scorer settings."""
    row_dataset.dataset_path = str(tmp_path)
    zero_shot = (
        ZeroShotBuilder(row_dataset)
        .use_likelihood(ScoringModel())
        .use_persister()
        .build()
    )
    assert not zero_shot.result_exists()

    zero_shot.run_zero_shot(persist=True)

    assert zero_shot.result_exists()
    run, = ResultsPersister.load_all_results(tmp_path)
    assert run["evaluation"] == "likelihood_vep"
    assert run["probe_config"]["score_method"] == "pseudo_likelihood"
    assert run["embedding_config"] == {"attn_implementation": None}
    assert set(run["metrics"]["all"]) == {"auroc", "auprc", "prevalence"}
