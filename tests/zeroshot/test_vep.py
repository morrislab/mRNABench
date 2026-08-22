import pytest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from mrna_bench.datasets import EvaluationMethod
from mrna_bench.datasets.utr_variants_bohn import UTRVariantsBohn
from mrna_bench.datasets.vep_traitgym import VEPTraitGym
from mrna_bench.models import ModelBehavior
from mrna_bench.zeroshot import ZeroShotVEP


class RowDataset(SimpleNamespace):
    get_vep_pairs = UTRVariantsBohn.get_vep_pairs


class PairedDataset(SimpleNamespace):
    def get_vep_pairs(self, dataframe, value_columns=None):
        return dataframe.copy()


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


def test_zeroshot_vep_from_embeddings():
    dataframe = pd.DataFrame({
        "transcript_id": ["tx1", "tx1", "tx2", "tx2"],
        "description": [
            "wild-type",
            "chr1:1 A:C",
            "wild-type",
            "chr2:1 A:C",
        ],
        "target": [0, 1, 0, 0],
    })
    dataset = RowDataset(
        data_df=dataframe,
        dataset_name="vep",
        metadata=SimpleNamespace(
            target_col=["target"],
            evaluations=(EvaluationMethod.EMBEDDING_VEP,),
        ),
    )
    embeddings = np.array([
        [0.0, 0.0],
        [3.0, 4.0],
        [0.0, 0.0],
        [0.0, 1.0],
    ])

    metrics = ZeroShotVEP.from_embeddings(dataset, embeddings).run()

    assert metrics == {"auroc": 1.0, "auprc": 1.0}


def test_zeroshot_vep_regression_uses_custom_signed_score():
    """Continuous effects use a caller-selected signed embedding score."""
    dataframe = pd.DataFrame({
        "ref_embeddings": [
            np.array([0.0, 0.0]),
            np.array([0.0, 0.0]),
            np.array([0.0, 0.0]),
        ],
        "alt_embeddings": [
            np.array([-1.0, 4.0]),
            np.array([0.0, 3.0]),
            np.array([2.0, 2.0]),
        ],
        "effect": [-1.0, 0.0, 2.0],
    })

    evaluator = ZeroShotVEP(
        dataframe,
        "effect",
        task="regression",
        scoring_fn=lambda delta: delta[:, 0],
    )

    assert evaluator.run() == {"mse": 0.0, "r": 1.0, "p": 1.0}
    assert evaluator.result_key == "embedding_vep"


def test_zeroshot_vep_regression_rejects_unsigned_default():
    """Regression requires an explicit signed embedding score."""
    dataframe = pd.DataFrame({
        "ref_embeddings": [np.array([0.0])],
        "alt_embeddings": [np.array([1.0])],
        "effect": [1.0],
    })

    with pytest.raises(ValueError, match="signed scoring_fn"):
        ZeroShotVEP(dataframe, "effect", task="regression")


def test_zeroshot_vep_uses_distinct_metadata_target():
    """VEP constructors resolve their task independently of LP targets."""
    dataframe = pd.DataFrame({
        "absolute": [0.1, 0.2],
        "effect": [-0.4, 0.4],
    })
    dataset = PairedDataset(
        data_df=dataframe,
        dataset_name="vep-regression",
        metadata=SimpleNamespace(
            evaluations=(EvaluationMethod.EMBEDDING_VEP,),
            vep_task_spec=SimpleNamespace(
                task="regression",
                target_col="effect",
            ),
        ),
    )
    embeddings = np.array([[-0.4], [0.4]])

    evaluator = ZeroShotVEP.from_embeddings(
        dataset,
        embeddings,
        scoring_fn=lambda delta: delta[:, 0],
    )

    assert evaluator.target_col == "effect"
    assert evaluator.task == "regression"


def test_zeroshot_vep_from_sequence_scores():
    class ScoringModel:
        calls = []
        behaviors = {ModelBehavior.PSEUDO_LIKELIHOOD}

        def sequence_score(
            self,
            sequences,
            method,
            normalization,
            cds=None,
            splice=None,
        ):
            assert method == ModelBehavior.PSEUDO_LIKELIHOOD
            assert normalization == "sum"
            self.calls.append((cds, splice))
            return [sequence.count("G") for sequence in sequences]

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
        data_df=df,
        dataset_name="vep",
        metadata=SimpleNamespace(
            target_col=["target"],
            compatible_evaluations=lambda model: (
                EvaluationMethod.LIKELIHOOD_VEP,
            ),
        ),
    )
    model = ScoringModel()
    evaluator = ZeroShotVEP.from_model(dataset, model)
    metrics = evaluator.run()

    assert metrics["auroc"] == 1.0
    assert metrics["auprc"] == 1.0
    assert model.calls[0][0] is not None
    assert model.calls[0][0][0].sum() == 0
    assert model.calls[1][0][0].sum() == 4
    assert model.calls[0][1][0].tolist() == [0, 1, 2, 3]
    assert evaluator.result_key == "pseudo_likelihood-sum-attn-none"


def test_likelihood_vep_supports_alt_minus_ref_direction():
    """Datasets can orient likelihood scores to match signed effects."""
    class ScoringModel:
        behaviors = {ModelBehavior.PSEUDO_LIKELIHOOD}

        def sequence_score(
            self,
            sequences,
            method,
            normalization,
            cds=None,
            splice=None,
        ):
            return [sequence.count("G") for sequence in sequences]

    dataframe = pd.DataFrame({
        "ref_sequence": ["AAAA", "AAAA", "AAAA"],
        "alt_sequence": ["AAAG", "AAGG", "AGGG"],
        "effect": [1.0, 2.0, 3.0],
    })
    dataset = PairedDataset(
        data_df=dataframe,
        dataset_name="vep-regression",
        metadata=SimpleNamespace(
            compatible_evaluations=lambda model: (
                EvaluationMethod.LIKELIHOOD_VEP,
            ),
            vep_task_spec=SimpleNamespace(
                task="regression",
                target_col="effect",
            ),
        ),
    )

    evaluator = ZeroShotVEP.from_model(
        dataset,
        ScoringModel(),
        score_method="pseudo_likelihood",
    )
    direct = ZeroShotVEP(
        dataframe,
        "effect",
        model=ScoringModel(),
        score_method="pseudo_likelihood",
        task="regression",
    )

    assert evaluator.run()["mse"] == 0.0
    assert evaluator.result_key.endswith("-alt-ref")
    assert direct.likelihood_direction == "alt-ref"


def test_zeroshot_vep_from_model_pairs_row_schema():
    class ScoringModel:
        behaviors = {ModelBehavior.PSEUDO_LIKELIHOOD}

    dataframe = pd.DataFrame({
        "transcript_id": ["tx1", "tx1"],
        "description": ["wild-type", "chr1:1 A:C"],
        "sequence": ["AAA", "CAA"],
        "cds": [np.zeros(3), np.zeros(3)],
        "splice": [np.zeros(3), np.zeros(3)],
        "target": [0, 1],
    })
    dataset = RowDataset(
        data_df=dataframe,
        dataset_name="vep",
        metadata=SimpleNamespace(
            target_col=["target"],
            compatible_evaluations=lambda model: (
                EvaluationMethod.LIKELIHOOD_VEP,
            ),
        ),
    )

    evaluator = ZeroShotVEP.from_model(dataset, ScoringModel())

    assert {"ref_sequence", "alt_sequence"} <= set(evaluator.data_df)


def test_zeroshot_vep_from_masked_marginal_scores():
    class ScoringModel:
        calls = []
        behaviors = {ModelBehavior.PSEUDO_LIKELIHOOD}

        def masked_marginal_llr(
            self,
            references,
            alternates,
            **kwargs,
        ):
            self.calls.append(kwargs)
            return [
                reference.count("G") - alternate.count("G")
                for reference, alternate in zip(references, alternates)
            ]

    df = pd.DataFrame({
        "ref_sequence": ["GGGG", "AAAA", "GGGA", "AAAG"],
        "alt_sequence": ["AAAA", "GGGG", "AAAA", "GGGG"],
        "ref_cds": [np.zeros(4)] * 4,
        "target": [1, 0, 1, 0],
    })
    dataset = PairedDataset(
        data_df=df,
        dataset_name="vep",
        metadata=SimpleNamespace(
            target_col=["target"],
            compatible_evaluations=lambda model: (
                EvaluationMethod.LIKELIHOOD_VEP,
            ),
        ),
    )
    model = ScoringModel()

    metrics = ZeroShotVEP.from_model(
        dataset, model, score_method="masked_marginal"
    ).run()

    assert metrics == {"auroc": 1.0, "auprc": 1.0}
    assert model.calls[0]["cds"][0].sum() == 0


def test_masked_marginal_requires_pseudo_likelihood():
    model = SimpleNamespace(
        behaviors={ModelBehavior.CAUSAL_LIKELIHOOD},
    )
    dataset = PairedDataset(
        data_df=pd.DataFrame({
            "ref_sequence": ["AAA"],
            "alt_sequence": ["CAA"],
            "target": [1],
        }),
        dataset_name="vep",
        metadata=SimpleNamespace(
            target_col=["target"],
            compatible_evaluations=lambda model: (
                EvaluationMethod.LIKELIHOOD_VEP,
            ),
        ),
    )

    with pytest.raises(ValueError, match="requires pseudo_likelihood"):
        ZeroShotVEP.from_model(
            dataset,
            model,
            score_method="masked_marginal",
        )
