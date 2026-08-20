import pandas as pd
import pytest

from mrna_bench.datasets import (
    BenchmarkDataset,
    DatasetMetadata,
    EvaluationMethod,
)
from mrna_bench.models import ModelBehavior


def test_vep_pairing_must_be_defined_by_dataset():
    """Datasets without VEP pairing fail instead of guessing a schema."""
    dataset = type("Dataset", (), {"dataset_name": "test"})()
    with pytest.raises(NotImplementedError, match="does not define"):
        BenchmarkDataset.get_vep_pairs(dataset, pd.DataFrame())


def test_dataset_metadata_expands_task_specs():
    """Dataset metadata expands targets and exposes VEP routes."""
    metadata = DatasetMetadata(
        dataset_name="test",
        species="human",
        task=["classification"],
        target_col=["label"],
        default_split_type="default",
        benchmark_set="core",
        evaluations=("linear_probe", "embedding_vep", "likelihood_vep"),
    )

    assert metadata.task_specs[0].task == "classification"
    assert EvaluationMethod.LINEAR_PROBE in metadata.evaluations
    assert EvaluationMethod.LIKELIHOOD_VEP in metadata.evaluations
    assert metadata.is_vep
    assert [spec.target_col for spec in metadata.task_specs] == [
        "label",
    ]


def test_dataset_metadata_rejects_ambiguous_task_targets():
    """Ambiguous task-to-target cardinality is rejected."""
    with pytest.raises(ValueError):
        DatasetMetadata(
            dataset_name="test",
            species="human",
            task=["classification", "regression"],
            target_col=["a", "b", "c"],
            default_split_type="default",
            benchmark_set="core",
            evaluations=("linear_probe",),
        )


def test_standard_dataset_declares_linear_probe():
    """Ordinary datasets explicitly declare linear probing."""
    metadata = DatasetMetadata(
        dataset_name="test",
        species="human",
        task=["regression"],
        target_col=["label"],
        default_split_type="default",
        benchmark_set="core",
        evaluations=("linear_probe",),
    )

    assert metadata.evaluations == (EvaluationMethod.LINEAR_PROBE,)
    assert not metadata.is_vep


def test_dataset_model_compatibility():
    """Compatibility is the intersection of dataset and model contracts."""
    metadata = DatasetMetadata(
        dataset_name="test",
        species="human",
        task=["classification"],
        target_col=["label"],
        default_split_type="default",
        benchmark_set="core",
        evaluations=("linear_probe", "embedding_vep", "likelihood_vep"),
    )

    class EmbeddingOnly:
        def supports(self, behavior):
            return behavior == ModelBehavior.EMBEDDING

    assert metadata.compatible_evaluations(EmbeddingOnly()) == (
        EvaluationMethod.LINEAR_PROBE,
        EvaluationMethod.EMBEDDING_VEP,
    )


def test_utr_vep_rejects_cds_only_likelihood():
    """UTR variants are incompatible with models that only score CDS."""
    metadata = DatasetMetadata(
        dataset_name="test",
        species="human",
        task=["classification"],
        target_col=["label"],
        default_split_type="default",
        benchmark_set="core",
        evaluations=("likelihood_vep",),
        variant_region="utr",
    )

    class CDSOnly:
        sequence_score_scope = "cds"

        def supports(self, behavior):
            return behavior == ModelBehavior.PSEUDO_LIKELIHOOD

    assert metadata.compatible_evaluations(CDSOnly()) == ()


def test_embedding_vep_requires_compatible_sequence_region():
    """CDS-only embeddings cannot evaluate UTR variants."""
    metadata = DatasetMetadata(
        dataset_name="test",
        species="human",
        task=["classification"],
        target_col=["target"],
        default_split_type="default",
        benchmark_set="core",
        evaluations=("embedding_vep",),
        variant_region="utr",
    )

    class CDSOnly:
        sequence_score_scope = "cds"

        def supports(self, behavior):
            return behavior == ModelBehavior.EMBEDDING

    assert metadata.compatible_evaluations(CDSOnly()) == ()
