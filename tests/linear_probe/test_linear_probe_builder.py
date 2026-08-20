import pytest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from mrna_bench.datasets import BenchmarkDataset
from mrna_bench.datasets.utr_variants_bohn import UTRVariantsBohn
from mrna_bench.linear_probe import LinearProbeBuilder
from mrna_bench.linear_probe.persister import LinearProbePersister


@pytest.fixture
def mock_dataset() -> BenchmarkDataset:
    """Return a mock dataset object."""
    mock_dataset = Mock()
    mock_dataset.__class__ = BenchmarkDataset
    return mock_dataset


@pytest.fixture
def mock_builder(mock_dataset: BenchmarkDataset) -> LinearProbeBuilder:
    """Return a mock LinearProbeBuilder object."""
    return LinearProbeBuilder(mock_dataset)


def test_builder_initialization_data(mock_dataset: BenchmarkDataset):
    """Test LinearProbeBuilder initialization.

    Checks dataset loading logic.
    """
    builder = LinearProbeBuilder(mock_dataset)
    assert builder.target_col == "target"

    with patch(
        "mrna_bench.linear_probe.linear_probe_builder.load_dataset"
    ) as mock_method:
        mock_method.return_value = mock_dataset
        LinearProbeBuilder(dataset_name="test_dataset")
        mock_method.assert_called_once_with("test_dataset")

    with pytest.raises(ValueError):
        LinearProbeBuilder(mock_dataset, dataset_name="test_dataset")

    with pytest.raises(ValueError):
        LinearProbeBuilder()


def test_fetch_embedding_model_name(mock_builder: LinearProbeBuilder):
    """Check that fetch functions set embeddings and model names."""
    with patch.object(LinearProbeBuilder, "load_persisted_embeddings") as mock:
        mock.return_value = np.zeros((10, 10))
        mock_builder.fetch_embedding_by_model_name("test_model")
        mock.assert_called_once()

        assert mock_builder.model_short_name == "test_model"
        assert mock_builder.embeddings is not None


def test_fetch_embedding_instance(mock_builder: LinearProbeBuilder):
    """Model instances load embeddings by their short name."""
    model = Mock(short_name="test_model")
    with patch.object(
        mock_builder,
        "fetch_embedding_by_model_name",
        return_value=mock_builder,
    ) as fetch:
        result = mock_builder.fetch_embedding_by_model_instance(model)

    fetch.assert_called_once_with("test_model")
    assert result is mock_builder


def test_fetch_embedding_file_name(mock_builder: LinearProbeBuilder):
    """Check that fetch functions set embeddings and model names."""
    embedding_fn = "dataset_model-name.npz"

    with patch.object(LinearProbeBuilder, "load_persisted_embeddings") as mock:
        mock.return_value = np.zeros((10, 10))
        mock_builder.fetch_embedding_by_filename(embedding_fn)
        mock.assert_called_once()

        assert mock_builder.model_short_name == "model-name"
        assert mock_builder.embeddings is not None


def test_build_splitter(mock_builder: LinearProbeBuilder):
    """Check that build_splitter returns a LinearProbeBuilder."""
    mock_splitter_class = Mock()
    with patch.dict(
        "mrna_bench.linear_probe.linear_probe_builder.SPLIT_CATALOG",
        {"homology": mock_splitter_class}
    ):
        mock_builder.build_splitter(
            "homology",
            eval_all_splits=True,
            species="human",
            keyword_arg_test="Test"
        )

        mock_splitter_class.assert_called_once_with(
            (0.7, 0.15, 0.15),
            species="human",
            keyword_arg_test="Test"
        )
        assert mock_builder.splitter is not None
        assert mock_builder.eval_all_splits is True
        assert mock_builder.split_type == "homology"


def test_homology_splitter_builds_inside_linear_probe(tmp_path):
    """A configured homology splitter survives the full builder path."""
    dataset = Mock()
    dataset.__class__ = BenchmarkDataset
    dataset.data_df = pd.DataFrame({
        "gene": ["A", "B", "C", "D", "E", "F"],
        "target": np.arange(6, dtype=float),
    })
    dataset.metadata.target_col = ["target"]
    dataset.metadata.task = ["regression"]
    dataset.metadata.default_split_type = "homology"
    dataset.metadata.species = "human"
    dataset.metadata.is_vep = False
    source = tmp_path / "paralogs.tsv"
    pd.DataFrame(
        [("A", "B", 50), ("B", "A", 50)],
        columns=[
            "Gene name",
            "Paralogue associated gene name",
            "Paralogue %id. target gene identical to query gene",
        ],
    ).to_csv(source, sep="\t", index=False)

    builder = LinearProbeBuilder(dataset)
    builder.embeddings = np.arange(12).reshape(6, 2)
    builder.build_splitter(
        "homology",
        homology_map_path=str(tmp_path),
        homology_source_path=str(source),
        similarity_threshold=35,
    )
    probe = builder.build()
    splits = probe.get_df_splits(random_seed=1)

    label_rows = sum(
        len(value)
        for key, value in splits.items()
        if key.endswith("_y")
    )
    assert label_rows == 6
    assert any(
        {0, 1}.issubset(values)
        for key, values in splits.items()
        if key.endswith("_y")
    )


def test_set_target(mock_builder: LinearProbeBuilder):
    """Check that set_target sets target."""
    assert mock_builder.target_col == "target"
    target_name = "test_target"
    mock_builder.set_target(target_name)
    assert mock_builder.target_col == target_name


def test_build_evaluator(mock_builder: LinearProbeBuilder):
    """Check that build_evaluator returns a LinearProbeEvaluator."""
    with patch(
        "mrna_bench.linear_probe.linear_probe_builder.LinearProbeEvaluator"
    ) as mock:
        mock.return_value = Mock()
        mock_builder.build_evaluator("task")
        mock.assert_called_once_with("task")
        assert mock_builder.evaluator is not None


def test_use_persister(mock_builder: LinearProbeBuilder):
    """Check that use_persister sets persister."""
    assert mock_builder.persister_flag is False

    mock_builder.use_persister()
    assert mock_builder.persister_flag is True


def test_set_regressor(mock_builder: LinearProbeBuilder):
    """Regression estimators are selected independently of dataset task."""
    assert mock_builder.set_regressor("ridge").regressor == "ridge"
    with pytest.raises(ValueError, match="regressor"):
        mock_builder.set_regressor("invalid")


def test_build(mock_builder: LinearProbeBuilder):
    """Check that build returns a LinearProbe instance."""
    mock_builder.embeddings = Mock()
    mock_builder.model_short_name = "test_model"
    mock_builder.target_col = "target"
    mock_builder.task = "task"
    mock_builder.splitter = Mock()
    mock_builder.split_type = "homology"
    mock_builder.evaluator = Mock()
    mock_builder.eval_all_splits = True

    with patch(
        "mrna_bench.linear_probe.linear_probe_builder.LinearProbePersister"
    ) as mock_persister:
        with patch(
            "mrna_bench.linear_probe.linear_probe_builder.LinearProbe"
        ) as mock_probe:
            with patch.object(
                LinearProbeBuilder, "validate", return_value=[]
            ):
                mock_persister.return_value = Mock()
                mock_persister.return_value.__class__ = LinearProbePersister

                mock_builder.build()
                mock_probe.assert_called_once()

                mock_persister.assert_not_called()

                mock_builder.use_persister()
                mock_builder.build()
                mock_persister.assert_called_once()


def test_builder_sets_is_vep_flag(mock_dataset: BenchmarkDataset):
    """Test that LinearProbeBuilder derives VEP from evaluations."""
    mock_dataset.metadata.is_vep = True
    builder = LinearProbeBuilder(mock_dataset)

    assert builder.is_vep is True


def test_builder_uses_dataset_vep_pairing():
    """Build embedding deltas through the dataset's pairing contract."""
    dataset = Mock()
    dataset.__class__ = BenchmarkDataset
    dataset.data_df = pd.DataFrame({
        "transcript_id": ["tx1", "tx1"],
        "description": ["wild-type", "chr1:10 A:T"],
        "target": [0, 1],
    })
    dataset.metadata.target_col = ["target"]
    dataset.metadata.task = ["regression"]
    dataset.metadata.default_split_type = "default"
    dataset.metadata.is_vep = True
    dataset.get_vep_pairs.side_effect = lambda dataframe, columns: (
        UTRVariantsBohn.get_vep_pairs(dataset, dataframe, columns)
    )
    builder = LinearProbeBuilder(dataset)
    builder.embeddings = np.array([[1.0, 1.0], [3.0, 4.0]])
    builder.splitter = Mock()

    probe = builder.build()

    dataset.get_vep_pairs.assert_called_once()
    np.testing.assert_array_equal(
        probe.data_df.iloc[0]["embeddings"],
        np.array([2.0, 3.0]),
    )


def test_builder_defers_default_splitter(mock_dataset: BenchmarkDataset):
    """Builder construction does not download homology data eagerly."""
    mock_dataset.metadata.default_split_type = "homology"
    mock_dataset.metadata.species = "human"

    with patch.object(
        LinearProbeBuilder,
        "_build_default_splitter",
    ) as build_splitter:
        builder = LinearProbeBuilder(mock_dataset)

    build_splitter.assert_not_called()
    assert builder.splitter is None
