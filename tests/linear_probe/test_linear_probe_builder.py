import pytest
from unittest.mock import Mock, patch

import numpy as np

from mrna_bench.datasets import BenchmarkDataset
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


def test_fetch_embedding_instance(mock_builder: LinearProbeBuilder):
    """Check that fetch functions set embeddings and model names."""
    mock_model = Mock()
    mock_model.short_name = "test_model"

    with patch.object(LinearProbeBuilder, "load_persisted_embeddings") as mock:
        mock.return_value = np.zeros((10, 10))
        mock_builder.fetch_embedding_by_model_instance(mock_model)
        mock.assert_called_once()

        assert mock_builder.model_short_name == "test_model"
        assert mock_builder.embeddings is not None
        assert mock_builder.seq_chunk_overlap == 0


def test_fetch_embedding_model_name(mock_builder: LinearProbeBuilder):
    """Check that fetch functions set embeddings and model names."""
    with patch.object(LinearProbeBuilder, "load_persisted_embeddings") as mock:
        mock.return_value = np.zeros((10, 10))
        mock_builder.fetch_embedding_by_model_name("test_model")
        mock.assert_called_once()

        assert mock_builder.model_short_name == "test_model"
        assert mock_builder.embeddings is not None
        assert mock_builder.seq_chunk_overlap == 0


def test_fetch_embedding_file_name(mock_builder: LinearProbeBuilder):
    """Check that fetch functions set embeddings and model names."""
    embedding_fn = "dataset_model-name_o10.npz"

    with patch.object(LinearProbeBuilder, "load_persisted_embeddings") as mock:
        mock.return_value = np.zeros((10, 10))
        mock_builder.fetch_embedding_by_filename(embedding_fn)
        mock.assert_called_once()

        assert mock_builder.model_short_name == "model-name"
        assert mock_builder.embeddings is not None
        assert mock_builder.seq_chunk_overlap == 10


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
            keyword_arg_test="Test"
        )

        mock_splitter_class.assert_called_once_with(
            (0.7, 0.15, 0.15),
            keyword_arg_test="Test"
        )
        assert mock_builder.splitter is not None
        assert mock_builder.eval_all_splits is True
        assert mock_builder.split_type == "homology"


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
    assert hasattr(mock_builder, "persister_flag") is False

    mock_builder.use_persister()
    assert mock_builder.persister_flag is True


def test_build(mock_builder: LinearProbeBuilder):
    """Check that build returns a LinearProbe instance."""
    mock_builder.embeddings = Mock()
    mock_builder.model_short_name = "test_model"
    mock_builder.seq_chunk_overlap = 0
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
            mock_persister.return_value = Mock()
            mock_persister.return_value.__class__ = LinearProbePersister

            mock_builder.build()
            mock_probe.assert_called_once()

            mock_persister.assert_not_called()

            mock_builder.use_persister()
            mock_builder.build()
            mock_persister.assert_called_once()
