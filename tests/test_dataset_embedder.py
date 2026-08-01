from unittest.mock import Mock

import h5py
import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from mrna_bench.embedder import DatasetEmbedder


@pytest.fixture
def embedder(tmp_path) -> DatasetEmbedder:
    """Create an embedder configured to persist into a temporary directory."""
    model = Mock()
    model.short_name = "mockmodel"

    dataset = Mock()
    dataset.data_df = []
    dataset.dataset_name = "mockdata"
    dataset.embedding_dir = str(tmp_path)

    return DatasetEmbedder(model=model, dataset=dataset)


def test_persist_pooled_bfloat16_as_float32(embedder, tmp_path):
    """Persist pooled bfloat16 embeddings in a NumPy-compatible dtype."""
    embeddings = [
        torch.tensor([1.25, -2.5], dtype=torch.bfloat16),
        torch.tensor([3.75, 4.5], dtype=torch.bfloat16),
    ]

    embedder.persist_embeddings(embeddings)

    with np.load(tmp_path / "mockdata_mockmodel.npz") as saved:
        persisted = saved["embedding"]

    assert persisted.dtype == np.float32
    np.testing.assert_array_equal(
        persisted,
        torch.stack(embeddings).float().numpy(),
    )


def test_persist_ragged_bfloat16_as_float32(embedder, tmp_path):
    """Persist ragged bfloat16 embeddings in an HDF5-compatible dtype."""
    embedder.ragged_out = True
    embeddings = [
        torch.tensor([[1.25, -2.5]], dtype=torch.bfloat16),
        torch.tensor(
            [[3.75, 4.5], [5.25, -6.5]],
            dtype=torch.bfloat16,
        ),
    ]

    embedder.persist_embeddings(embeddings)

    with h5py.File(tmp_path / "mockdata_mockmodel.h5", "r") as saved:
        persisted = [saved["embeddings"][str(i)][:] for i in range(2)]

    for actual, expected in zip(persisted, embeddings):
        assert actual.dtype == np.float32
        np.testing.assert_array_equal(actual, expected.float().numpy())
