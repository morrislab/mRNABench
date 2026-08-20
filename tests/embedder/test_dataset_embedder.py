from unittest.mock import Mock

import h5py
import numpy as np
import pandas as pd
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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_persist_pooled_half_as_float32(embedder, tmp_path, dtype):
    """Persist pooled half-precision embeddings as float32."""
    embeddings = [
        torch.tensor([1.25, -2.5], dtype=dtype),
        torch.tensor([3.75, 4.5], dtype=dtype),
    ]

    embedder.persist_embeddings(embeddings)

    with np.load(tmp_path / "mockdata_mockmodel.npz") as saved:
        persisted = saved["embedding"]

    assert persisted.dtype == np.float32
    np.testing.assert_array_equal(
        persisted,
        torch.stack(embeddings).float().numpy(),
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_persist_ragged_half_as_float32(embedder, tmp_path, dtype):
    """Persist ragged half-precision embeddings as float32."""
    embedder.ragged_out = True
    embeddings = [
        torch.tensor([[1.25, -2.5]], dtype=dtype),
        torch.tensor(
            [[3.75, 4.5], [5.25, -6.5]],
            dtype=dtype,
        ),
    ]

    embedder.persist_embeddings(embeddings)

    with h5py.File(tmp_path / "mockdata_mockmodel.h5", "r") as saved:
        persisted = [saved["embeddings"][str(i)][:] for i in range(2)]

    for actual, expected in zip(persisted, embeddings):
        assert actual.dtype == np.float32
        np.testing.assert_array_equal(actual, expected.float().numpy())


def test_embed_dataset_batches_rows_in_order():
    """Pass dataset rows to embed in configurable batches."""
    model = Mock()
    model.embed.side_effect = lambda sequences, **kwargs: [
        torch.tensor([len(sequence)]) for sequence in sequences
    ]
    dataset = Mock()
    dataset.data_df = pd.DataFrame({
        "sequence": ["A", "CC", "GGG", "TTTT", "AAAAA"],
        "cds": [np.zeros(length) for length in range(1, 6)],
        "splice": [np.ones(length) for length in range(1, 6)],
    })

    embeddings = DatasetEmbedder(
        model=model,
        dataset=dataset,
        batch_size=2,
    ).embed_dataset()

    assert model.embed.call_count == 3
    assert [value.item() for value in embeddings] == [1, 2, 3, 4, 5]
    first_call = model.embed.call_args_list[0]
    assert first_call.args[0] == ["A", "CC"]
    assert [len(track) for track in first_call.kwargs["cds"]] == [1, 2]


def test_embed_dataset_supports_sequence_only_data():
    """Omit optional tracks when the dataset does not provide them."""
    model = Mock()
    model.embed.return_value = [torch.tensor([1.0]), torch.tensor([2.0])]
    dataset = Mock()
    dataset.data_df = pd.DataFrame({"sequence": ["A", "CC"]})

    DatasetEmbedder(
        model=model,
        dataset=dataset,
        batch_size=2,
    ).embed_dataset()

    call = model.embed.call_args
    assert call.kwargs["cds"] is None
    assert call.kwargs["splice"] is None


def test_from_dataframe_preserves_optional_tracks():
    """Custom dataframes require sequence but retain optional model tracks."""
    model = Mock()
    model.short_name = "mockmodel"
    sequence_only = pd.DataFrame({"sequence": ["AC"]})

    embedder = DatasetEmbedder.from_dataframe(model, sequence_only)

    assert embedder.dataset.data_df.equals(sequence_only)
    with pytest.raises(ValueError, match="sequence"):
        DatasetEmbedder.from_dataframe(model, pd.DataFrame({"cds": [[]]}))
