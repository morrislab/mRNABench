from dataclasses import dataclass
from unittest.mock import Mock

import h5py
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
import torch

from mrna_bench.embedder import (
    DatasetEmbedder,
    find_embedding_file,
    read_embedding_config,
)
from mrna_bench.models.pooling import Pooling


@dataclass(frozen=True)
class _WindowPool(Pooling):
    """Non-default pooling for config tests: mean over `window`-long chunks."""

    window: int

    def __call__(self, hidden_states, dim=0):
        _ = dim
        return hidden_states.float().reshape(-1, self.window, hidden_states.shape[-1]).mean(1)


def mock_model(**model_options) -> Mock:
    """Create a model mock with the attributes stored in embedding configs."""
    return Mock(
        short_name="mockmodel",
        attn_implementation=None,
        default_attn_implementation="sdpa",
        model_options=model_options,
    )


@pytest.fixture
def embedder(tmp_path) -> DatasetEmbedder:
    """Create an embedder configured to persist into a temporary directory."""
    model = mock_model()

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

    path = tmp_path / f"mockdata_mockmodel_{embedder.config_hash}.npz"
    with np.load(path) as saved:
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

    path = tmp_path / f"mockdata_mockmodel_{embedder.config_hash}.h5"
    with h5py.File(path, "r") as saved:
        persisted = [saved["embeddings"][str(i)][:] for i in range(2)]

    for actual, expected in zip(persisted, embeddings):
        assert actual.dtype == np.float32
        np.testing.assert_array_equal(actual, expected.float().numpy())


def test_persisted_config_finds_file(embedder, tmp_path):
    """Embeddings are named by config hash and found by stored config."""
    embedder.persist_embeddings([torch.zeros(2), torch.ones(2)])

    path = find_embedding_file(str(tmp_path), "mockmodel", "mockdata")
    assert path.stem.split("_")[2] == embedder.config_hash
    assert read_embedding_config(path) == {
        "pooling": {"name": "MeanPool"},
        "layer": "final",
        "attn_implementation": "sdpa",
    }
    with pytest.raises(FileNotFoundError):
        find_embedding_file(
            str(tmp_path), "mockmodel", "mockdata",
            pooling=_WindowPool(64),
        )


def test_model_options_separate_files(tmp_path):
    """Non-default model options get their own file and lookup."""
    dataset = Mock(dataset_name="mockdata", embedding_dir=str(tmp_path))
    dataset.data_df = []
    for model, value in ((mock_model(), 0.0), (mock_model(kmer_list=[3]), 1.0)):
        DatasetEmbedder(model, dataset).persist_embeddings([torch.full((2,), value)])

    default = find_embedding_file(str(tmp_path), "mockmodel", "mockdata")
    custom = find_embedding_file(
        str(tmp_path), "mockmodel", "mockdata",
        model_options={"kmer_list": [3]},
    )
    assert np.load(default)["embedding"][0, 0] == 0
    assert np.load(custom)["embedding"][0, 0] == 1
    assert "model_options" not in read_embedding_config(default)


def test_merged_chunks_keep_config(tmp_path):
    """Merging chunk files writes the config into the merged file."""
    dataset = Mock(dataset_name="mockdata", embedding_dir=str(tmp_path))
    dataset.data_df = list(range(4))
    pooling = _WindowPool(2)
    for i in range(2):
        chunk = DatasetEmbedder(
            mock_model(), dataset, d_chunk_ind=i, d_num_chunks=2,
            agg_fn=pooling,
        )
        chunk.persist_embeddings([torch.full((1, 2), float(i))] * 2)
    chunk.merge_embeddings()

    path = find_embedding_file(
        str(tmp_path), "mockmodel", "mockdata", pooling=pooling
    )
    assert read_embedding_config(path)["pooling"] == pooling.config
    np.testing.assert_array_equal(
        np.load(path)["embedding"][:, 0, 0], [0, 0, 1, 1]
    )


def test_merged_ragged_chunks_keep_row_order(tmp_path):
    """Merged ragged chunks keep row order past row 9 of a chunk."""
    dataset = Mock(dataset_name="mockdata", embedding_dir=str(tmp_path))
    dataset.data_df = pd.DataFrame({"row": range(24)})
    for i in range(2):
        chunk = DatasetEmbedder(
            mock_model(), dataset, d_chunk_ind=i, d_num_chunks=2,
            ragged_out=True,
        )
        rows = chunk.get_dataset_chunk()["row"]
        chunk.persist_embeddings([torch.full((1, 2), float(r)) for r in rows])
    chunk.merge_embeddings()

    path = find_embedding_file(str(tmp_path), "mockmodel", "mockdata")
    with h5py.File(path, "r") as merged:
        grp = merged["embeddings"]
        values = [float(grp[str(i)][0, 0]) for i in range(len(grp))]
    assert values == list(range(24))


def test_embedder_rejects_plain_functions():
    """Pooling must be a Pooling so it can be stored with the embeddings."""
    with pytest.raises(TypeError, match="Pooling"):
        DatasetEmbedder(mock_model(), Mock(), agg_fn=lambda h: h.mean(0))


def test_embed_dataset_batches_rows_in_order():
    """Pass dataset rows to embed in configurable batches."""
    model = mock_model()
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
    model = mock_model()
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
    model = mock_model()
    sequence_only = pd.DataFrame({"sequence": ["AC"]})

    embedder = DatasetEmbedder.from_dataframe(model, sequence_only)

    assert embedder.dataset.data_df.equals(sequence_only)
    with pytest.raises(ValueError, match="sequence"):
        DatasetEmbedder.from_dataframe(model, pd.DataFrame({"cds": [[]]}))
