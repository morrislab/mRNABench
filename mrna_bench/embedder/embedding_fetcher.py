from pathlib import Path
from typing import Any, Self

import h5py
import numpy as np

from mrna_bench.datasets import BenchmarkDataset
from mrna_bench.embedder.embedder_utils import (
    find_embedding_file,
    get_config_hash,
    get_embedding_config,
    get_embedding_filepath,
    read_embedding_config,
)
from mrna_bench.models import EmbeddingModel
from mrna_bench.models.pooling import MeanPool, Pooling


class EmbeddingFetcher:
    """Mixin for builders that load persisted embeddings of their dataset.

    The builder sets ``dataset``. Each fetch method sets the embeddings,
    the embedding config stored with them, and the model short name, and
    returns the builder.
    """

    dataset: BenchmarkDataset
    model_short_name: str | None
    embeddings: Any
    embedding_config: dict | None

    @staticmethod
    def load_persisted_embeddings(
        embeddings_fn: str | Path,
    ) -> np.ndarray | list[np.ndarray]:
        """Load pre-computed embeddings from persisted location.

        Args:
            embeddings_fn: Path to an .npz embedding file, or an .h5 file of
                one array per sequence.

        Returns:
            Embeddings of the dataset rows, as one array for .npz files or
            a list of arrays for .h5 files.
        """
        if Path(embeddings_fn).suffix == ".h5":
            with h5py.File(embeddings_fn, "r") as f:
                grp = f["embeddings"]
                return [grp[str(i)][:] for i in range(len(grp))]
        else:
            return np.load(embeddings_fn)["embedding"]

    def fetch_embedding_by_model_instance(
        self,
        model: EmbeddingModel,
        pooling: Pooling = MeanPool(),
    ) -> Self:
        """Load persisted embeddings computed with a model instance.

        Args:
            model: Model used to generate embeddings.
            pooling: Pooling applied when embedding.

        Returns:
            Builder with set embeddings.
        """
        embedding_config = get_embedding_config(model, pooling)
        path = get_embedding_filepath(
            self.dataset.embedding_dir,
            model.short_name,
            self.dataset.dataset_name,
            get_config_hash(embedding_config),
        )
        suffix = ".npz" if Path(path + ".npz").exists() else ".h5"
        return self.fetch_embedding_by_embedding_instance(
            model.short_name,
            self.load_persisted_embeddings(path + suffix),
            embedding_config,
        )

    def fetch_embedding_by_model_name(
        self,
        model_short_name: str,
        pooling: Pooling = MeanPool(),
        attn_implementation: str | None = None,
        model_options: dict | None = None,
    ) -> Self:
        """Load persisted embeddings using model short name.

        Args:
            model_short_name: Short name of model used to generate embeddings.
            pooling: Pooling applied when embedding.
            attn_implementation: Attention implementation used when
                embedding. None accepts any, if only one file matches.
            model_options: Model options that differ from their defaults.

        Returns:
            Builder with set embeddings.
        """
        path = find_embedding_file(
            self.dataset.embedding_dir,
            model_short_name,
            self.dataset.dataset_name,
            pooling=pooling,
            attn_implementation=attn_implementation,
            model_options=model_options,
        )
        return self.fetch_embedding_by_embedding_instance(
            model_short_name,
            self.load_persisted_embeddings(path),
            read_embedding_config(path),
        )

    def fetch_embedding_by_filename(self, embedding_name: str) -> Self:
        """Load persisted embeddings using embedding file name.

        Args:
            embedding_name: Name of embedding file, ``.npz`` if it has no
                extension.

        Returns:
            Builder with set embeddings.
        """
        path = Path(self.dataset.embedding_dir) / embedding_name
        if path.suffix not in (".npz", ".h5"):
            path = path.with_name(path.name + ".npz")

        emb_fn_arr = path.stem.split("_")
        if len(emb_fn_arr) != 3:
            raise ValueError(
                "Invalid embedding filename format: {}".format(embedding_name)
            )

        return self.fetch_embedding_by_embedding_instance(
            emb_fn_arr[1],
            self.load_persisted_embeddings(path),
            read_embedding_config(path),
        )

    def fetch_embedding_by_embedding_instance(
        self,
        model_short_name: str,
        embeddings: Any,
        embedding_config: dict | None = None,
    ) -> Self:
        """Store embeddings computed outside the builder.

        Args:
            model_short_name: Short name of model used to generate embeddings.
            embeddings: Embeddings of the dataset rows.
            embedding_config: Settings the embeddings were computed with,
                see get_embedding_config. Required to persist results.

        Returns:
            Builder with set embeddings.
        """
        self.model_short_name = model_short_name
        self.embeddings = embeddings
        self.embedding_config = embedding_config
        return self
