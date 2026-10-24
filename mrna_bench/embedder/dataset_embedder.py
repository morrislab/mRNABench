from pathlib import Path
import fcntl
import json
import os
import uuid

import numpy as np
import pandas as pd
from tqdm import tqdm
import h5py

import torch

from mrna_bench.models import EmbeddingModel
from mrna_bench.models.pooling import MeanPool, Pooling
from mrna_bench.datasets import BenchmarkDataset, DatasetMetadata
from mrna_bench.embedder.embedder_utils import (
    get_config_hash,
    get_embedding_config,
    get_embedding_filepath,
)


class DatasetEmbedder:
    """Embeds sequences associated with dataset using specified embedder.

    This class is built to split the sequences in a dataset into chunks of
    sequences which can then be processed in parallel. This is denoted d_chunk,
    whereas s_chunk denotes the sequence chunking that occur within each model
    to handle sequences that exceed model maximum length.
    """

    def __init__(
        self,
        model: EmbeddingModel,
        dataset: BenchmarkDataset,
        d_chunk_ind: int = 0,
        d_num_chunks: int = 0,
        agg_fn: Pooling = MeanPool(),
        ragged_out: bool = False,
        batch_size: int = 1,
    ):
        """Initialize DatasetEmbedder.

        Args:
            model: Model used to embed sequences.
            dataset: Dataset to embed.
            d_chunk_ind: Current dataset chunk to be processed.
            d_num_chunks: Total number of chunks to divide dataset into.
            agg_fn: Pooling applied to sequence embeddings.
            ragged_out: Whether the model produces ragged output under agg_fn.
            batch_size: Number of dataset rows passed to model.embed at once.

        """
        self.model = model
        self.dataset = dataset
        self.data_df = dataset.data_df

        self.d_chunk_ind = d_chunk_ind
        self.d_num_chunks = d_num_chunks
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        self.batch_size = batch_size

        if not isinstance(agg_fn, Pooling):
            raise TypeError(
                "agg_fn must be a Pooling, e.g. MeanPool() or NoPool()."
            )
        self.agg_fn = agg_fn
        self.ragged_out = ragged_out
        self.config = get_embedding_config(model, agg_fn)
        self.config_hash = get_config_hash(self.config)

        if self.d_num_chunks == 0:
            self.d_chunk_size = len(self.data_df)
        else:
            self.d_chunk_size = (len(self.data_df) // self.d_num_chunks) + 1

    def get_dataset_chunk(self) -> pd.DataFrame:
        """Retrieve current dataset chunk to be embedded.

        Returns:
            Current dataset chunk to be embedded.
        """
        if self.d_num_chunks == 0:
            return self.data_df

        s = self.d_chunk_size * self.d_chunk_ind
        e = s + self.d_chunk_size

        chunk_df = self.data_df.iloc[s:e]
        return chunk_df

    def embed_dataset(self) -> list[torch.Tensor]:
        """Compute embeddings for current dataset chunk.

        Returns:
            Embeddings for current dataset chunk in original order.
            - pooled: (1, H)
            - unpooled: (1, L_i, H)
        """
        dataset_chunk = self.get_dataset_chunk()
        self.model.set_inference_mode()

        dataset_embeddings: list[torch.Tensor] = []
        num_batches = (
            len(dataset_chunk) + self.batch_size - 1
        ) // self.batch_size
        with torch.inference_mode():
            for start in tqdm(
                range(0, len(dataset_chunk), self.batch_size),
                total=num_batches,
            ):
                batch = dataset_chunk.iloc[start:start + self.batch_size]
                cds = (
                    [np.asarray(track, dtype=np.int32)
                     for track in batch["cds"]]
                    if "cds" in batch.columns else None
                )
                splice = (
                    [np.asarray(track, dtype=np.int32)
                     for track in batch["splice"]]
                    if "splice" in batch.columns else None
                )
                embeddings = self.model.embed(
                    batch["sequence"].tolist(),
                    cds=cds,
                    splice=splice,
                    agg_fn=self.agg_fn
                )
                if len(embeddings) != len(batch):
                    raise RuntimeError(
                        "model.embed returned {} embeddings for {} sequences."
                        .format(len(embeddings), len(batch))
                    )
                dataset_embeddings.extend(embeddings)

        return dataset_embeddings

    @staticmethod
    def _prepare_for_persistence(embedding: torch.Tensor) -> torch.Tensor:
        """Move an embedding to CPU with a NumPy-compatible dtype."""
        embedding = embedding.detach().cpu()
        if embedding.is_floating_point() and embedding.dtype != torch.float32:
            embedding = embedding.float()
        return embedding

    def persist_embeddings(self, embeddings: list[torch.Tensor]):
        """Persist embeddings at global data storage location.

        Args:
            embeddings: Embeddings to persist.
        """
        out_path = get_embedding_filepath(
            self.dataset.embedding_dir,
            self.model.short_name,
            self.dataset.dataset_name,
            self.config_hash,
            self.d_chunk_ind,
            self.d_num_chunks
        )

        embeddings = [
            self._prepare_for_persistence(emb)
            for emb in embeddings
        ]

        suffix = ".h5" if self.ragged_out else ".npz"
        path = Path(out_path + suffix)
        tmp_path = self._temporary_path(path)

        if self.ragged_out:
            with h5py.File(tmp_path, "w") as f:
                f.attrs["config"] = json.dumps(self.config)
                grp = f.create_group("embeddings")
                for i, emb in enumerate(embeddings):
                    grp.create_dataset(
                        str(i),
                        data=emb.numpy(),
                        shuffle=True,
                        compression="gzip",
                    )
        else:
            embeddings_tensor = torch.stack(embeddings, dim=0)
            np.savez_compressed(
                tmp_path,
                embedding=embeddings_tensor.numpy(),
                config=json.dumps(self.config),
            )
        os.replace(tmp_path, path)

    @staticmethod
    def _temporary_path(path: Path) -> Path:
        """Get a hidden name to write a file under before renaming it.

        Renaming a complete file into place means neither a reader nor a job
        killed while writing ever leaves a partly written file at path.

        Args:
            path: Final file path.

        Returns:
            Unique hidden path in the same directory.
        """
        return path.with_name(".{}.{}.tmp{}".format(
            path.stem, uuid.uuid4().hex[:8], path.suffix
        ))

    def merge_embeddings(self):
        """Merge persisted dataset chunks into a single file.

        Merges only once every chunk is persisted. Holds a file lock, so
        that jobs finishing their chunks at the same time merge once.
        """
        suffix = ".h5" if self.ragged_out else ".npz"
        chunk_paths = [
            Path(get_embedding_filepath(
                self.dataset.embedding_dir,
                self.model.short_name,
                self.dataset.dataset_name,
                self.config_hash,
                i,
                self.d_num_chunks,
            ) + suffix)
            for i in range(self.d_num_chunks)
        ]
        lock_path = Path(self.dataset.embedding_dir) / "{}_{}_{}.lock".format(
            self.dataset.dataset_name,
            self.model.short_name,
            self.config_hash,
        )
        with open(lock_path, "w") as lock_fh:
            fcntl.flock(lock_fh, fcntl.LOCK_EX)
            if not all(path.exists() for path in chunk_paths):
                return
            print("All embedding chunks computed. Merging.")
            if self.ragged_out:
                self._merge_h5(chunk_paths)
            else:
                self._merge_npz(chunk_paths)

    def _merge_h5(self, processed_files_paths):
        """Merge .h5 embedding files.

        Args:
            processed_files_paths: List of file paths to merge.
        """
        out_path = Path(get_embedding_filepath(
            self.dataset.embedding_dir,
            self.model.short_name,
            self.dataset.dataset_name,
            self.config_hash,
        ) + ".h5")
        tmp_path = self._temporary_path(out_path)

        with h5py.File(tmp_path, "w") as out_f:
            out_f.attrs["config"] = json.dumps(self.config)
            out_grp = out_f.create_group("embeddings")

            idx = 0
            for file_path in processed_files_paths:
                with h5py.File(file_path, "r") as f:
                    # h5py lists entries by name, so "10" comes before "2".
                    grp = f["embeddings"]
                    for name in sorted(grp, key=int):
                        out_grp.create_dataset(
                            str(idx),
                            data=grp[name][:],
                            compression="gzip"
                        )
                        idx += 1
        os.replace(tmp_path, out_path)

        for file in processed_files_paths:
            Path(file).unlink()

    @classmethod
    def from_dataframe(
        cls,
        model: EmbeddingModel,
        data_df: pd.DataFrame,
        embedding_dir: str = ".",
        **kwargs,
    ) -> "DatasetEmbedder":
        """Create a DatasetEmbedder from a custom sequence dataframe.

        Args:
            model: Model used to embed sequences.
            data_df: Dataframe with a "sequence" column, and optionally
                "cds" and "splice" tracks.
            embedding_dir: Directory embeddings are persisted to.
            **kwargs: Other DatasetEmbedder arguments, such as agg_fn,
                batch_size, d_chunk_ind and d_num_chunks.

        Returns:
            DatasetEmbedder for the dataframe.
        """
        if "sequence" not in data_df:
            raise ValueError("DataFrame is missing required column: sequence")

        class MinimalBenchmarkDataset(BenchmarkDataset):
            METADATA = DatasetMetadata(
                dataset_name="custom",
                species="custom",
                task=["regression"],
                target_col=["target"],
                default_split_type="default",
                benchmark_set="extended",
                evaluations=("linear_probe",),
            )

            def __init__(self, dataframe: pd.DataFrame):
                self.data_df = dataframe
                self.dataset_name = "custom"
                self.dataset_path = embedding_dir
                self.embedding_dir = embedding_dir
                self.metadata = self.METADATA

            def _get_data_from_raw(self) -> pd.DataFrame:
                raise NotImplementedError

        return cls(
            model=model,
            dataset=MinimalBenchmarkDataset(data_df),
            **kwargs,
        )

    def _merge_npz(self, processed_files_paths):
        """Merge .npz embedding files.

        Args:
            processed_files_paths: List of file paths to merge.
        """
        embeddings = []
        for file_path in processed_files_paths:
            embedding_chunk = np.load(file_path)["embedding"]
            embeddings.append(embedding_chunk)

        all_embeddings = np.concatenate(embeddings, axis=0)

        out_path = Path(get_embedding_filepath(
            self.dataset.embedding_dir,
            self.model.short_name,
            self.dataset.dataset_name,
            self.config_hash,
        ) + ".npz")
        tmp_path = self._temporary_path(out_path)

        np.savez_compressed(
            tmp_path,
            embedding=all_embeddings,
            config=json.dumps(self.config),
        )
        os.replace(tmp_path, out_path)

        for file in processed_files_paths:
            Path(file).unlink()
