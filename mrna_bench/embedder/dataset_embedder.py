from pathlib import Path
from collections.abc import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm
import h5py

import torch

from mrna_bench.models import EmbeddingModel, mean_pool
from mrna_bench.datasets import BenchmarkDataset, DatasetMetadata
from mrna_bench.embedder.embedder_utils import get_embedding_filepath


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
        agg_fn: Callable = mean_pool,
        ragged_out: bool = False,
        batch_size: int = 1,
    ):
        """Initialize DatasetEmbedder.

        Args:
            model: Model used to embed sequences.
            dataset: Dataset to embed.
            d_chunk_ind: Current dataset chunk to be processed.
            d_num_chunks: Total number of chunks to divide dataset into.
            agg_fn: Aggregation function to apply to sequence embeddings.
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

        self.agg_fn = agg_fn
        self.ragged_out = ragged_out

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
            self.d_chunk_ind,
            self.d_num_chunks
        )

        embeddings = [
            self._prepare_for_persistence(emb)
            for emb in embeddings
        ]

        if self.ragged_out:
            with h5py.File(out_path + ".h5", "w") as f:
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
                out_path + ".npz",
                embedding=embeddings_tensor.numpy()
            )

    def merge_embeddings(self):
        """Merge persisted processed dataset chunks into single file.

        Process will only complete if all chunks are finished processing.
        """
        all_chunks = list(range(self.d_num_chunks))
        processed_files_paths = []
        processed_chunk_inds = []

        base = "{}_{}_".format(
            self.dataset.dataset_name,
            self.model.short_name
        )

        glob_patterns = [
            base + "*.npz",
            base + "*.h5"
        ]

        # Check that all chunks are processed
        for glob_pattern in glob_patterns:
            for file in Path(self.dataset.embedding_dir).glob(glob_pattern):
                if not file.is_file():
                    continue

                file_name_arr = file.stem.split("_")
                if len(file_name_arr) < 3:
                    continue  # merged file, skip

                start, end = map(int, file_name_arr[2].split("-"))
                if end != self.d_num_chunks:
                    continue

                processed_chunk_inds.append(start)
                processed_files_paths.append(file)

        if len(set(all_chunks) - set(processed_chunk_inds)) > 0:
            return

        print("All embedding chunks computed. Merging.")

        processed_files_paths = sorted(
            processed_files_paths,
            key=lambda x: int(Path(x).stem.split("_")[-1].split("-")[0])
        )

        suffixes = set([file.suffix for file in processed_files_paths])
        if len(suffixes) != 1:
            raise ValueError(
                "Inconsistent file types found when merging embeddings."
            )

        suffix = suffixes.pop()

        if suffix == ".h5":
            self._merge_h5(processed_files_paths)
        elif suffix == ".npz":
            self._merge_npz(processed_files_paths)
        else:
            raise ValueError(
                f"Unsupported file type {suffix} found when merging "
                "embeddings."
            )

    def _merge_h5(self, processed_files_paths):
        """Merge .h5 embedding files.

        Args:
            processed_files_paths: List of file paths to merge.
        """
        out_fn = get_embedding_filepath(
            self.dataset.embedding_dir,
            self.model.short_name,
            self.dataset.dataset_name
        ) + ".h5"

        with h5py.File(out_fn, "w") as out_f:
            out_grp = out_f.create_group("embeddings")

            idx = 0
            for file_path in processed_files_paths:
                with h5py.File(file_path, "r") as f:
                    for _, emb in f["embeddings"].items():
                        out_grp.create_dataset(
                            str(idx),
                            data=emb[:],
                            compression="gzip"
                        )
                        idx += 1

        for file in processed_files_paths:
            Path(file).unlink()

    @classmethod
    def from_dataframe(
        cls,
        model: EmbeddingModel,
        data_df: pd.DataFrame,
    ) -> "DatasetEmbedder":
        """Create a DatasetEmbedder from a custom sequence dataframe."""
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
                self.dataset_path = "custom"
                self.embedding_dir = "custom"
                self.metadata = self.METADATA

            def _get_data_from_raw(self) -> pd.DataFrame:
                raise NotImplementedError

        return cls(model=model, dataset=MinimalBenchmarkDataset(data_df))

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

        out_fn = get_embedding_filepath(
            self.dataset.embedding_dir,
            self.model.short_name,
            self.dataset.dataset_name
        ) + ".npz"

        np.savez_compressed(out_fn, embedding=all_embeddings)

        for file in processed_files_paths:
            Path(file).unlink()
