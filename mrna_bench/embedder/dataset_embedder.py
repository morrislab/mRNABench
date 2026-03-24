from pathlib import Path
from collections.abc import Callable
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm
import h5py

import torch

from mrna_bench.models import EmbeddingModel
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
        agg_fn: Callable = partial(torch.mean, dim=0),
        ragged_out: bool = False
    ):
        """Initialize DatasetEmbedder.

        Args:
            model: Model used to embed sequences.
            dataset: Dataset to embed.
            d_chunk_ind: Current dataset chunk to be processed.
            d_num_chunks: Total number of chunks to divide dataset into.
            agg_fn: Aggregation function to apply to sequence embeddings.
            ragged_out: Whether the model produces ragged output under agg_fn.

        """
        self.model = model
        self.dataset = dataset
        self.data_df = dataset.data_df

        self.d_chunk_ind = d_chunk_ind
        self.d_num_chunks = d_num_chunks

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

        dataset_embeddings = []
        for _, row in tqdm(dataset_chunk.iterrows(), total=len(dataset_chunk)):
            embedding = self.model.embed(
                [row["sequence"]],
                cds=[row["cds"].astype(np.int32)],
                splice=[row["splice"].astype(np.int32)],
                agg_fn=self.agg_fn
            )
            dataset_embeddings.extend(embedding)

        return dataset_embeddings

    def persist_embeddings(self, embeddings: list[torch.Tensor]):
        """Persist embeddings at global data storage location.

        Args:
            embedding: Embedding to persist.
        """
        out_path = get_embedding_filepath(
            self.dataset.embedding_dir,
            self.model.short_name,
            self.dataset.dataset_name,
            self.d_chunk_ind,
            self.d_num_chunks
        )

        embeddings = [emb.detach().cpu() for emb in embeddings]

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

    @classmethod
    def from_dataframe(
        cls,
        model: EmbeddingModel,
        data_df: pd.DataFrame,
    ) -> "DatasetEmbedder":
        """Create a DatasetEmbedder instance from a custom dataframe.

        Args:
            model: Model used to embed sequences.
            data_df: DataFrame containing sequences and required columns:
                - sequence: RNA sequence
                - cds: CDS track information (as int32)
                - splice: Splice track information (as int32)

        Returns:
            Initialized DatasetEmbedder.

        Raises:
            ValueError: If required columns are missing from the dataframe.
        """
        # Check for required columns
        required_cols = ["sequence", "cds", "splice"]
        missing_cols = [
            col for col in required_cols if col not in data_df.columns
        ]
        if missing_cols:
            raise ValueError(
                f"DataFrame is missing required columns: {missing_cols}"
            )

        # Create a minimal BenchmarkDataset instance
        class MinimalBenchmarkDataset(BenchmarkDataset):
            METADATA = DatasetMetadata(
                dataset_name="custom",
                species="custom",
                task=["regression"],
                target_col=["target"],
                default_split_type="default",
                benchmark_set="extended",
                vep=False,
            )

            def __init__(self, data_df: pd.DataFrame):
                self.data_df = data_df
                self.dataset_name = "custom"
                self.dataset_path = "custom"
                self.embedding_dir = "custom"
                self.metadata = self.METADATA

            def _get_data_from_raw(self) -> pd.DataFrame:
                """Abstract method - not used for custom datasets."""
                raise NotImplementedError

        dataset = MinimalBenchmarkDataset(data_df)

        return cls(
            model=model,
            dataset=dataset,
        )
