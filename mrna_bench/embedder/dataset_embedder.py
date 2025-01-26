from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch

from mrna_bench.models import EmbeddingModel
from mrna_bench.datasets import BenchmarkDataset
from mrna_bench.embedder.embedder_utils import get_output_filepath


class DatasetEmbedder:
    """Embeds sequences associated with dataset using specified embedder.

    This class is built to split the sequences in a dataset into chunks of
    sequences which can then be processed in parallel. This is denoted d_chunk,
    while s_chunk denotes the sequence chunking that occur within each model
    to handle sequences that exceed model maximum length.
    """

    def __init__(
        self,
        model: EmbeddingModel,
        dataset: BenchmarkDataset,
        s_chunk_overlap: int = 0,
        d_chunk_ind: int = 0,
        d_num_chunks: int = 0,
    ):
        """Initialize DatasetEmbedder.

        Args:
            model: Model used to embed sequences.
            dataset: Dataset to embed.
            s_chunk_overlap: Number of overlapping tokens between chunks in
                individual sequences when using chunking to handle input
                exceeding maximum model length.
            d_chunk_ind: Current dataset chunk to be processed.
            d_num_chunks: Total number of chunks to divide dataset into.
        """
        self.model = model
        self.dataset = dataset
        self.data_df = dataset.data_df
        self.s_chunk_overlap = s_chunk_overlap

        self.d_chunk_ind = d_chunk_ind
        self.d_num_chunks = d_num_chunks

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

    def embed_dataset(self) -> torch.Tensor:
        """Compute embeddings for current dataset chunk.

        Returns:
            Embeddings for current dataset chunk in original order.
        """
        dataset_chunk = self.get_dataset_chunk()

        dataset_embeddings = []
        for _, row in tqdm(dataset_chunk.iterrows(), total=len(dataset_chunk)):
            if self.model.is_sixtrack:
                embedding = self.model.embed_sequence_sixtrack(
                    row["sequence"],
                    row["cds"].astype(np.int32),
                    row["splice"].astype(np.int32),
                    self.s_chunk_overlap,
                )
            else:
                embedding = self.model.embed_sequence(
                    row["sequence"],
                    self.s_chunk_overlap,
                )
            dataset_embeddings.append(embedding)

        embeddings = torch.cat(dataset_embeddings, dim=0)
        return embeddings

    def persist_embeddings(self, embeddings: torch.Tensor):
        """Persist embeddings at global data storage location.

        Args:
            embedding: Embedding to persist.
        """
        out_path = get_output_filepath(
            self.dataset.embedding_dir,
            self.model.short_name,
            self.dataset.dataset_name,
            self.s_chunk_overlap,
            self.d_chunk_ind,
            self.d_num_chunks
        )

        np_embeddings = embeddings.float().detach().cpu().numpy()
        np.savez_compressed(out_path, embedding=np_embeddings)

    def merge_embeddings(self):
        """Merge persisted processed dataset chunks into single file.

        Process will only complete if all chunks are finished processing.
        """
        all_chunks = list(range(self.d_num_chunks))
        processed_files_paths = []
        processed_chunk_inds = []

        # Check that all chunks are processed
        for file in Path(self.dataset.embedding_dir).iterdir():
            if not file.is_file():
                continue

            file_name = file.stem
            file_name_arr = file_name.split("_")

            if file_name_arr[0] != self.dataset.dataset_name:
                continue
            if file_name_arr[1] != self.model.short_name:
                continue
            if int(file_name_arr[2][1:]) != self.s_chunk_overlap:
                continue

            chunk_coords = file_name_arr[3].split("-")
            if int(chunk_coords[-1]) != self.d_num_chunks:
                continue

            processed_chunk_inds.append(int(chunk_coords[0]))
            processed_files_paths.append(file)

        if len(set(all_chunks) - set(processed_chunk_inds)) > 0:
            return

        print("All embedding chunks computed. Merging.")

        processed_files_paths = sorted(
            processed_files_paths,
            key=lambda x: int(Path(x).stem.split("_")[-1].split("-")[0])
        )

        embeddings = []
        for file_path in processed_files_paths:
            embedding_chunk = np.load(file_path)["embedding"]
            embeddings.append(embedding_chunk)

        all_embeddings = np.concatenate(embeddings, axis=0)

        out_fn = get_output_filepath(
            self.dataset.embedding_dir,
            self.model.short_name,
            self.dataset.dataset_name,
            self.s_chunk_overlap
        )

        np.savez_compressed(out_fn, embedding=all_embeddings)

        for file in processed_files_paths:
            Path(file).unlink()
