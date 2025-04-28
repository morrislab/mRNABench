from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch

from mrna_bench.models import EmbeddingModel
from mrna_bench.datasets import BenchmarkDataset
from mrna_bench.embedder.embedder_utils import get_embedding_filepath
from sklearn.preprocessing import StandardScaler


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
    ):
        """Initialize DatasetEmbedder.

        Args:
            model: Model used to embed sequences.
            dataset: Dataset to embed.
            d_chunk_ind: Current dataset chunk to be processed.
            d_num_chunks: Total number of chunks to divide dataset into.
        """
        self.model = model
        self.dataset = dataset
        self.data_df = dataset.data_df

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
                )
            else:
                embedding = self.model.embed_sequence(row["sequence"])
            dataset_embeddings.append(embedding)

        embeddings = torch.cat(dataset_embeddings, dim=0)
        return embeddings

    def persist_embeddings(self, embeddings: torch.Tensor):
        """Persist embeddings at global data storage location.

        Args:
            embedding: Embedding to persist.
        """
        out_path = get_embedding_filepath(
            self.dataset.embedding_dir,
            self.model.short_name,
            self.dataset.dataset_name,
            self.d_chunk_ind,
            self.d_num_chunks,
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

        glob_pattern = "{}_{}_*.npz".format(
            self.dataset.dataset_name, self.model.short_name
        )

        # Check that all chunks are processed
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
            key=lambda x: int(Path(x).stem.split("_")[-1].split("-")[0]),
        )

        embeddings = []
        for file_path in processed_files_paths:
            embedding_chunk = np.load(file_path)["embedding"]
            embeddings.append(embedding_chunk)

        all_embeddings = np.concatenate(embeddings, axis=0)

        out_fn = get_embedding_filepath(
            self.dataset.embedding_dir, self.model.short_name, self.dataset.dataset_name
        )

        np.savez_compressed(out_fn, embedding=all_embeddings)

        for file in processed_files_paths:
            Path(file).unlink()

    @classmethod
    def from_dataframe(
        cls,
        model: EmbeddingModel,
        data_df: pd.DataFrame,
        s_chunk_overlap: int = 0,
        transcript_avg: bool = False,
    ) -> "DatasetEmbedder":
        """Create a DatasetEmbedder instance from a custom dataframe.

        Args:
            model: Model used to embed sequences.
            data_df: DataFrame containing sequences and required columns:
                - sequence: RNA sequence
                - cds: CDS track information (as int32)
                - splice: Splice track information (as int32)
                - gene_id: (optional) Used when transcript_avg is True
            s_chunk_overlap: Number of overlapping tokens between chunks in
                individual sequences when using chunking to handle input
                exceeding maximum model length.
            transcript_avg: Whether to average embeddings of all transcripts
                for a given gene.

        Returns:
            Initialized DatasetEmbedder.

        Raises:
            ValueError: If required columns are missing from the dataframe.
        """
        # Check for required columns
        required_cols = ["sequence", "cds", "splice"]
        missing_cols = [col for col in required_cols if col not in data_df.columns]
        if missing_cols:
            raise ValueError(f"DataFrame is missing required columns: {missing_cols}")

        if transcript_avg and "gene_id" not in data_df.columns:
            raise ValueError("gene_id column is required when transcript_avg is True")

        # Create a minimal BenchmarkDataset instance
        class MinimalBenchmarkDataset:
            def __init__(self, data_df):
                self.data_df = data_df
                self.dataset_name = "custom"
                self.dataset_path = "custom"
                self.embedding_dir = "custom"
                self.species = "custom"  # Required for homology splitter

        dataset = MinimalBenchmarkDataset(data_df)

        return cls(
            model=model,
            dataset=dataset,
            s_chunk_overlap=s_chunk_overlap,
            transcript_avg=transcript_avg,
        )


class KmerDatasetEmbedder(DatasetEmbedder):
    """Embeds sequences associated with dataset using specified embedder.

    This class is built to split the sequences in a dataset into chunks of
    sequences which can then be processed in parallel. This is denoted d_chunk,
    whereas s_chunk denotes the sequence chunking that occur within each model
    to handle sequences that exceed model maximum length.

    This class specifically handles the naive Kmer embedding model.
    """

    def __init__(
        self,
        model: EmbeddingModel,
        dataset: BenchmarkDataset,
        d_chunk_ind: int = 0,
        d_num_chunks: int = 0,
    ):
        """Initialize KmerDatasetEmbedder.

        Args:
            model: Model used to embed sequences.
            dataset: Dataset to embed.
            d_chunk_ind: Current dataset chunk to be processed.
            d_num_chunks: Total number of chunks to divide dataset into.
        """
        super().__init__(model, dataset, d_chunk_ind, d_num_chunks)

    def embed_dataset(self) -> torch.Tensor:
        """Compute embeddings for current dataset chunk.

        Returns:
            Embeddings for current dataset chunk in original order.
        """
        embeddings = super().embed_dataset()

        # Desparsify the embeddings
        embeddings = self.desparsify_embeddings_and_scale(embeddings)

        return embeddings

    def desparsify_embeddings_and_scale(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Remove rows with 0s across all columns and scales the embeddings.

        Args:
            embeddings: Embeddings to desparsify.

        Returns:
            Desparsified embeddings.
        """
        # Remove rows with 0s across all columns
        non_zero_cols = torch.any(embeddings != 0, dim=0)
        desparsified_embeddings = embeddings[:, non_zero_cols]

        # Scale the embeddings
        desparsified_and_scaled_embeddings = torch.tensor(
            (StandardScaler().fit_transform(desparsified_embeddings)),
            dtype=torch.float32,
        )

        return desparsified_and_scaled_embeddings
