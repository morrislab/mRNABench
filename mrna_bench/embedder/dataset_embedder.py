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
        transcript_avg: bool = False
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
            transcript_avg: Whether to average embeddings of all transcripts
                for a given gene.
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
        
        self.transcript_avg = transcript_avg
        
        if transcript_avg:
            self.gene_order = self.data_df["gene_id"].drop_duplicates().tolist()

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

    def embed_dataset(self) -> torch.Tensor | tuple[torch.Tensor, list, list]:
        """Compute embeddings for current dataset chunk.

        Returns:
            Embeddings for current dataset chunk in original order.
        """
        dataset_chunk = self.get_dataset_chunk()

        dataset_embeddings = []

        if self.transcript_avg:
            gene_list = []
            row_index_list = []


        for idx, row in tqdm(dataset_chunk.iterrows(), total=len(dataset_chunk)):
            seq = row["sequence"]
            if self.model.is_sixtrack:
                embedding = self.model.embed_sequence_sixtrack(
                    seq,
                    row["cds"].astype(np.int32),
                    row["splice"].astype(np.int32),
                    self.s_chunk_overlap,
                )
            else:
                embedding = self.model.embed_sequence(
                    seq,
                    self.s_chunk_overlap,
                )
            dataset_embeddings.append(embedding)

            if self.transcript_avg:
                gene_list.append(row["gene_id"])
                row_index_list.append(idx)

        if self.transcript_avg and self.d_num_chunks == 0: # not chunking, so we can average over all transcripts for a gene
            temp_df = pd.DataFrame(
                {
                    "gene_id": gene_list,
                    "row_index": row_index_list,
                    "embedding": [embedding.detach().cpu().numpy() for embedding in dataset_embeddings]
                }
            )

            _avg_embeds = lambda x: np.stack(x).mean(axis=0)

            temp_df = temp_df.groupby("gene_id").agg(
                {
                    "embedding": _avg_embeds,
                    "row_index": "first"
                }
            ).reset_index()

            temp_df = temp_df.sort_values("row_index")
            
            embeddings = torch.tensor(np.stack(temp_df["embedding"].tolist(), axis=0), device=self.model.device).squeeze() # shape: (N, embedding_dim)

            return embeddings

        embeddings = torch.cat(dataset_embeddings, dim=0) # shape: (chunk_size, embedding_dim)

        # chunking, so we can't average over all transcripts for a gene yet
        # there may be transcripts in other chunks
        if self.transcript_avg and self.d_num_chunks != 0: 
            return embeddings, gene_list, row_index_list
        else:
            return embeddings

    def persist_embeddings(self, 
        embeddings: torch.Tensor, 
        gene_list: list = None,
        row_index_list: list = None
    ):
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

        if len(gene_list) > 0 and len(row_index_list) > 0:
            np_genes = np.array(gene_list)
            np_row_ix = np.array(row_index_list)
            np.savez_compressed(out_path,
                embedding=np_embeddings,
                gene=np_genes,
                row_idx=np_row_ix
            )
        else:
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

        if self.transcript_avg:
            
            gene_chunk = []
            row_index_chunk = []

            for file_path in processed_files_paths:
                data = np.load(file_path)
                embeddings = data["embedding"]
                gene_chunk = data["gene"]
                row_index_chunk = data["row_idx"]

            temp_df = pd.DataFrame(
                {
                    "gene_id": np.concatenate(gene_chunk).tolist(), # shape: (N,)
                    "row_index": np.concatenate(row_index_chunk).tolist(), # shape: (N,)
                    "embedding": np.concatenate(embeddings).tolist() # shape: (N, embed_dim)
                }
            )

            _avg_embeds = lambda x: np.stack(x).mean(axis=0)

            temp_df = temp_df.groupby("gene_id").agg(
                {
                    "embedding": _avg_embeds,
                    "row_index": "first"
                }
            ).reset_index()

            temp_df = temp_df.sort_values("row_index")
        
            all_embeddings = np.concatenate(temp_df["embedding"].tolist(), axis=0).squeeze() # shape: (N, embed_dim)

        else:

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

    @classmethod
    def from_dataframe(
        cls,
        model: EmbeddingModel,
        data_df: pd.DataFrame,
        s_chunk_overlap: int = 0,
        transcript_avg: bool = False
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
            transcript_avg=transcript_avg
        )
