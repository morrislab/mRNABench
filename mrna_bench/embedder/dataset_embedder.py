from pathlib import Path

import pandas as pd
import numpy as np

import torch

from mrna_bench.models import EmbeddingModel
from mrna_bench.datasets import BenchmarkDataset
from mrna_bench.embedder.embedder_utils import get_output_filename


class DatasetEmbedder:
    def __init__(
        self,
        model: EmbeddingModel,
        dataset: BenchmarkDataset,
        s_chunk_overlap: int = 0,
        d_chunk_ind: int = 0,
        d_chunk_max_ind: int = 0,
    ):
        self.model = model
        self.dataset = dataset
        self.data_df = dataset.data_df
        self.s_chunk_overlap = s_chunk_overlap

        self.d_chunk_ind = d_chunk_ind
        self.d_chunk_max_ind = d_chunk_max_ind

        if self.d_chunk_max_ind == 0:
            self.d_chunk_size = len(self.data_df)
        else:
            self.d_chunk_size = (len(self.data_df) // self.d_chunk_max_ind) + 1

    def get_dataset_chunk(self) -> pd.DataFrame:
        if self.d_chunk_max_ind == 0:
            return self.data_df

        s = self.d_chunk_size * self.d_chunk_ind
        e = s + self.d_chunk_size

        chunk_df = self.data_df.iloc[s:e]
        return chunk_df

    def embed_dataset(self) -> torch.Tensor:
        dataset_chunk = self.get_dataset_chunk()

        dataset_embeddings = []
        for _, row in dataset_chunk.iterrows():
            if self.model.is_sixtrack:
                embedding = self.model.embed_sequence_sixtrack(
                    row["sequence"],
                    row["cds"],
                    row["splice"],
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
        out_path = get_output_filename(
            self.dataset.embedding_dir,
            self.model.short_name,
            self.dataset.dataset_name,
            self.s_chunk_overlap,
            self.d_chunk_ind,
            self.d_chunk_max_ind
        )

        np_embeddings = embeddings.detach().cpu().numpy()
        np.savez_compressed(out_path, embedding=np_embeddings)

    def merge_embeddings(self):
        all_chunks = list(range(self.d_chunk_max_ind))
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
            if int(chunk_coords[-1]) != self.d_chunk_max_ind:
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

        out_fn = get_output_filename(
            self.dataset.embedding_dir,
            self.model.short_name,
            self.dataset.dataset_name,
            self.s_chunk_overlap
        )

        np.savez_compressed(out_fn, embedding=all_embeddings)

        for file in processed_files_paths:
            Path(file).unlink()
