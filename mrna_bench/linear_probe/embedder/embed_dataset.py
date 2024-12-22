import argparse
from pathlib import Path

import pandas as pd
import numpy as np

import torch

from mrna_bench.models.embedding_model import EmbeddingModel
from mrna_bench.models.model_catalog import MODEL_CATALOG
from mrna_bench.tasks.benchmark_dataset import BenchmarkDataset
from mrna_bench.tasks.dataset_catalog import DATASET_CATALOG
from mrna_bench.linear_probe.embedder.embedder_utils import get_output_filename

parser = argparse.ArgumentParser()
parser.add_argument("--model_class", type=str)
parser.add_argument("--model_version", type=str)
parser.add_argument("--dataset", type=str)
parser.add_argument("--embedding_dir", type=str)
parser.add_argument("--s_chunk_overlap", type=int, default=0)
parser.add_argument("--d_chunk_ind", type=int, default=0)
parser.add_argument("--d_chunk_max_ind", type=int, default=0)
parser.add_argument("--force_recompute", action="store_true")
args = parser.parse_args()


class DatasetEmbedder:
    def __init__(
        self,
        model: EmbeddingModel,
        dataset: BenchmarkDataset,
        embedding_output_dir: str,
        s_chunk_overlap: int = 0,
        d_chunk_ind: int = 0,
        d_chunk_max_ind: int = 0,
    ):
        self.model = model
        self.dataset = dataset
        self.data_df = dataset.data_df
        self.embedding_output_dir = embedding_output_dir
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
            self.embedding_output_dir,
            self.model.get_model_short_name(),
            self.dataset.short_name,
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
        for file in Path(self.embedding_output_dir).iterdir():
            if not file.is_file():
                continue

            file_name = file.stem
            file_name_arr = file_name.split("_")

            if file_name_arr[0] != self.dataset.short_name:
                continue
            if file_name_arr[1] != self.model.get_model_short_name():
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
            self.embedding_output_dir,
            self.model.get_model_short_name(),
            self.dataset.short_name,
            self.s_chunk_overlap
        )

        np.savez_compressed(out_fn, embedding=all_embeddings)

        for file in processed_files_paths:
            Path(file).unlink()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_class = MODEL_CATALOG[args.model_class]
    model: EmbeddingModel = model_class(args.model_version, device)

    dataset_class = DATASET_CATALOG[args.dataset]
    dataset: BenchmarkDataset = dataset_class()

    out_fn = get_output_filename(
        args.embedding_dir,
        model.get_model_short_name(),
        dataset.short_name,
        args.s_chunk_overlap,
        d_chunk_ind=args.d_chunk_ind,
        d_chunk_max_ind=args.d_chunk_max_ind
    )

    embedder = DatasetEmbedder(
        model=model,
        dataset=dataset,
        embedding_output_dir=args.embedding_dir,
        s_chunk_overlap=args.s_chunk_overlap,
        d_chunk_ind=args.d_chunk_ind,
        d_chunk_max_ind=args.d_chunk_max_ind
    )

    if Path(out_fn + ".npz").exists() and not args.force_recompute:
        print("Embedding already computed.")
    else:
        embeddings = embedder.embed_dataset()
        embedder.persist_embeddings(embeddings)

    if args.d_chunk_max_ind != 0:
        embedder.merge_embeddings()
